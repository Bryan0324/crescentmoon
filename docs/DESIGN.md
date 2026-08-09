# 設計說明

這份文件解釋「為什麼這樣寫」，特別是那些為了守住核心規則而做的取捨。

---

## 1. 唯一的介面

`environments/base.py` 定義的 `BaseEnvironment` 是整個專案的骨幹：

```python
reset(seed=None) -> Observation
observe()        -> Observation
step(action)     -> (observation, reward, terminated, truncated, info)
action_space()   -> BoxSpace
observation_space() -> DictSpace
```

三個 stage 都實作這同一個介面，所以 `evaluation/runner.py` 的實驗迴圈、
`evaluation/experiment.py` 的 Stage 2/3 driver、以及三個 agent，
在三個 stage 之間都是**同一份程式**。

Space 刻意不用 Gymnasium 的型別：核心不應該綁 RL framework。
只有 `environments/gym_adapter.py` 會做轉換，也只有 PPO 會用到它。

### Observation 的形狀

```python
{
    "image":  uint8 [obs_size, obs_size, 3],   # 低解析度的 camera view
    "vector": float32 [n],                     # step 進度、上一動是否成功、上一動的內容
}
```

三個 stage 的 observation **結構相同、維度不同**，這是 Agent 能夠
environment-agnostic 的關鍵：agent 只問 `action_space()` 要邊界，
不假設任何語意。

注意 victim 看到的是 `render_size`（512 / 384）的完整畫面，
agent 只拿到 `obs_size`（64）的縮圖。這既是「Agent 只能看到相機」的
物理直覺，也讓 PPO 在 CPU 上跑得動。

### 為什麼 vector 裡放「上一個 action」

那是 agent 自己送出去的東西，不構成資訊洩漏，但能讓 policy 有一點時序感。
真正的內部狀態（速度、座標、目標位置、confidence）一律不給。

---

## 2. Rule 1 的執行方式

三層防線：

1. **命名慣例** — 環境內部一律 `_` 開頭。
2. **`seal(env)`（`environments/sealed.py`）** — 一個只轉發五個方法的 proxy，
   其餘 `__getattr__` / `__setattr__` 直接丟 `EnvironmentAccessError`。
   實驗迴圈和 PPO 訓練都跑在這個 proxy 上。
3. **測試** — `tests/test_sealed_environment.py` 逐一嘗試存取
   `_victim` / `_renderer` / `_body` / `_baseline_bbox` / `render_human` / `pop_telemetry`…
   並要求每一個都被擋下；`tests/test_environment_api.py` 另外檢查
   observation 與 `info` 裡沒有 confidence / bbox / 座標。

特權資料不是不存在，而是走另一條路：
`pop_telemetry()`（每步的 confidence、位置、reward 分解）、
`render_human()`（全解析度畫面）、`victim_report(image)`（畫框用）。
這三個只有實驗腳本和 notebook 拿得到。

### Agent 端的對應設計

```python
def act(self, observation): ...
def observe_step(self, observation, reward, terminated, truncated): ...
```

簽章裡沒有 `info`、沒有 env handle。這是刻意的：即使有人在 `info` 裡
不小心塞了診斷資料，agent 也接不到。

---

## 3. Reward：一個定義，三個 stage

```text
reward = w_conf · confidence_drop − w_move · movement_cost
         − w_invalid · invalid_action + success_bonus
```

`RewardConfig` 的權重放在 `configs/default.yaml`，三個 stage 共用同一組。
Stage 2/3 沒有「重新設計 reward」，只是 `movement_cost` 的定義換成
「這一步實際移動的距離 / 每步移動上限」，Stage 1 則是「patch 面積佔畫面比例」——
兩者都在 `[0, 1]`，語意都是「這個攻擊在物理上有多貴」。

`invalid_action` 在 Stage 1 指「送出超出範圍的 action」，
在 Stage 2/3 額外包含「這一步被牆或邊界擋住了」——都是「你要求的事情環境沒能照做」。

### 兩個為了避免 reward hacking 的決定

**(1) Stage 2/3 的 episode 是固定長度，不會因為攻擊成功就提早結束。**
如果成功就 terminate，agent 會發現「把 confidence 壓在 threshold 上面一點點、
撐滿 40 步」拿到的累積 reward 遠大於「壓到 0 然後結束」——它會學會刻意不成功。
改成固定長度後，持續遮擋才是最優解。
因此 `EpisodeRecord.success` 不看 `terminated`，而是看 telemetry 裡
「這個 episode 中是否曾經有一步低於 threshold」。
Stage 1 每個 episode 只有一步，這個問題不存在，仍然保留 `terminated`。

**(2) 撞到世界邊界不算 invalid action，撞到障礙物才算。**
最初的版本把 `collided or hit_boundary` 都算成 invalid，結果 agent 一貼牆
就每步被扣 0.25，40 步扣 10 分，完全淹沒最大只有 0.86 的攻擊訊號
（實測 greedy 平均 reward 為 −6.8）。現在：

- `invalid` = 送出超出範圍的 action，或推進障礙物 → 進 reward 懲罰
- 碰到世界邊界 → 只反映在 observation 的 `last_action_success`，不扣分

物理已經讓它動不了了，那本身就是懲罰。

### 為什麼要追蹤 baseline bbox

場景裡可能有多個物件。如果只看「最高信心的 person」，Agent 可能把 A 遮掉、
但畫面裡的 B 讓 confidence 看起來沒降。所以 `reset()` 會記下 baseline
偵測框，之後每步都用 `Detections.best_matching(baseline_bbox, cls, iou≥0.2)`
只追同一個目標；完全追丟就記為 confidence = 0（攻擊成功）。

---

## 4. 物理

`environments/physics.py` 是 dimension-agnostic 的：`Body` / `AABB` / `integrate`
用 n 維向量寫成，2D 傳 2 維、3D 傳 3 維，**同一份程式**。
這是 Stage 3 不需要重寫 Stage 2 的直接原因。

一個 tick 做的事：

```text
v ← clip_norm(v · damping + a · dt, max_speed)
Δ ← clip_norm(v · dt, max_step)
逐軸移動：先夾到世界邊界，再檢查是否會撞進障礙物
        撞到 → 該軸回退、該軸速度歸零（可以貼牆滑動，不能穿牆）
```

範圍刻意只有 prompt 第 8 節列的那些：position、velocity、collision、
boundary、movement limit。沒有重力、沒有摩擦、沒有旋轉。

---

## 5. World：場景由物件組成，不是由參數組成

現實中的物理攻擊是**以物件為單位**的：你可以印一塊板子、舉著它、移動它，
但你不能「跨物件著色」——不能把攻擊者的像素直接畫進 target 或背景裡，
唯一能造成的跨物件效果只有**遮擋**（把一個不透明的東西擋在鏡頭前面）。

最初的 Stage 2/3 實作沒有把這件事變成結構性保證：`Renderer2D`/`Renderer3D`
把 target 的位置、obstacle 的清單、patch 的貼圖都當成建構子參數直接吃進去，
「target 不會被攻擊者的動作改到」只是因為程式*剛好*沒有寫錯，而不是
架構上不可能寫錯。`environments/world.py` 把這件事做成第一級的抽象：

```python
@dataclass
class WorldObject:
    id: str
    kind: Literal["target", "obstacle", "attacker"]
    position: np.ndarray        # 2D：像素座標；3D：世界座標（公尺）
    half_extents: np.ndarray    # 碰撞範圍 / billboard 尺寸
    sprite: Image.Image         # 這個物件自己的、固定的外觀
    movable: bool = False       # 只有 attacker 是 True

class World:
    def target(self) -> WorldObject: ...
    def obstacles(self) -> list[WorldObject]: ...
    def attacker(self) -> WorldObject: ...
```

`Physics2DEnvironment` / `Physics3DEnvironment` 在建構時把 target、每個
obstacle、attacker 各自組成一個 `WorldObject`，放進同一個 `World`。之後：

- **Physics**（`physics.py::integrate`）只操作 attacker 的 `Body`；
  obstacle 的碰撞範圍是從 `world.obstacles()` 讀出來的 AABB，本身
  唯讀，`integrate()` 不會寫回去。
- **Renderer**（`renderer_2d.py` / `renderer_3d.py`）不再認得
  「target」「obstacle」這些名字，只認得 `world.objects`：
  依序（2D 是固定 paint order；3D 是逐幀依深度排序）把每個物件自己的
  `sprite` 貼在自己的 `position`。畫某個物件，永遠只會動到那個物件自己
  配置到的像素。
- `step()` 裡唯一寫入 World 的一行是
  `self._world.attacker().position = self._body.position`——
  這是整份程式碼裡**唯一**一處會修改 WorldObject 位置的地方，
  而且只碰 attacker 自己的物件。

`tests/test_world.py` 直接斷言這個不變量：跑幾步任意動作後，
`world.target().position` 與每個 obstacle 的 position 必須和 reset 時完全相同
（`test_the_attacker_is_the_only_object_that_moves`）。

### Stage 1 為什麼沒有套用 World

Stage 1 的輸入是一張真實照片，本來就沒有被切成離散物件（沒有逐物件的
mask/depth），把它硬套進 `WorldObject` 需要先做物件分割，屬於
過度工程化（prompt 第 24 節明確要求不要）。Stage 1 的場景其實只有兩個
「物件」：**背景照片**（固定、agent 不可更動）和**攻擊者 patch**
（agent 唯一能控制的東西）。`paste_patch()` 只會寫入 patch 自己那塊
不透明的像素、貼在背景之上，不會去改背景其他像素的顏色——這已經滿足
「不能跨物件著色」，只是沒有一個顯式的 `World` 類別去描述它。

## 6. Renderer

- `rendering/image_renderer.py` — patch 貼圖生成（固定 seed 的高對比色塊 + 噪點）
  與合成；`load_sprite()` 讀取真實裁圖或退回一塊純色作為 fallback。
  patch 的**像素是固定的**，agent 優化的是放置。
- `rendering/renderer_2d.py` — world 座標就是相機像素座標。
  只認得 `World`：依 `WorldObject.paint_order`（target → obstacle →
  attacker）依序把每個物件的 sprite 貼上去。
- `rendering/renderer_3d.py` — 針孔相機在原點看向 +Z，
  投影 `u = cx + f·x/z`、`v = cy − f·y/z`；每個 `WorldObject` 是一塊
  billboard，依自己的 `position.z` 由遠到近繪製，物件本身完全不知道
  自己是「target」還是「obstacle」。

3D 沒有引入 PyBullet/MuJoCo/Godot，是因為 prompt 第 24 節明確要求不要複雜 3D 引擎，
而這個專案真正需要 3D 的地方只有一件事：**深度**。
攻擊者必須繞到目標「前面」才能遮擋，站在目標後面完全無效——這一點 billboard
渲染就足以正確表現，代價是沒有剛體動力學。

背景是程序生成的低紋理漸層。這是刻意的：如果背景放真實照片，
YOLO 會偵測到一堆其他物件，reward 訊號會被汙染。

---

## 7. Victim model

`models/yolo_victim.py`：Ultralytics YOLO，`eval()` + `requires_grad_(False)`，
只透過 `detect(image_rgb) -> Detections` 對外。
`conf_threshold` 刻意設得很低（0.05），這樣環境才看得到**漸進的**信心下降，
而不是在預設 0.25 直接掉成階梯函數。

`models/stub_victim.py`（`ColorBlobVictim`）是離線用的玩具偵測器：
偵測一個特徵顏色，信心 = 可見面積 / 基準面積。
它存在的理由有兩個——(1) 讓測試不需要網路與權重，
(2) 它的遮擋反應是單調的，所以 reward 契約可以被斷言。
它**不是**研究結果的一部分。

---

## 8. Agents

三個 agent 都只需要 `action_space()` 和純量 reward：

- `RandomAgent` — 在合法範圍內均勻取樣。誠實的下界。
- `GreedyAgent` — (1+1) hill climbing。在 Stage 1（每 episode 一步）就是
  「保留最好的放置、擾動再試」；在 Stage 2/3 變成「有效的動作就重複，
  沒效就擾動」。同一份程式，兩種語意。
- `PPOAgent` — Stable-Baselines3 PPO，透過 `GymEnvAdapter(seal(env))` 訓練。
  PPO 拿到的 API 和 random 完全一樣。

`agents/__init__.py` 對 `PPOAgent` 用 lazy import，所以 Stage 1 不需要裝 SB3。

---

## 9. 實驗流程

`evaluation/runner.py::run_episodes` 是唯一的實驗迴圈：
它持有真正的 env（拿 telemetry 畫圖），但只把 `seal(env)` 的結果餵給 agent。

`evaluation/experiment.py::run_physics_stage` 是 Stage 2 和 Stage 3
**共用**的 driver（variant × method 的兩層迴圈、訓練 PPO、畫圖、存檔）。
`scripts/run_stage2.py` 和 `run_stage3.py` 只是不同的 env factory ——
這件事本身就是 prompt Q5 的答案。

---

## 10. 刻意沒做的事

- 沒有做 white-box 梯度攻擊（違反 Rule 1 的精神）。
- 沒有訓練或微調 YOLO。
- 沒有自己實作 PPO。
- 沒有多 agent、沒有分散式 RL、沒有大型 dataset。
- 沒有把 Stage 1 的程式在 Stage 2 重寫；三個 stage 共用
  `BaseEnvironment` / `AttackAgent` / `AttackReward` / `run_episodes` / metrics。
