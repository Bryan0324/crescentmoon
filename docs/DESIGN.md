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
這一節描述的設計是在 review 中被明確指出「受限環境接口並未實現、與現實差異
過大」之後的修正版——最初的 Stage 1/2/3 實作沒有把這件事變成結構性保證。

**Stage 1（照片）原本的問題最大**：直接把一張有公車、有好幾個行人的街景照
當畫布，用 `paste_patch()` 把 patch 貼上去。target 是「YOLO 剛好在這張照片
裡信心最高的 person」，沒有自己的邊界、沒有自己的貼圖，只是一個鬆散的
`(x1,y1,x2,y2)` tuple；attacker 的落點可以自由蓋住畫面裡任何東西，
包括其他沒被追蹤的物件。**Stage 2/3 的問題比較隱性**：`Renderer2D`/
`Renderer3D` 把 target 的位置、obstacle 的清單、patch 的貼圖都當成建構子
參數直接吃進去，「target 不會被攻擊者的動作改到」只是因為程式*剛好*
沒有寫錯，而不是架構上不可能寫錯。

`environments/world.py` 把「場景由物件組成」做成第一級的抽象，
**三個 stage 現在共用同一個 World/WorldObject 定義**：

```python
@dataclass
class WorldObject:
    id: str
    kind: Literal["background", "target", "obstacle", "attacker"]
    position: np.ndarray        # 2D：像素座標；3D：世界座標（公尺）
    half_extents: np.ndarray    # 碰撞範圍 / billboard 尺寸
    sprite: Image.Image | None  # 這個物件自己的、固定的外觀
    movable: bool = False       # 只有 attacker 是 True
    rotation_deg: float = 0.0   # 只有 2D 使用

class World:
    def background(self) -> WorldObject: ...
    def target(self) -> WorldObject: ...
    def obstacles(self) -> list[WorldObject]: ...
    def attacker(self) -> WorldObject: ...
```

`ImageEnvironment` / `Physics2DEnvironment` / `Physics3DEnvironment` 在建構
時把 background、target、每個 obstacle、attacker 各自組成一個
`WorldObject`，放進同一個 `World`。之後：

- **Physics**（`physics.py::integrate`，只有 Stage 2/3 使用）只操作
  attacker 的 `Body`；obstacle 的碰撞範圍是從 `world.obstacles()` 讀出來
  的 AABB，本身唯讀，`integrate()` 不會寫回去。Stage 1 沒有物理，
  attacker 的位置由 action 直接寫入（見下）。
- **Renderer**（`renderer_2d.py` / `renderer_3d.py`）不再認得
  「target」「obstacle」這些名字，只認得 `world.objects`：
  依序（2D 是固定 paint order；3D 是逐幀依深度排序）把每個物件自己的
  `sprite` 貼在自己的 `position`。畫某個物件，永遠只會動到那個物件自己
  配置到的像素；`sprite=None`（純記帳用、沒有自己外觀）的物件會被跳過。
- 每個 stage 的 `step()` 裡唯一寫入 World 的一行都只碰 attacker 自己的物件：
  - Stage 1：`attacker.sprite/position/half_extents/rotation_deg = ...`
    （action 直接決定這一步的放置）
  - Stage 2/3：`self._world.attacker().position = self._body.position`
    （物理算完之後才寫回去）

  這是整份程式碼裡**唯一**會修改 WorldObject 的地方。

`tests/test_world.py` 直接斷言這個不變量：跑幾步任意動作後，
`world.target().position` 與每個 obstacle 的 position 必須和 reset 時完全相同
（`test_the_attacker_is_the_only_object_that_moves`，涵蓋 Stage 1/2/3 三種環境）。

### 物件圖庫：target 是真的去背，不是矩形裁圖

只把 World 結構做對還不夠——如果 target 這個物件的「sprite」其實是一個
包含背景像素的矩形裁圖，那麼「物件邊界」仍然是假的。`scripts/prepare_assets.py`
改用 `yolov8n-seg.pt`（Ultralytics 的實例分割模型）而不是一般的偵測模型，
而且不再只切一張圖，而是對每張來源照片跑分割、把**每一個**夠完整、夠有
信心的 instance 都去背切出來，組成一個小型物件圖庫：

```text
assets/
  source/            來源照片，只餵給分割模型；任何 Environment 都不會直接渲染它
  objects/           去背裁圖，每個真實物件一張 RGBA PNG
  objects.json       索引：[{id, class, file, confidence, source}, ...]
```

`configs/objects.py::ObjectLibrary` 負責查詢——用物件 id（`"person_0"`）
或 class 名稱（`"person"`，取該 class 信心最高的那個）都可以。
每張裁圖的 alpha channel 直接來自該 instance 的 segmentation mask，
不是 bounding box：物件的「邊界」就是它自己的輪廓。

**Stage 1 只從圖庫挑一個物件當攻擊目標**（`stage1.target_object`）——場景
就是「這一個物件 + 攻擊者」，對應 prompt 的「先驗證單一物件是否能被攻擊」。
**Stage 2/3 的場景由圖庫裡多個物件組成**：除了 target，`obstacles` 清單
裡的每一項也是圖庫裡一個真實物件（目前是另一個人物 `person_1`——圖庫裡
唯一通過完整性檢查的車輛只有巴士，而它被排除了，見下一節），用和 target 一樣的方式
放進場景——自己的 center + 顯示高度（`Physics2DEnvironment`/
`Physics3DEnvironment` 的 `_build_world()` 對 target 和每個 obstacle
呼叫同一套 `load_sprite() → 依高度縮放 → 建立 WorldObject` 邏輯，沒有兩套
程式碼）。離線（沒有網路/沒有 ultralytics）時退回兩張橢圓形的合成 alpha
cutout（`build_offline_library()`），而不是純色矩形，讓離線路徑練到的
仍然是「非矩形 alpha 合成、多物件組成場景」這條程式碼路徑。

### 圖庫會過濾「不完整」的物件

第一版分割出來的 instance 裡，有些明顯「不完整」——不是遮罩雜訊，是物件
本身就缺了一塊。實際檢查 bus.jpg + zidane.jpg 分割出的 8 個 instance，
發現三種不同成因，處理方式也不一樣：

**1. 被照片邊框切掉。** zidane.jpg 是半身特寫，兩位主角的 bounding box
下緣幾乎貼齊照片底部（`y2 = 712/720`、`709/720`）——不是分割演算法漏切
了腿，是那段畫面在照片裡根本不存在。用這種裁圖當「物件」放進合成場景，
會是一個懸空的半身軀幹。更隱蔽的例子：bus.jpg 裡信心最高的 person
（0.878）bbox 右緣正好卡在照片右邊界（`x2 = 810/810`）——伸出去的那隻手
被裁掉了，乍看仍是完整的人形，實際上手臂不見了。這類問題用一個**四邊都檢查**
的邊界過濾解決（`BORDER_MARGIN = 0.02`：bbox 任一邊落在照片寬/高 2% 以內
就丟棄）——第一版只檢查上下邊界，這個右手被裁掉的案例就是後來才發現漏掉
左右邊的教訓。

**2. 整體被嚴重遮擋，只剩一小塊。** bus.jpg 裡有一個 person instance
信心只有 0.412（其他 person 都在 0.84 以上），去背後只剩一小條軀幹/
手臂。這類交給信心門檻擋住：`MIN_CONFIDENCE = 0.6`。

**3. 被畫面中「站在它前面」的東西挖了一個洞。** bus.jpg 裡有三個人站在
巴士前面，巴士的分割遮罩因此被挖出三個人形的洞——遮罩本身是對的（那些
像素真的屬於人、不屬於巴士），但拿來當一個獨立「物件」使用時，觀感上明顯
缺了一塊。這一類**沒有**用幾何方法自動偵測：我們試過一種作法（把遮罩內
「被自己輪廓完全包住、摸不到邊界」的背景區域，跟其他 instance 的真實遮罩
做比對，只有兩者重疊時才算真的被遮擋），結果並不可靠——在 bus.jpg 上，
明顯缺了三個人形洞的巴士算出來的分數（18.8%），反而比人工確認完整的人物
（16.4%～36.6%）還要低：一個人交叉雙臂、雙腿分開產生的「自己身體圍起來的
背景空隙」，跟真的被別的物件遮擋，在純幾何上很難分辨。要可靠判斷「這看起來
像不像一個完整物件」需要真正的影像理解，對一個小型專題的素材前處理腳本
是不成比例的投入，所以改用 `MANUAL_EXCLUDE`——人工看過裁圖後，把
`(來源照片, class, confidence)` 記錄下來直接排除，誠實地承認這一步是
人工判斷，而不是假裝有一個自動、可靠的偵測器。

三個規則套用後，bus.jpg + zidane.jpg 的 8 個 instance 只剩 2 個都通過人工
檢查的完整人物：`person_0`（0.860）、`person_1`（0.841）。原本信心最高的
那個 person（0.878，手被裁掉）、巴士（0.838，MANUAL_EXCLUDE，也剛好同時
被邊界過濾擋住）、zidane.jpg 的兩個人物與領帶、以及那個 0.412 的人，都
不會進圖庫。目前圖庫裡沒有車輛類別，Stage 2/3 的 obstacle 因此也改用
`person_1`——見下一節。

### 攻擊不能超出目標物件自己的邊界

第二個在 review 中被指出的問題：即使 target 已經是真的去背裁圖，
attacker 這個物件本身仍然可以比 target 大很多。舊版 Stage 1 允許 patch 邊長
到 canvas 邊長的 45%（在 512px 畫布上約 230px），但 target 裁圖本身只有
~72px 寬——攻擊者可以貼一塊比目標寬三倍的板子，這已經不是「遮擋物件」，
更接近「用一塊無關的大看板蓋住整個場景」，不符合「現實中的攻擊應當以物件
為單位」。Stage 2 也一樣：`patch_size` 曾經是固定 82px，比 target 自己
的寬度（~55px）還寬。

修法是讓 patch 的合法尺寸**永遠以 target 自己的尺寸為準，不是畫布**：

```text
target_min_dim = target 自己的某個「窄」的量測
patch 邊長上限 = patch_max_frac × target_min_dim   (patch_max_frac ≤ 1.0)
```

在 `patch_max_frac = 1.0` 時，patch 邊長恰好等於 target 較窄的那個維度，
不可能再寬過 target 本身；`ImageEnvConfig`/`Physics2DEnvConfig`/
`Physics3DEnvConfig` 建構時都會檢查 `patch_max_frac`／`patch_frac`／
`patch_world_frac` 落在 `(0, 1]`，超過就直接 `raise ValueError`——這和
Rule 1 的執行方式一樣：規則不是寫在文件裡希望大家遵守，而是程式碼結構上
不給違反的機會。

三個 stage 的差異只在**誰、何時**套用這條規則：

- **Stage 1** — size 是 agent action 的一部分，所以規則做在
  `action_space()` 的上界：`high[2] = patch_max_frac * target_min_dim`。
  Agent 無論怎麼選，`clip()` 之後的 size 永遠不可能超過 target 自己的寬度。
- **Stage 2/3** — patch 尺寸是環境設定值，不是 agent 能控制的（agent 只能
  推動它，不能改變它的大小），所以規則做在 `_build_world()` 建構時：
  `patch_frac`／`patch_world_frac` 現在取代了原本寫死的
  `patch_size`／`patch_world_size`，實際像素 / 世界單位大小是從
  target 的 `half_extents` 動態算出來的——換一張長寬比不同的 target sprite
  （例如以後改成車輛），patch 大小會自動跟著調整，不會有人忘記手動改設定
  而悄悄違反這條規則。

#### target 不是矩形：bounding box 寬度也不是安全的量測

第一版的 `target_min_dim` 用 target 裁圖的 **bounding box** 寬度（`WorldObject.half_extents`
乘 2）。這在視覺上驗證時被抓到一個問題：把 patch 直接疊在 target 的
垂直中心（也就是 Stage 2/3 常見的接觸位置）時，patch 明顯從人物的左右兩側
「露出來」。原因是人的輪廓在不同高度寬度不一樣——bounding box 的寬度是
整個人最寬的那一橫排（例如交叉的雙臂、聳起的肩膀）決定的，但腰部、雙腿之間
的實際輪廓比這個寬度窄得多。用 bounding box 寬度當上限，等於允許 patch 在
「最寬處」剛好合身，卻在其他任何比較窄的高度都放大到超出人物本身。

修法是量測**整張裁圖裡最窄的那一橫排**，而不是 bounding box：

```python
# rendering/image_renderer.py::silhouette_min_span
逐一橫排掃描 alpha channel > 127 的像素
每一排取「最左不透明像素」到「最右不透明像素」的跨距
target_min_dim = 所有橫排裡最小的跨距
```

這跟旋轉安全係數是同一種思路：用**最壞情況**而不是「大致上」的量測，
保證邊界不會在任何情況下被違反，而不是大部分情況下不會。上下各 3% 的
橫排會被排除（`edge_trim`），否則髮絲、鞋尖這種只有一兩個像素寬的
反鋸齒邊緣會讓上限塌縮到幾乎是 0。以 `person_0`（202×516px）為例：
bounding box 寬度 202px，但最窄橫排只有 59px——差距接近 3.4 倍，
如果沒有這個修正，Stage 2 疊在腰部的 patch 會明顯比實際輪廓寬。

三個 stage 都改用同一個函式：Stage 1／2 直接對像素座標的 sprite 呼叫；
Stage 3 額外乘上 `target_world_height / sprite.height` 把像素跨距換算成
世界座標的距離。

#### 旋轉會讓正方形「看起來更寬」，size 上限也要跟著扣掉這個安全邊際

只限制旋轉前的 `size` 還不夠。Stage 1 的 patch 是正方形，旋轉 θ 度之後，
它的（軸對齊）外接框邊長會變成 `size × (|cos θ| + |sin θ|)`——在 θ=45°
時最大，等於 `size × √2`（正方形有 90° 旋轉對稱，所以 45° 已經是最壞情況，
角度再大也不會更壞）。換句話說，即使 `size ≤ target_min_dim`，一個旋轉
45° 的 patch 實際佔據的範圍仍可能到 `target_min_dim × √2`，明顯超出
target 自己的邊界——這是實測 review 時真的觀察到的：一塊旋轉 61° 的
patch，即便邊長已經受 `patch_max_frac` 限制，視覺上仍明顯比 target 寬。

修法是在算 size 上限之前，先把 `target_min_dim` 除掉這個最壞情況的安全
係數：

```python
worst_theta = min(45°, max_rotation_deg)
rotation_safety = cos(worst_theta) + sin(worst_theta)   # 45° 時 = √2
size_bound_dim = target_min_dim / rotation_safety
size 上限 = patch_max_frac × size_bound_dim
```

`max_rotation_deg = 90`（預設值）涵蓋 45°，所以 `rotation_safety = √2`；
這樣不管 agent 選了哪個旋轉角度，貼上去的 patch 都保證留在 target 自己的
邊界之內。Stage 2/3 的 attacker 不旋轉（`rotation_deg` 恆為 0），不需要
這個修正。

`tests/test_world.py` 裡對應的測試：
`test_stage1_action_space_caps_size_at_the_targets_own_silhouette`、
`test_stage1_max_size_survives_worst_case_rotation`、
`test_physics_attacker_patch_never_exceeds_the_targets_own_silhouette`，
以及三個 `test_*_rejects_a_patch_*_frac_above_one`（確認 > 1.0 一定會被拒絕）。

## 6. Renderer

- `rendering/image_renderer.py` — patch 貼圖生成（固定 seed 的高對比色塊 + 噪點）
  與合成；`load_sprite()` 讀取真實去背裁圖或退回一塊有 alpha 的合成 cutout。
  patch 的**像素是固定的**，agent 優化的是放置。
- `rendering/renderer_2d.py` — world 座標就是相機像素座標。
  完全由 `World` 驅動，包含背景：background/target/obstacle/attacker
  都是 `world.objects` 裡的一個物件，依 `WorldObject.paint_order`
  （background → target → obstacle → attacker）依序把每個物件的 sprite
  貼上去，2D 專屬的 `rotation_deg` 在貼之前套用。`ImageEnvironment`
  （Stage 1）與 `Physics2DEnvironment`（Stage 2）共用這一個 renderer。
- `rendering/renderer_3d.py` — 針孔相機在原點看向 +Z，
  投影 `u = cx + f·x/z`、`v = cy − f·y/z`；target/obstacle/attacker 都是
  `WorldObject`，每個是一塊 billboard，依自己的 `position.z` 由遠到近繪製，
  物件本身完全不知道自己是「target」還是「obstacle」。背景（天空/地面漸層）
  仍由 renderer 自己生成並固定畫在最底層——它不是攻擊者能佔據或遮擋的
  「東西」，只是相機背後的環境，所以沒有必要也做成 `WorldObject`。

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
