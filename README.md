# CrescentMoon — 受限環境接口下的物理視覺對抗攻擊

攻擊 Agent 不直接修改圖片、不碰 victim model、不讀世界狀態。
它只能像一個玩家一樣，透過 Environment 提供的 **Observation / Action / Reward** 介面和世界互動。

```text
                 ┌─────────────────────────┐
                 │       Environment       │
                 │  World · Physics        │
                 │  Renderer · YOLO        │
                 │  Reward                 │
                 └───────────┬─────────────┘
                             │
                    Environment API
                 reset / observe / step
              action_space / observation_space
                             │
                 ┌───────────┴───────────┐
             Observation               Action
                 │                       │
                 ▼                       ▼
                 ┌─────────────────────────┐
                 │      Attack Agent       │
                 │  Random / Greedy / PPO  │
                 └─────────────────────────┘
```

照片、2D、3D 只是同一個 Interface 底下三種 Environment 實作。三個 stage 的場景
都是從同一個**物件圖庫**（`assets/objects/`，見下）組成的，差別在於場景由
幾個物件組成、以及攻擊者怎麼移動：

| Stage | Environment | 場景 | Action | Agent | 重點 |
| --- | --- | --- | --- | --- | --- |
| 1 | `ImageEnvironment` | 單一物件（圖庫挑一個當 target） | `(x, y, size, rotation)` 放置 | Random / Greedy | 沒有物理，先驗證「受限 API 也能影響 YOLO」 |
| 2 | `Physics2DEnvironment` | 多物件組成（target + obstacle，皆為真實去背裁圖） | `(dx, dy)` 推力 | Random / Greedy / **PPO** | 加入速度上限、碰撞、邊界、障礙物 |
| 3 | `Physics3DEnvironment` | 同 Stage 2，換成 3D 場景 | `(dx, dy, dz)` 推力 | Random / Greedy / PPO | 只換 physics 與 renderer，Agent 與 reward 完全不動 |

---

## 快速開始

```powershell
# 1. 建立環境（uv 會自動建立 .venv 並安裝 torch(CPU) / ultralytics / SB3）
uv sync --extra dev

# 2. 專案健康檢查：三個 Environment、存取規則、Agent 都能跑（不需 YOLO、幾秒鐘）
uv run python scripts/check_project.py

# 3. 準備素材：下載來源照片，用 YOLOv8-seg 對每張照片做實例分割，
#    把每個夠完整、夠有信心的物件去背切出來，組成 assets/objects.json 圖庫
uv run python scripts/prepare_assets.py

# 4. 逐階段跑實驗
uv run python scripts/run_stage1.py
uv run python scripts/run_stage2.py
uv run python scripts/run_stage3.py

# 5. 產生報告（docs/RESULTS.md）
uv run python scripts/make_report.py
```

一次跑完全部：

```powershell
uv run python scripts/run_all.py            # 完整版
uv run python scripts/run_all.py --quick    # 小預算煙霧測試（幾分鐘）
```

單元測試（全部用確定性的 stub victim，不需要網路或權重）：

```powershell
uv run pytest
```

Notebook（會讀 `results/` 裡已經跑完的資料）：

```powershell
uv run jupyter lab notebooks/
```

---

## 核心規則與它的執行方式

**Rule 1：Agent 不得直接存取 Environment 內部狀態。**

這在本專案不是註解，而是執行期保證。`environments/sealed.py` 的 `seal(env)`
只開放五個方法，其他任何屬性存取都會丟 `EnvironmentAccessError`：

```python
api = seal(env)
api.step(action)      # ok
api._victim           # EnvironmentAccessError
api.render_human()    # EnvironmentAccessError
api.pop_telemetry()   # EnvironmentAccessError
```

`evaluation/runner.py` 裡的實驗迴圈**永遠**透過 `seal(env)` 呼叫環境，
所以任何違規都會在跑實驗時炸掉，而不是安靜地產生一個作弊的 Agent。

其他配套設計：

- `AttackAgent.act(observation)` 與 `observe_step(observation, reward, terminated, truncated)`
  的簽章裡**沒有 `info`**，Agent 拿不到任何診斷資料。
- `step()` 回傳的 `info` 只有 `step / action_valid / action_applied / last_action_success`，
  沒有 confidence、沒有 bbox、沒有座標（有測試在守這件事）。
- Victim model 由 Environment 建構並持有，`eval()` + `requires_grad_(False)`，
  不參與訓練，也不會被交給 Agent。
- 畫圖用的特權資料走 `env.pop_telemetry()` / `env.render_human()` / `env.victim_report()`，
  這三個都被 `seal()` 擋住。

---

## 專案結構

```text
CrescentMoon/
├── notebooks/           01_setup / 02_stage1 / 03_stage2 / 04_stage3 / 05_evaluation
├── environments/
│   ├── base.py          BaseEnvironment：整個專案唯一的介面
│   ├── spaces.py        BoxSpace / ImageSpace / DictSpace（不依賴 gym）
│   ├── sealed.py        seal()：把 Rule 1 變成執行期保證
│   ├── world.py         World / WorldObject：場景由物件組成，攻擊只能動自己的物件
│   ├── physics.py       共用物理核心（2D / 3D 同一份程式）
│   ├── image_env.py     Stage 1
│   ├── physics2d_env.py Stage 2
│   ├── physics3d_env.py Stage 3
│   └── gym_adapter.py   唯一依賴 RL framework 的地方
├── agents/              base / random / greedy / ppo（全部 environment-agnostic）
├── models/              victim 介面、YOLO victim、離線用的 stub victim
├── reward/              attack_reward.py：三個 stage 共用同一份 reward
├── rendering/           image / 2D / 3D renderer
├── evaluation/          runner（實驗迴圈）、metrics、plots、experiment、report
├── configs/
│   ├── default.yaml      所有超參數的單一來源
│   ├── loader.py          從設定檔組出三個 stage 的 Environment
│   └── objects.py         ObjectLibrary：讀 assets/objects.json 的小型索引
├── assets/
│   ├── source/            來源照片（只餵給分割模型，任何 Environment 都不會直接渲染它）
│   ├── objects/            去背裁圖，每個真實物件一張 PNG
│   └── objects.json        圖庫索引（id / class / confidence / 來源照片）
├── scripts/             check / assets / run_stage1-3 / run_all / make_report / build_notebooks
├── tests/               介面契約、封閉性、物理、reward、agent、圖庫測試
├── results/             實驗數據（json / csv / png / 訓練好的 policy）
└── docs/                DESIGN.md（設計說明）、RESULTS.md（自動產生的實驗報告）
```

---

## Reward

三個階段共用同一個定義（`reward/attack_reward.py`）：

```text
reward = w_conf · (baseline_confidence − current_confidence)
       − w_move · movement_cost
       − w_invalid · invalid_action
       + success_bonus   （confidence < success_threshold 時給一次）
```

Agent 永遠只拿到最後那個純量。confidence 是 Environment 自己算的
（`render → victim.detect → best_matching(baseline_bbox) → reward`），
Agent 沒有任何方式取得它。

---

## 已知限制

- **三個 stage 的場景都是**`environments/world.py` 裡一個由 `WorldObject`
  （background / target / obstacle / attacker）組成的 `World`：每個物件有
  自己的位置與貼圖，攻擊者的 action 只能移動自己那個物件，物理與 renderer
  都不會、也無法去改 target 或 obstacle 的像素——這是為了符合「現實攻擊以
  物件為單位，無法跨物件著色」而做的結構性保證，細節見 `docs/DESIGN.md` 第 5 節。
- **場景由一個共用的物件圖庫組成**，不是單一寫死的 sprite：
  `scripts/prepare_assets.py` 用 YOLOv8-seg 對每張來源照片做實例分割，
  把每個物件自己的 segmentation mask 當作 alpha channel 去背裁出來，
  存進 `assets/objects/<class>_<n>.png`，並記錄進 `assets/objects.json`
  （id、class、confidence、來源照片）。`configs/objects.py` 的
  `ObjectLibrary` 負責用 id（`"person_0"`）或 class 名稱（`"person"`，
  取信心最高的那個）查詢。
  - **Stage 1** 只從圖庫挑一個物件當攻擊目標（`stage1.target_object`）——
    場景就是「這一個物件 + 攻擊者」，符合「先驗證單一物件是否能被攻擊」。
  - **Stage 2/3** 的場景是圖庫裡多個物件組成的：target 之外，
    `obstacles` 清單裡的每一項也是圖庫裡的一個真實物件（目前是另一個人物
    `person_1`，見下），以自己的 center + 顯示高度放進場景，而不是純色矩形。
  - 每個物件在 Stage 1/2/3 裡都是同一張裁圖，形狀就是它自己的輪廓，
    攻擊者只能遮擋它，不能把像素畫進它的輪廓裡。
- **圖庫會過濾「不完整」的物件，其中一種過濾是人工的**：
  - 被照片邊框切掉（四邊都會檢查——例如半身照下半身缺失，或站姿完整但
    伸出去的手被照片邊緣切斷）用幾何邊界檢查擋掉；嚴重遮擋、只剩一小條
    的 instance 通常信心也偏低，用信心門檻（0.6）擋掉——這兩種是自動判斷。
  - 但「被畫面中站在它前面的東西挖了洞」（例如 bus.jpg 裡三個行人站在
    巴士前面，巴士的分割遮罩因此缺了三塊人形）**沒有**用幾何方法自動偵測：
    試過的一種寫法（遮罩內被自己輪廓完全包住的背景洞，去跟其他 instance
    的真實遮罩比對）並不可靠，缺了三個人形洞的巴士算出來的分數，反而比
    人工確認完整的人物還低。改成人工看過裁圖後手動排除（見
    `scripts/prepare_assets.py::MANUAL_EXCLUDE`），誠實承認這一步是人工
    判斷。目前的圖庫因此只剩兩個人物物件（`person_0`、`person_1`），沒有
    車輛——巴士被上述兩種問題同時排除了，Stage 2/3 的 obstacle 也就改用
    另一個人物代替。細節見 `docs/DESIGN.md` 第 5 節。
- **攻擊者的尺寸不能超過目標物件自己的邊界**：patch 邊長永遠是
  `patch_max_frac`（Stage 1）/ `patch_frac`（Stage 2）/ `patch_world_frac`
  （Stage 3）乘上 target 自己的「窄度」，三個設定值都限制在 `(0, 1]`，
  建構環境時就會驗證、超過直接 `ValueError`。這個「窄度」量測過兩版：
  - 第一版用 target 裁圖的 bounding box 寬度，結果 patch 疊在人物腰部這種
    比 bounding box 窄的部位時會明顯露出來——bounding box 寬度是整個人
    最寬那一橫排（例如交叉的雙臂）決定的，不代表其他高度的寬度。
    改用 `silhouette_min_span()`：掃描裁圖每一橫排的 alpha，取**最窄**
    那一排的跨距，才是不論疊在哪個高度都不會露出來的量測。
  - Stage 1 的 patch 還會旋轉，而旋轉正方形的外接框最多可以脹大 √2 倍
    （45° 時最大），所以算 size 上限前會先把這個「窄度」除以旋轉安全係數，
    確保**旋轉到任何角度**都不會超出 target 的邊界。
  細節見 `docs/DESIGN.md` 第 5 節「攻擊不能超出目標物件自己的邊界」。
- 3D 使用自寫的 pinhole camera + billboard renderer，不是 PyBullet/MuJoCo。
  這是刻意的取捨（prompt 第 24 節要求不要複雜 3D 引擎）：深度排序與遮擋是真的，
  但沒有剛體動力學、沒有旋轉、沒有陰影。
- 攻擊方式是「用一塊實體看板遮擋 / 干擾」，patch 的**貼圖是固定的**，
  Agent 優化的是放置與移動，不是像素。這正是「物理可行」的定義所要求的。
- PPO 的預算刻意調小（CPU 可跑完）。要更漂亮的曲線就把
  `configs/default.yaml` 裡的 `total_timesteps` 調大。
- 三個 stage 的背景都是程序生成的低紋理背景（`make_background()`），
  這讓 reward 訊號乾淨（YOLO 不會被背景裡其他物件干擾），但場景比真實街景單純。

細節見 [docs/DESIGN.md](docs/DESIGN.md)，實驗結果見 [docs/RESULTS.md](docs/RESULTS.md)。
