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

照片、2D、3D 只是同一個 Interface 底下三種 Environment 實作：

| Stage | Environment | Action | Agent | 重點 |
| --- | --- | --- | --- | --- |
| 1 | `ImageEnvironment` | `(x, y, size, rotation)` 放置 | Random / Greedy | 沒有物理，先驗證「受限 API 也能影響 YOLO」 |
| 2 | `Physics2DEnvironment` | `(dx, dy)` 推力 | Random / Greedy / **PPO** | 加入速度上限、碰撞、邊界、障礙物 |
| 3 | `Physics3DEnvironment` | `(dx, dy, dz)` 推力 | Random / Greedy / PPO | 只換 physics 與 renderer，Agent 與 reward 完全不動 |

---

## 快速開始

```powershell
# 1. 建立環境（uv 會自動建立 .venv 並安裝 torch(CPU) / ultralytics / SB3）
uv sync --extra dev

# 2. 專案健康檢查：三個 Environment、存取規則、Agent 都能跑（不需 YOLO、幾秒鐘）
uv run python scripts/check_project.py

# 3. 準備素材：下載來源照片 + 用 YOLOv8-seg 對它做實例分割，去背切出 target sprite
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
├── configs/default.yaml 所有超參數的單一來源
├── scripts/             check / assets / run_stage1-3 / run_all / make_report / build_notebooks
├── tests/               介面契約、封閉性、物理、reward、agent 測試
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
- **target 是真正去背的物件**，不是 bounding-box 矩形裁圖：
  `scripts/prepare_assets.py` 用 YOLOv8-seg 對來源照片做實例分割，把該物件的
  segmentation mask 當作 alpha channel，存成 `assets/target_sprite.png`。
  三個 stage 共用同一張 sprite，所以「target 這個物件」在 Stage 1/2/3 裡
  形狀一致、邊界就是它自己的輪廓，攻擊者只能遮擋它，不能把像素畫進它的輪廓裡。
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
