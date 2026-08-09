# Project Prompt：受限環境接口下的物理視覺對抗攻擊

你是一名熟悉 Python、PyTorch、Reinforcement Learning、Computer Vision 與物理模擬的研究型程式設計師。

請協助我實作一個**小型、可逐階段完成的研究專題**：

> **在受限環境接口（Environment Interface）下，使用攻擊 Agent 對視覺模型進行物理可行的對抗攻擊。**

核心概念不是直接修改圖片，而是讓攻擊 Agent 像一個只能透過 API 操作世界的玩家：

```text
                 ┌─────────────────────────┐
                 │       Environment       │
                 │                         │
                 │  World                  │
                 │  Physics                │
                 │  Victim Model           │
                 │  Reward                 │
                 └───────────┬─────────────┘
                             │
                    Environment API
                             │
                 ┌───────────┴───────────┐
                 │                       │
             Observation               Action
                 │                       │
                 ▼                       ▼
                 ┌─────────────────────────┐
                 │      Attack Agent       │
                 │                         │
                 │ RL / PPO / Search       │
                 └─────────────────────────┘
```

---

# 一、最重要的設計原則

整個專案必須遵守以下規則。

## Rule 1：Agent 不得直接存取 Environment 內部狀態

Agent 不可以取得：

- target 的真實座標
- obstacle 的真實座標
- physics engine 的內部狀態
- ground-truth bounding box
- YOLO 的內部 feature
- YOLO 的 gradient
- YOLO 的 model parameters
- 未經 Environment API 公開的任何資訊

例如以下做法禁止：

```python
env.target.x
env.target.y
env.world.objects
env.yolo.model
env.internal_image
```

Agent 只能使用：

```python
observation = env.observe()
```

---

# 二、Agent 與 Environment 的唯一溝通方式

所有 Environment 都必須實作統一介面：

```python
class BaseEnvironment:

    def reset(self):
        ...

    def observe(self):
        ...

    def step(self, action):
        ...

    def action_space(self):
        ...

    def observation_space(self):
        ...
```

其中：

### `reset()`

建立新的 episode。

### `observe()`

只回傳 Agent 被允許知道的資訊。

### `step(action)`

接受 Agent 的 action，經過 Environment 驗證與執行後，回傳：

```python
observation
reward
terminated
truncated
info
```

### `action_space()`

描述 Agent 可以執行的操作。

---

# 三、Agent 必須是 Environment-Agnostic

Attack Agent 不應該知道自己目前處於：

- Image Environment
- 2D Physics Environment
- 3D Physics Environment

Agent 只知道：

```text
Observation
    ↓
Agent
    ↓
Action
    ↓
Environment
    ↓
Observation + Reward
```

因此應盡量讓：

```python
agent.act(observation)
```

與環境實作解耦。

---

# 四、Reward 必須由 Environment 提供

Agent 不可以自行計算 YOLO confidence。

例如禁止：

```python
confidence = yolo(image)
reward = -confidence
```

如果這段邏輯存在，應放在 Environment：

```python
observation, reward, terminated, truncated, info = env.step(action)
```

Environment 內部可以：

```text
Action
 ↓
World update
 ↓
Render
 ↓
Victim Model
 ↓
Evaluation
 ↓
Reward
```

但 Agent 只能收到最後的 reward。

---

# 五、Victim Model

第一階段使用 YOLO 作為 victim model。

YOLO 必須：

- 預訓練
- frozen
- 不參與 RL training
- Agent 不可直接存取

Environment 負責呼叫 YOLO。

概念：

```text
Image / Rendered Scene
        ↓
       YOLO
        ↓
   Detection Result
        ↓
     Evaluation
        ↓
      Reward
```

第一版不需要訓練 YOLO。

---

# 六、整體開發分成三個階段

整個專案必須循序漸進：

```text
Stage 1
Photo
  ↓
Image Environment
  ↓
Attack
  ↓
YOLO
```

↓

```text
Stage 2
Photo / Scene
  ↓
2D Environment
  ↓
2D Physics
  ↓
Attack Agent
  ↓
YOLO
```

↓

```text
Stage 3
3D Scene
  ↓
3D Physics
  ↓
Attack Agent
  ↓
YOLO
```

最重要的是：

**不能把前一階段整個推翻重寫。**

三個階段必須共享：

- BaseEnvironment
- Attack Agent interface
- Reward interface
- Victim Model interface
- Experiment / Evaluation interface

只有 Environment 的具體實作逐步增加能力。

---

# 七、Stage 1：Photo Environment

## 目標

先不加入物理模擬，也不要求 RL。

目標是驗證：

> Agent 是否可以透過有限的 Environment API，找到可以降低 YOLO 偵測效果的操作。

輸入：

```text
photo.jpg
```

Environment 將圖片包裝成：

```text
ImageEnvironment
```

---

## Stage 1 Action

設計一組簡單、可控制的 action。

例如：

```python
{
    "x": ...,
    "y": ...,
    "size": ...,
    "rotation": ...
}
```

Action 必須經過 Environment 驗證：

- x 不得超過圖片範圍
- y 不得超過圖片範圍
- size 必須在合法範圍
- rotation 必須在合法範圍

Agent 不得直接修改圖片。

---

## Stage 1 Environment

流程：

```text
Agent
 ↓
action
 ↓
ImageEnvironment.step()
 ↓
驗證 action
 ↓
產生 attacked image
 ↓
YOLO
 ↓
evaluation
 ↓
reward
 ↓
Agent
```

---

## Stage 1 Agent

先不要使用 PPO。

先實作：

1. Random Search
2. Greedy / simple optimization

用來建立 baseline。

例如：

```text
Random
 ↓
100 / 500 / 1000 次 action
 ↓
選 reward 最好的結果
```

---

## Stage 1 評估

至少記錄：

- 原始 detection confidence
- 攻擊後 confidence
- confidence decrease
- attack success rate
- action cost

產生簡單圖表。

例如：

```text
Original confidence
        ↓
Attack confidence
        ↓
Confidence reduction
```

Stage 1 完成的判定：

> 能夠透過 Environment API 產生對 YOLO 有影響的 attacked image。

---

# 八、Stage 2：2D Physics Environment

Stage 1 成功後，建立：

```python
Physics2DEnvironment(BaseEnvironment)
```

不要修改 Agent、Victim Model、Reward 的基本介面。

---

## 2D World

建立簡單 2D 世界：

```text
┌─────────────────────────┐
│                         │
│          TARGET         │
│            ●            │
│                         │
│       █████             │
│                         │
│  ATTACKER               │
│      ●                  │
│                         │
└─────────────────────────┘
```

至少包含：

- target
- attacker object
- obstacle
- boundary

---

## Physics

第一版只需要：

- position
- velocity
- collision
- boundary
- movement limit

不要一開始加入複雜物理。

例如：

```text
maximum velocity
maximum movement per step
collision
```

即可。

---

# 九、Stage 2 Action

Agent 不可以直接設定：

```python
object.x = 100
object.y = 200
```

而是只能：

```python
action = {
    "dx": ...,
    "dy": ...
}
```

Environment 再根據 physics：

```text
requested action
       ↓
constraint check
       ↓
physics simulation
       ↓
actual world state
```

因此如果 Agent 嘗試穿過牆：

```text
Agent
 ↓
move right
 ↓
Environment
 ↓
collision
 ↓
不能穿牆
```

Agent 無法繞過這個限制。

---

# 十、Stage 2 Observation

Observation 也必須受到限制。

推薦第一版：

```text
camera image
+
有限的公開資訊
```

例如：

```python
observation = {
    "image": rendered_camera,
    "last_action_success": ...,
}
```

不要直接提供：

```python
target_position
ground_truth_bbox
yolo_confidence
internal_physics_state
```

除非明確把它定義成環境允許的 observation。

---

# 十一、Stage 2 RL

此時加入 PPO。

使用：

- PyTorch
- Stable-Baselines3
- Gymnasium

流程：

```text
Observation
     ↓
    PPO
     ↓
   Action
     ↓
Physics2DEnvironment
     ↓
Physics
     ↓
Renderer
     ↓
YOLO
     ↓
Reward
     ↓
PPO
```

---

# 十二、Stage 2 Reward

沿用 Stage 1 的 reward 原則。

例如：

```text
attack effectiveness
-
movement cost
-
invalid action penalty
```

不要因為進入 2D 就重新設計整個 reward。

Reward 必須維持「攻擊效果越好越高」的基本意義。

---

# 十三、Stage 2 實驗

至少比較：

```text
Random
Greedy
PPO
```

並比較：

- attack success rate
- average reward
- YOLO confidence reduction
- movement cost
- episode length

再加入 obstacle：

```text
No obstacle
vs
Obstacle
```

驗證：

> 在受到物理限制後，Agent 是否仍能找到有效攻擊策略？

---

# 十四、Stage 3：3D Environment

Stage 2 穩定後才實作。

建立：

```python
Physics3DEnvironment(BaseEnvironment)
```

核心要求：

**不要修改 Agent 的核心介面。**

只將：

```text
2D Physics
```

替換成：

```text
3D Physics
```

---

# 十五、3D Environment

可以選擇：

- Godot
- Unity
- PyBullet
- MuJoCo
- 其他適合的 3D physics framework

若希望保持 Python / Notebook 友善，可以優先考慮容易從 Python 控制的方案。

3D 場景至少包含：

- camera
- target
- attacker object
- obstacles
- environment boundary

---

# 十六、3D Action

2D：

```python
{
    "dx": ...,
    "dy": ...
}
```

3D：

```python
{
    "dx": ...,
    "dy": ...,
    "dz": ...
}
```

可逐步加入：

```text
rotation
velocity
object interaction
```

但不要一次全部加入。

---

# 十七、Notebook 實作方式

整個專案優先使用：

```text
.ipynb
```

方便：

- 展示圖片
- 展示環境
- 即時觀察 training
- 畫 reward curve
- 比較不同方法
- 逐步執行

建議 notebook：

```text
01_setup.ipynb
02_stage1_image.ipynb
03_stage2_2d.ipynb
04_stage3_3d.ipynb
05_evaluation.ipynb
```

---

# 十八、Notebook 內容規範

每個 Notebook 必須有：

### Section 1：Setup

```python
import ...
```

### Section 2：Environment

展示 Environment API。

### Section 3：Visualization

顯示目前環境。

### Section 4：Baseline

執行 Random / Greedy。

### Section 5：Attack Agent

執行 RL。

### Section 6：Evaluation

比較結果。

### Section 7：Visualization

畫：

- reward curve
- confidence curve
- attack success rate
- example attacks

---

# 十九、程式架構

建議：

```text
project/
│
├── notebooks/
│   ├── 01_setup.ipynb
│   ├── 02_stage1_image.ipynb
│   ├── 03_stage2_2d.ipynb
│   ├── 04_stage3_3d.ipynb
│   └── 05_evaluation.ipynb
│
├── environments/
│   ├── base.py
│   ├── image_env.py
│   ├── physics2d_env.py
│   └── physics3d_env.py
│
├── agents/
│   ├── base.py
│   ├── random_agent.py
│   ├── greedy_agent.py
│   └── ppo_agent.py
│
├── models/
│   └── yolo_victim.py
│
├── reward/
│   └── attack_reward.py
│
├── rendering/
│   ├── image_renderer.py
│   ├── renderer_2d.py
│   └── renderer_3d.py
│
├── evaluation/
│   └── metrics.py
│
└── configs/
    └── default.yaml
```

---

# 二十、設計上的重要要求

## Requirement 1

不要一開始就實作 3D。

必須先完成：

```text
Image → 2D → 3D
```

每個階段都必須可以獨立執行。

---

## Requirement 2

不要一開始就使用 PPO。

Stage 1：

```text
Random / Greedy
```

Stage 2：

```text
Random / Greedy / PPO
```

Stage 3：

```text
PPO
```

---

## Requirement 3

不要直接修改圖片作為 Agent 的操作。

Agent 只能：

```text
Action
 ↓
Environment
```

Environment 才能：

```text
Action
 ↓
World
 ↓
Render
```

---

## Requirement 4

Agent 不得直接取得 victim model 的資訊。

例如不能：

```python
agent.get_yolo_confidence()
```

只能：

```python
observation, reward, done, info = env.step(action)
```

---

## Requirement 5

Environment 必須負責所有限制。

包括：

- action validity
- boundary
- collision
- movement limit
- physics
- rendering
- victim model
- reward

Agent 不可以繞過這些限制。

---

# 二十一、優先順序

這是一個小型專題，因此請遵守：

```text
P0
Environment API
       ↓
P1
Image Environment
       ↓
P2
YOLO evaluation
       ↓
P3
Random attack
       ↓
P4
2D Physics
       ↓
P5
PPO
       ↓
P6
3D
```

如果時間不足：

### 最低完成版本

```text
Image Environment
+
YOLO
+
Random / Greedy
```

### 建議完成版本

```text
Image Environment
+
2D Physics Environment
+
YOLO
+
PPO
```

### 完整版本

```text
Image
 ↓
2D
 ↓
3D
```

---

# 二十二、最終實驗問題

最終不要只回答：

> 「攻擊成功了嗎？」

而是回答：

### Q1

Agent 是否能透過有限 API 影響 victim model？

### Q2

加入物理限制後，攻擊成功率如何改變？

### Q3

PPO 是否優於 Random / Greedy？

### Q4

加入 obstacle 後，Agent 是否能適應？

### Q5

從 2D 擴展到 3D 時，同一套 Agent / API 設計是否仍然成立？

---

# 二十三、最終展示

最後至少展示以下內容：

```text
Original Scene
      ↓
Agent Action
      ↓
Environment
      ↓
Attacked Scene
      ↓
YOLO Result
```

並比較：

```text
                Original    Attacked
Confidence        0.XX        0.XX
Success            ✓           ✗
Cost               -           XX
```

再展示：

```text
Random
Greedy
PPO
```

三種方法的比較。

---

# 二十四、實作原則

請不要過度工程化。

這是一個小型研究題目，不需要：

- 大型分散式 RL
- 複雜多 agent
- 自行從零實作 YOLO
- 自行從零實作 PPO
- 複雜 3D 引擎
- 大型 dataset

優先建立一個：

> **小、完整、可驗證、可逐步擴充的 prototype。**

每完成一個階段，都必須能產生可展示的實驗結果。

---

# 最終核心架構

整個專案最終應該呈現：

```text
                    ┌──────────────────────┐
                    │    Attack Agent      │
                    │                      │
                    │ Random / Greedy /    │
                    │ PPO                  │
                    └──────────┬───────────┘
                               │
                     Environment API
                               │
              ┌────────────────┴────────────────┐
              │                                 │
        Observation                         Action
              │                                 │
              ▼                                 ▼
      ┌────────────────────────────────────────────┐
      │                 Environment                 │
      │                                            │
      │  ImageEnv → Physics2DEnv → Physics3DEnv   │
      │                                            │
      │  ┌──────────┐  ┌─────────┐  ┌──────────┐ │
      │  │ Physics  │  │Renderer │  │ YOLO     │ │
      │  └──────────┘  └─────────┘  └──────────┘ │
      │                         │                 │
      │                         ▼                 │
      │                      Reward               │
      └────────────────────────────────────────────┘
```

**核心原則只有一句：**

> **Attack Agent 不直接攻擊模型，也不直接操作世界；它只能透過 Environment 提供的 Observation / Action / Reward Interface 與環境互動。**

照片、2D、3D 都只是這個 Interface 底下不同的 Environment 實作。

請依照上述設計實作，並採取「先完成最小可運行版本，再逐步增加功能」的方式。任何階段如果功能複雜，優先保留 Interface 與實驗流程，而不是增加功能導致整個專案無法完成。


# 記得留下實驗數據和過程 若可以的話寫一下文件和做好檢查確保整個專案可動