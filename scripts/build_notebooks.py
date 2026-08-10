"""Generate the five project notebooks.

Keeping the notebooks generated (rather than hand-edited) means the code inside
them is the same code the scripts run -- there is no second, drifting copy of
the experiment.

    uv run python scripts/build_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOKS = PROJECT_ROOT / "notebooks"

BOOTSTRAP = """\
# Section 1: Setup
import sys, pathlib

ROOT = pathlib.Path.cwd()
if not (ROOT / "configs" / "default.yaml").exists():
    ROOT = ROOT.parent          # running from notebooks/
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from configs.loader import load_config, build_victim, resolve

cfg = load_config()
print("project root:", ROOT)
print("victim config:", cfg["victim"])
"""


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook(*cells: nbf.NotebookNode) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=list(cells))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


# ----------------------------------------------------------------------
def build_01_setup() -> nbf.NotebookNode:
    return notebook(
        md(
            """
# 01 · Setup — the Environment API

This project has exactly one rule:

> **The attack agent never touches the world, the victim model, or the pixels.
> It only calls `reset()`, `observe()`, `step()`, `action_space()`,
> `observation_space()`.**

This notebook checks the install, loads the frozen victim, and takes a tour of
that API — including a demonstration that the "agent must not peek" rule is
enforced at runtime, not just documented.
"""
        ),
        code(BOOTSTRAP),
        code(
            """
import torch, ultralytics, gymnasium, stable_baselines3
print("torch              ", torch.__version__)
print("ultralytics        ", ultralytics.__version__)
print("gymnasium          ", gymnasium.__version__)
print("stable-baselines3  ", stable_baselines3.__version__)
"""
        ),
        md(
            """
## Section 2: Environment

`BaseEnvironment` is the single interface all three stages implement.  Note that
the victim model, the renderer and the physics are *constructor arguments of the
environment* — they are never handed to an agent.
"""
        ),
        code(
            """
from environments.base import BaseEnvironment
import inspect

print(inspect.getsource(BaseEnvironment)[:2000])
"""
        ),
        code(
            """
from configs.objects import load_library

library = load_library(resolve(cfg["assets"]["objects_index"]), resolve(cfg["assets"]["objects_dir"]))
print(f"object library: {len(library)} objects")
for object_id in library.ids():
    a = library.get(object_id)
    print(f"  {a.id:<12} class={a.cls_name:<8} conf={a.confidence:.3f}  (from {a.source})")
print("\\nStage 1 attacks a single one of these:", cfg["stage1"]["target_object"])
"""
        ),
        code(
            """
victim = build_victim(cfg)          # frozen, pretrained, owned by the environment
print("victim:", victim.name)

from configs.loader import build_stage1_env
env = build_stage1_env(cfg, victim)   # a World: background + target cutout + attacker
clean = env.clean_image()
detections = env.victim_report(clean)   # experimenter-only call -- an agent could not make this
for d in sorted(detections, key=lambda d: -d.confidence)[:5]:
    print(f"  {d.cls_name:<12} {d.confidence:.3f}  {tuple(round(v) for v in d.bbox)}")
"""
        ),
        md("## Section 3: Visualization"),
        code(
            """
from PIL import Image
from evaluation.plots import draw_detections

target_asset = library.resolve(cfg["stage1"]["target_object"])
with Image.open(target_asset.path) as im:
    cutout = im.convert("RGBA")
checker = Image.new("RGBA", cutout.size, (235, 235, 235, 255))
checker.paste(cutout, (0, 0), cutout)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
axes[0].imshow(checker); axes[0].axis("off")
axes[0].set_title(f"{target_asset.id} -- a real cutout, own alpha, own object")
axes[1].imshow(draw_detections(clean, detections, highlight=cfg["victim"]["target_class"]))
axes[1].axis("off"); axes[1].set_title("Stage 1's clean World, rendered")
plt.show()
"""
        ),
        md(
            """
## Section 4: Baseline — the API tour

Everything an agent is allowed to know, printed out.
"""
        ),
        code(
            """
from configs.loader import build_stage1_env

env = build_stage1_env(cfg, victim)
print("action_space():")
print(env.action_space().describe())
print()
print("observation_space():")
print(env.observation_space().describe())

obs = env.reset(seed=0)
print()
print("observation keys :", list(obs))
print("image shape      :", obs["image"].shape, obs["image"].dtype)
print("vector           :", np.round(obs["vector"], 3))
"""
        ),
        md(
            """
## Section 5: Attack Agent — and the wall between it and the world

`seal(env)` hands out a restricted handle.  The five public calls work; anything
else raises.  Every experiment in this project runs the agent through this
handle, so a rule violation cannot pass silently.
"""
        ),
        code(
            """
from environments.sealed import seal, EnvironmentAccessError

api = seal(env)
obs = api.reset(seed=0)
obs, reward, terminated, truncated, info = api.step(api.action_space().sample(np.random.default_rng(0)))
print("step() ->", f"reward={reward:+.4f}", f"terminated={terminated}", f"truncated={truncated}", info)

for forbidden in ["_victim", "_world", "_baseline_conf", "render_human", "pop_telemetry"]:
    try:
        getattr(api, forbidden)
        print(f"{forbidden:<16} LEAKED (this would be a bug)")
    except EnvironmentAccessError as exc:
        print(f"{forbidden:<16} blocked -> {type(exc).__name__}")
"""
        ),
        md(
            """
## Section 6: Evaluation

The metric set is identical in all three stages, so results can be put in one
table: attack success rate, confidence drop, mean reward, movement cost,
episode length.
"""
        ),
        code(
            """
from agents.random_agent import RandomAgent
from evaluation.runner import run_episodes
from evaluation.metrics import summarize
from evaluation.report import format_table

records, _ = run_episodes(env, RandomAgent(seed=0), n_episodes=10, method="random", seed=0)
summary = summarize(records)
summary["label"] = "random (smoke test)"
print(format_table([summary]))
"""
        ),
        md("## Section 7: Visualization"),
        code(
            """
plt.figure(figsize=(7, 3.6))
plt.plot([r.baseline_confidence for r in records], label="clean confidence")
plt.plot([r.best_confidence for r in records], label="after attack")
plt.xlabel("episode"); plt.ylabel("victim confidence"); plt.grid(alpha=.3); plt.legend()
plt.title("10 random placements through the Environment API")
plt.show()
"""
        ),
        md(
            """
Setup is complete when the cell above shows the attacked confidence dipping
below the clean line at least once.  Continue with `02_stage1_image.ipynb`.
"""
        ),
    )


# ----------------------------------------------------------------------
def build_02_stage1() -> nbf.NotebookNode:
    return notebook(
        md(
            """
# 02 · Stage 1 — ImageEnvironment

**Question:** can an agent that may only call the Environment API find
placements that hurt YOLO?

No physics and no RL yet.  The scene is a `World`: a real cutout of one object
sitting on a plain background.  The agent proposes `(x, y, size, rotation)`
for its own patch object; the environment validates it, renders the scene,
runs the frozen victim, and returns one scalar.
"""
        ),
        code(BOOTSTRAP),
        code(
            """
from configs.loader import build_stage1_env
from environments.sealed import seal

victim = build_victim(cfg)
env = build_stage1_env(cfg, victim)
api = seal(env)                      # what the agent gets
print(api.action_space().describe())
"""
        ),
        md(
            """
## Section 2: Environment

The full loop lives *inside* `step()`:
`validate -> render -> victim -> evaluate -> reward`.
"""
        ),
        code(
            """
obs = api.reset(seed=0)
baseline = env.pop_telemetry()               # experimenter-side telemetry
print("clean detection:", baseline["baseline_class"], f"{baseline['baseline_confidence']:.3f}")

# size is capped at the target's own silhouette -- the patch can never be
# bigger than the object it attacks (docs/DESIGN.md section 5)
tx, ty = cfg["stage1"]["target_center"]
max_size = api.action_space().high[2]
n_texture = api.action_space().n - 4
texture = np.random.default_rng(0).uniform(0, 1, n_texture)  # a random pattern
obs, reward, terminated, truncated, info = api.step(
    np.concatenate([[tx, ty, max_size, 20.0], texture])
)
print("reward:", round(reward, 4), "| info:", info)
"""
        ),
        md("## Section 3: Visualization"),
        code(
            """
from evaluation.plots import draw_detections

clean = env.clean_image()
attacked = env.render_human()
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
axes[0].imshow(draw_detections(clean, env.victim_report(clean), highlight=baseline["baseline_class"]))
axes[0].set_title("clean"); axes[0].axis("off")
axes[1].imshow(draw_detections(attacked, env.victim_report(attacked), highlight=baseline["baseline_class"]))
axes[1].set_title("one hand-picked placement"); axes[1].axis("off")
plt.show()
"""
        ),
        md(
            """
## Section 4: Baseline — Random and Greedy search

Both baselines see only `action_space()` and the scalar reward.  Raise
`BUDGET` for a stronger (slower) search.
"""
        ),
        code(
            """
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from evaluation.runner import run_episodes
from evaluation.metrics import summarize
from evaluation.report import format_table

BUDGET = 60          # scripts/run_stage1.py uses cfg["stage1"]["search_episodes"]

results, summaries, curves = {}, [], {}
for name, agent in [("random", RandomAgent(seed=0)), ("greedy", GreedyAgent(seed=0))]:
    records, traces = run_episodes(env, agent, BUDGET, method=name, variant="photo", seed=0)
    results[name] = (records, traces)
    curves[name] = [r.best_confidence for r in records]
    s = summarize(records); s["label"] = name
    summaries.append(s)

print(format_table(summaries))
"""
        ),
        md(
            """
## Section 5: Attack Agent

Stage 1 deliberately stops at search — PPO arrives in Stage 2, where the world
has dynamics worth learning.  The best placement found so far:
"""
        ),
        code(
            """
best_method = min(curves, key=lambda k: min(curves[k]))
records, traces = results[best_method]
best_idx = int(np.argmin([r.best_confidence for r in records]))
best_action = np.array(traces[best_idx].telemetry["steps"][0]["action"])
print("best method:", best_method, "| action:", np.round(best_action, 1))

api.reset(seed=0)
api.step(best_action)
best_frame = env.render_human()
best_conf = env.pop_telemetry()["steps"][-1]["current_confidence"]
print(f"confidence {baseline['baseline_confidence']:.3f} -> {best_conf:.3f}")
"""
        ),
        md("## Section 6: Evaluation"),
        code(
            """
import pandas as pd
from evaluation.metrics import to_dataframe

df = pd.concat([to_dataframe(r) for r, _ in results.values()])
df.groupby("method")[
    ["baseline_confidence", "best_confidence", "confidence_drop", "total_reward", "success"]
].mean().round(4)
"""
        ),
        md("## Section 7: Visualization"),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for name, values in curves.items():
    axes[0].plot(np.minimum.accumulate(values), label=name, linewidth=2)
axes[0].axhline(baseline["baseline_confidence"], ls="--", c="k", lw=1, label="clean")
axes[0].set_xlabel("queries to the environment"); axes[0].set_ylabel("best confidence found")
axes[0].grid(alpha=.3); axes[0].legend(); axes[0].set_title("search progress")

axes[1].imshow(draw_detections(best_frame, env.victim_report(best_frame), highlight=baseline["baseline_class"]))
axes[1].axis("off"); axes[1].set_title(f"best attack ({best_method}): {best_conf:.3f}")
plt.show()
"""
        ),
        md(
            """
### Stage 1 verdict

Stage 1 is done when the attacked confidence is clearly below the clean
confidence — i.e. an agent restricted to the Environment API *can* move the
victim model.  Full run with the configured budget:

```
uv run python scripts/run_stage1.py
```
"""
        ),
    )


# ----------------------------------------------------------------------
def _physics_notebook(stage: str, builder: str, title: str, intro: str, extra: str) -> nbf.NotebookNode:
    return notebook(
        md(title),
        code(BOOTSTRAP),
        md(intro),
        code(
            f"""
from configs.loader import {builder}
from environments.sealed import seal

victim = build_victim(cfg)
env = {builder}(cfg, victim, obstacles=False)
api = seal(env)

print(api.action_space().describe())
print()
print(api.observation_space().describe())
"""
        ),
        md(
            """
## Section 2: Environment

The agent asks for a *push*, not a position.  Constraint check, physics,
collision and rendering all happen behind `step()`.
"""
        ),
        code(
            """
obs = api.reset(seed=0)
base = env.pop_telemetry()
print("clean scene:", base["baseline_class"], f"{base['baseline_confidence']:.3f}")

for _ in range(5):
    obs, reward, terminated, truncated, info = api.step(np.ones(api.action_space().n) * 0.8)
    print(f"reward={reward:+.4f}  valid={info['action_valid']}  terminated={terminated}")
"""
        ),
        md("## Section 3: Visualization"),
        code(
            f"""
env_obs = {builder}(cfg, victim, obstacles=True)
env_obs.reset(seed=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(env.clean_image());   axes[0].set_title("clean scene (no attacker)")
axes[1].imshow(env.render_human());  axes[1].set_title("after 5 pushes")
axes[2].imshow(env_obs.render_human()); axes[2].set_title("obstacle variant")
for ax in axes: ax.axis("off")
plt.show()
"""
        ),
        md(
            """
## Section 4: Baseline — Random and Greedy

Exactly the agents from Stage 1, unchanged.
"""
        ),
        code(
            """
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from evaluation.runner import run_episodes
from evaluation.metrics import summarize
from evaluation.report import format_table

EPISODES = 8        # the script uses cfg[stage]["eval_episodes"]

summaries, conf_curves = [], {}
for name, agent in [("random", RandomAgent(seed=0)), ("greedy", GreedyAgent(seed=0))]:
    records, traces = run_episodes(env, agent, EPISODES, method=name, seed=0)
    s = summarize(records); s["label"] = name
    summaries.append(s)
    conf_curves[name] = [t.confidences for t in traces]

print(format_table(summaries))
"""
        ),
        md(
            f"""
## Section 5: Attack Agent — PPO

{extra}
"""
        ),
        code(
            f"""
from pathlib import Path
from agents.ppo_agent import PPOAgent

model_path = resolve(cfg["output"]["results_dir"]) / "{stage}" / "ppo_no_obstacle.zip"
TRAIN_HERE = False        # flip to True to train inside the notebook (slow on CPU)

if model_path.exists() and not TRAIN_HERE:
    ppo = PPOAgent.load(model_path)
    print("loaded", model_path)
else:
    ppo = PPOAgent.train(
        env_factory=lambda: {builder}(cfg, victim, obstacles=False),
        total_timesteps=2000,       # a demo budget; scripts use cfg[...]["ppo"]
        seed=0,
    )
    print("trained a short demo policy")
"""
        ),
        code(
            """
records, traces = run_episodes(env, ppo, EPISODES, method="ppo", seed=0)
s = summarize(records); s["label"] = "ppo"
summaries.append(s)
conf_curves["ppo"] = [t.confidences for t in traces]
print(format_table(summaries))
"""
        ),
        md("## Section 6: Evaluation"),
        code(
            f"""
import json
summary_file = resolve(cfg["output"]["results_dir"]) / "{stage}" / "summary.json"
if summary_file.exists():
    import pandas as pd
    full = pd.DataFrame(json.loads(summary_file.read_text()))
    display(full[["label", "attack_success_rate", "mean_confidence_drop",
                  "mean_reward", "mean_movement_cost", "mean_episode_length"]].round(3))
else:
    print("run scripts/run_{stage}.py for the full-budget numbers")
"""
        ),
        md("## Section 7: Visualization"),
        code(
            """
from evaluation.plots import PALETTE

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for name, episodes in conf_curves.items():
    length = max(len(e) for e in episodes)
    padded = np.array([e + [e[-1]] * (length - len(e)) for e in episodes], dtype=float)
    axes[0].plot(np.arange(1, length + 1), padded.mean(0), label=name,
                 color=PALETTE.get(name), linewidth=2)
axes[0].axhline(base["baseline_confidence"], ls="--", c="k", lw=1, label="clean")
axes[0].set_xlabel("step"); axes[0].set_ylabel("victim confidence")
axes[0].grid(alpha=.3); axes[0].legend(); axes[0].set_title("confidence during an episode")

names = [s["label"] for s in summaries]
axes[1].bar(names, [s["attack_success_rate"] for s in summaries],
            color=[PALETTE.get(n, "#888") for n in names])
axes[1].set_title("attack success rate"); axes[1].grid(axis="y", alpha=.3)
plt.show()
"""
        ),
        code(
            f"""
from evaluation.plots import draw_detections

_, demo = run_episodes(env, ppo, 1, method="ppo", seed=123, collect_frames=True)
frame = demo[0].best_frame()
if frame is not None:
    plt.figure(figsize=(6, 6))
    plt.imshow(draw_detections(frame, env.victim_report(frame), highlight=base["baseline_class"]))
    plt.axis("off"); plt.title(f"best PPO frame: confidence {{min(demo[0].confidences):.3f}}")
    plt.show()
"""
        ),
        md(
            f"""
Full experiment (both variants, full budgets, all figures):

```
uv run python scripts/run_{stage}.py
```
"""
        ),
    )


def build_03_stage2() -> nbf.NotebookNode:
    return _physics_notebook(
        stage="stage2",
        builder="build_stage2_env",
        title="""
# 03 · Stage 2 — Physics2DEnvironment

**Question:** once the attacker is a physical object that has to be *pushed*
into place, and a wall may be in the way, does the attack still work?

Nothing about the agent interface, the reward or the victim changes here.  Only
the environment implementation gains physics.
""",
        intro="""
The action is now `(dx, dy)` — a push, not a placement.  The environment applies
acceleration limits, a speed cap, a per-step movement cap, collisions and the
world boundary before anything is rendered.
""",
        extra="""
This is the first stage with reinforcement learning.  PPO (Stable-Baselines3)
trains through `GymEnvAdapter`, which wraps the *sealed* environment — so PPO
gets exactly the same five calls as the random baseline.

Training on CPU takes a few minutes; the cell below loads the policy saved by
`scripts/run_stage2.py` if it exists.
""",
    )


def build_04_stage3() -> nbf.NotebookNode:
    return _physics_notebook(
        stage="stage3",
        builder="build_stage3_env",
        title="""
# 04 · Stage 3 — Physics3DEnvironment

**Question:** does the same Agent / API design survive the move to 3D?

The action grows to `(dx, dy, dz)`, the physics core is called with 3-vectors,
and a pinhole camera projects depth-sorted billboards.  `BaseEnvironment`,
`AttackAgent`, `AttackReward` and the experiment driver are untouched.
""",
        intro="""
Depth now matters: occluding the target requires getting *in front of* it, and
an object further from the camera than the target has no effect at all.
""",
        extra="""
The same `PPOAgent`, the same hyper-parameters, one extra action dimension.
""",
    )


# ----------------------------------------------------------------------
def build_05_evaluation() -> nbf.NotebookNode:
    return notebook(
        md(
            """
# 05 · Evaluation — the five questions

This notebook reads whatever `results/` contains and answers the questions from
the project brief.  Run the three stage scripts first:

```
uv run python scripts/run_stage1.py
uv run python scripts/run_stage2.py
uv run python scripts/run_stage3.py
```
"""
        ),
        code(BOOTSTRAP),
        code(
            """
import json
import pandas as pd

RESULTS = resolve(cfg["output"]["results_dir"])
FIGURES = resolve(cfg["output"]["figures_dir"])

frames = {}
for stage in ["stage1", "stage2", "stage3"]:
    path = RESULTS / stage / "summary.json"
    if path.exists():
        df = pd.DataFrame(json.loads(path.read_text()))
        df["stage_name"] = stage
        frames[stage] = df
    else:
        print(f"missing: {path}")

summary = pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()
summary[["stage_name", "label", "n_episodes", "attack_success_rate",
         "mean_baseline_confidence", "mean_best_confidence", "mean_confidence_drop",
         "mean_reward", "mean_movement_cost", "mean_episode_length"]].round(3)
"""
        ),
        md("## Section 2–3: Environments and what they look like"),
        code(
            """
from IPython.display import Image as ShowImage, display

for name in ["stage1_best_attack.png", "stage2_no_obstacle_best_attack.png",
             "stage3_no_obstacle_best_attack.png"]:
    path = FIGURES / name
    if path.exists():
        print(name)
        display(ShowImage(filename=str(path), width=760))
"""
        ),
        md(
            """
## Section 4–5: Baselines vs PPO

**Q3 — is PPO better than Random / Greedy?**
"""
        ),
        code(
            """
if not summary.empty:
    view = summary[summary.stage_name != "stage1"]
    pivot = view.pivot_table(index="method", columns="variant",
                             values=["attack_success_rate", "mean_confidence_drop", "mean_reward"])
    display(pivot.round(3))
"""
        ),
        md("**Q2 / Q4 — what do physics and obstacles cost the attacker?**"),
        code(
            """
if not summary.empty:
    display(summary.pivot_table(index=["stage_name", "variant"], columns="method",
                                values="attack_success_rate").round(3))
    display(summary.pivot_table(index=["stage_name", "variant"], columns="method",
                                values="mean_confidence_drop").round(3))
"""
        ),
        md("## Section 6: Evaluation — the five questions"),
        code(
            """
def answer(question, condition, yes, no):
    print(f"{question}\\n  -> {yes if condition else no}\\n")

if not summary.empty:
    s1 = summary[summary.stage_name == "stage1"]
    s2 = summary[summary.stage_name == "stage2"]
    s3 = summary[summary.stage_name == "stage3"]

    answer("Q1  Can the agent affect the victim through a restricted API?",
           len(s1) and s1.mean_confidence_drop.max() > 0.05,
           f"yes - stage 1 confidence drop up to {s1.mean_confidence_drop.max():.3f}",
           "no measurable effect")

    if len(s1) and len(s2):
        answer("Q2  What do physical constraints cost?",
               True,
               f"stage1 success {s1.attack_success_rate.max():.2f} vs "
               f"stage2 success {s2.attack_success_rate.max():.2f}",
               "")

    if len(s2):
        ppo = s2[s2.method == "ppo"].mean_reward.mean()
        base = s2[s2.method != "ppo"].mean_reward.mean()
        answer("Q3  Is PPO better than the baselines?",
               ppo > base,
               f"yes - mean reward {ppo:.3f} vs {base:.3f}",
               f"not in this budget - mean reward {ppo:.3f} vs {base:.3f}")

    if len(s2) and s2.variant.nunique() > 1:
        with_obs = s2[s2.variant == "obstacle"].attack_success_rate.mean()
        without = s2[s2.variant == "no_obstacle"].attack_success_rate.mean()
        answer("Q4  Does the agent adapt to obstacles?",
               with_obs > 0.0,
               f"success {without:.2f} without obstacles vs {with_obs:.2f} with",
               "obstacles defeated the attack entirely")

    answer("Q5  Does the same Agent/API design survive 2D -> 3D?",
           len(s3) > 0,
           "yes - stage 3 ran the same agents, reward and driver unchanged",
           "stage 3 has not been run yet")
"""
        ),
        md("## Section 7: Visualization"),
        code(
            """
for name in sorted(p.name for p in FIGURES.glob("*.png")):
    print(name)
    display(ShowImage(filename=str(FIGURES / name), width=900))
"""
        ),
    )


# ----------------------------------------------------------------------
BUILDERS = {
    "01_setup.ipynb": build_01_setup,
    "02_stage1_image.ipynb": build_02_stage1,
    "03_stage2_2d.ipynb": build_03_stage2,
    "04_stage3_3d.ipynb": build_04_stage3,
    "05_evaluation.ipynb": build_05_evaluation,
}


def main() -> int:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        path = NOTEBOOKS / name
        nbf.write(builder(), str(path))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
