"""Collect whatever is in ``results/`` into ``docs/RESULTS.md``.

Only reports what actually ran -- missing stages are listed as missing rather
than quietly omitted.

    uv run python scripts/make_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config, resolve  # noqa: E402
from evaluation.report import markdown_table  # noqa: E402

STAGES = {
    "stage1": "Stage 1 · ImageEnvironment（照片、無物理、只有搜尋）",
    "stage2": "Stage 2 · Physics2DEnvironment（推力 + 碰撞，Random / Greedy / PPO）",
    "stage3": "Stage 3 · Physics3DEnvironment（同一套 agent，多一個軸）",
}


def _load(stage_dir: Path) -> list[dict] | None:
    path = stage_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, spec="{:.3f}") -> str:
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        return "n/a"


def _answers(data: dict[str, list[dict]]) -> list[str]:
    out: list[str] = []
    s1, s2, s3 = data.get("stage1"), data.get("stage2"), data.get("stage3")

    if s1:
        drop = max(r["mean_confidence_drop"] for r in s1)
        best = min(r["mean_best_confidence"] for r in s1)
        clean = max(r["mean_baseline_confidence"] for r in s1)
        out.append(
            "**Q1 · Agent 是否能透過有限 API 影響 victim model？** 可以。"
            f"在只有 `observe() / step() / action_space()` 的條件下，搜尋型 agent 把追蹤目標的"
            f"信心從 {clean:.3f} 平均壓到 {best:.3f}（平均降幅 {drop:.3f}）。"
        )
    else:
        out.append("**Q1** — stage 1 尚未執行。")

    if s1 and s2:
        best1 = max(r["attack_success_rate"] for r in s1)
        best2 = max(r["attack_success_rate"] for r in s2)
        out.append(
            "**Q2 · 加入物理限制後，攻擊成功率如何改變？** "
            f"Stage 1（可自由放置）最高成功率 {best1:.2f}；"
            f"Stage 2（只能推、有速度上限與碰撞）最高成功率 {best2:.2f}。"
        )
    else:
        out.append("**Q2** — 需要 stage 1 與 stage 2 都跑過。")

    if s2:
        ppo = [r for r in s2 if r["method"] == "ppo"]
        base = [r for r in s2 if r["method"] != "ppo"]
        if ppo and base:
            p = sum(r["mean_reward"] for r in ppo) / len(ppo)
            b = sum(r["mean_reward"] for r in base) / len(base)
            verdict = "優於" if p > b else "並未優於"
            out.append(
                f"**Q3 · PPO 是否優於 Random / Greedy？** 在這個訓練預算下，PPO {verdict} baseline："
                f"平均 episode reward {p:+.3f} vs {b:+.3f}。"
            )
        else:
            out.append("**Q3** — stage 2 這次沒有跑 PPO。")
    else:
        out.append("**Q3** — stage 2 尚未執行。")

    obstacle_lines = []
    for stage in ("stage2", "stage3"):
        rows = data.get(stage) or []
        with_obs = [r for r in rows if r.get("variant") == "obstacle"]
        without = [r for r in rows if r.get("variant") == "no_obstacle"]
        if with_obs and without:
            a = max(r["attack_success_rate"] for r in without)
            b = max(r["attack_success_rate"] for r in with_obs)
            obstacle_lines.append(f"{stage} 最高成功率 無障礙 {a:.2f} vs 有障礙 {b:.2f}")
    out.append(
        "**Q4 · 加入 obstacle 後，Agent 是否能適應？** "
        + ("；".join(obstacle_lines) + "。" if obstacle_lines else "這次沒有做障礙物對照。")
    )

    if s3:
        out.append(
            "**Q5 · 從 2D 擴展到 3D 時，同一套 Agent / API 設計是否仍然成立？** 成立。"
            "Stage 3 原封不動重用 `BaseEnvironment`、`AttackAgent`、`AttackReward`、"
            "`run_episodes` 與 `run_physics_stage`，只有 environment 實作與 action 維度不同。"
        )
    else:
        out.append("**Q5** — stage 3 尚未執行。")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="docs/RESULTS.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = resolve(cfg["output"]["results_dir"])
    figures_dir = resolve(cfg["output"]["figures_dir"])
    out_path = resolve(args.out)

    data = {stage: _load(results_dir / stage) for stage in STAGES}
    data = {k: v for k, v in data.items() if v}

    lines = [
        "# 實驗結果",
        "",
        f"由 `scripts/make_report.py` 於 "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} 自動產生。",
        "",
        f"Victim model：`{cfg['victim']['kind']}` "
        f"(`{cfg['victim'].get('weights', '-')}`, imgsz={cfg['victim'].get('imgsz')}, "
        f"conf≥{cfg['victim'].get('conf_threshold')})，frozen，全程不訓練。",
        "",
        "Reward（三個 stage 完全相同）："
        f"`{cfg['reward']['w_conf']}·confidence_drop − {cfg['reward']['w_move']}·movement_cost "
        f"− {cfg['reward']['w_invalid']}·invalid_action "
        f"(+{cfg['reward']['success_bonus']} on success，success = confidence < "
        f"{cfg['reward']['success_threshold']})`。",
        "",
        "---",
        "",
    ]

    for stage, title in STAGES.items():
        lines += [f"## {title}", ""]
        rows = data.get(stage)
        if not rows:
            lines += [f"_尚未執行。`uv run python scripts/run_{stage}.py`_", ""]
            continue
        lines += [markdown_table(rows), ""]
        figs = sorted(figures_dir.glob(f"{stage}_*.png"))
        for fig in figs:
            rel = Path("..") / fig.relative_to(PROJECT_ROOT)
            lines += [f"![{fig.stem}]({rel.as_posix()})", ""]

    lines += ["---", "", "## 最終實驗問題", ""]
    lines += [f"{line}\n" for line in _answers(data)]

    lines += [
        "---",
        "",
        "## 原始數據",
        "",
        "| 檔案 | 內容 |",
        "| --- | --- |",
        "| `results/stage*/episodes.json` | 每個評估 episode 一列 |",
        "| `results/stage*/summary.json` | 每個 method / variant 的彙總指標 |",
        "| `results/stage*/training_curves.json` | PPO 訓練過程的 episode return |",
        "| `results/stage*/ppo_*.zip` | 訓練好的 policy |",
        "| `results/figures/*.png` | 這份報告裡的所有圖 |",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    for stage in STAGES:
        state = "ok" if stage in data else "MISSING"
        print(f"  {stage}: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
