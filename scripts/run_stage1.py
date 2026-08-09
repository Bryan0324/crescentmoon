"""Stage 1 experiment: can an agent hurt YOLO through the ImageEnvironment API?

    uv run python scripts/run_stage1.py [--episodes 300] [--config configs/default.yaml]

Writes records, summaries and figures under ``results/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(line_buffering=True)  # keep progress visible when redirected

from agents.greedy_agent import GreedyAgent  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from configs.loader import build_stage1_env, build_victim, load_config, resolve  # noqa: E402
from evaluation.metrics import save_records, summarize  # noqa: E402
from evaluation.plots import (  # noqa: E402
    draw_detections,
    plot_attack_example,
    plot_method_comparison,
    plot_search_progress,
)
from evaluation.report import format_table, markdown_table, save_json  # noqa: E402
from evaluation.runner import run_episodes  # noqa: E402
from rendering.image_renderer import save_rgb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=None, help="query budget per method")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    budget = args.episodes or int(cfg["stage1"]["search_episodes"])
    results_dir = resolve(cfg["output"]["results_dir"]) / "stage1"
    figures_dir = resolve(cfg["output"]["figures_dir"])

    print("== Stage 1: ImageEnvironment ==")
    victim = build_victim(cfg)
    env = build_stage1_env(cfg, victim)
    print(f"victim          : {victim.name}")
    print(f"action space    :\n{env.action_space().describe()}")
    print(f"observation     :\n{env.observation_space().describe()}")

    env.reset(seed=seed)
    baseline_telemetry = env.pop_telemetry()
    baseline_conf = float(baseline_telemetry["baseline_confidence"])
    baseline_class = baseline_telemetry["baseline_class"]
    print(f"clean detection : {baseline_class} @ {baseline_conf:.3f}\n")

    agents = {
        "random": RandomAgent(seed=seed),
        "greedy": GreedyAgent(seed=seed),
    }

    all_records = []
    summaries = []
    progress_series = {}
    best_by_method = {}

    for name, agent in agents.items():
        started = time.time()
        records, traces = run_episodes(
            env, agent, budget, method=name, variant="photo", seed=seed
        )
        elapsed = time.time() - started
        all_records.extend(records)

        summary = summarize(records)
        summary["label"] = name
        summary["seconds"] = elapsed
        summary["queries"] = budget
        summaries.append(summary)

        progress_series[name] = [r.best_confidence for r in records]
        best_idx = int(np.argmin([r.best_confidence for r in records]))
        best_by_method[name] = traces[best_idx].telemetry["steps"][0]["action"]

        print(
            f"{name:<7} {budget:>4} queries in {elapsed:6.1f}s | "
            f"success {summary['attack_success_rate']:.2f} | "
            f"conf {baseline_conf:.3f} -> {summary['mean_best_confidence']:.3f} | "
            f"best conf {min(progress_series[name]):.3f}"
        )

    # ---------------- figures + artefacts ------------------------------
    save_records(results_dir / "episodes.json", all_records)
    save_json(results_dir / "summary.json", summaries)
    (results_dir / "summary.md").write_text(markdown_table(summaries), encoding="utf-8")

    plot_search_progress(
        progress_series,
        figures_dir / "stage1_search_progress.png",
        title="Stage 1: best victim confidence found vs query budget",
    )
    plot_method_comparison(
        summaries, figures_dir / "stage1_method_comparison.png", title="Stage 1 (photo)"
    )

    # Replay the single best attack found, and draw what the victim now sees.
    best_method = min(best_by_method, key=lambda m: min(progress_series[m]))
    best_action = np.asarray(best_by_method[best_method], dtype=float)
    env.reset(seed=seed)
    clean = env.clean_image()
    env.step(best_action)
    attacked = env.render_human()
    telemetry = env.pop_telemetry()
    attacked_conf = float(telemetry["steps"][-1]["current_confidence"])

    save_rgb(results_dir / "best_attack.png", attacked)
    save_rgb(
        results_dir / "best_attack_detections.png",
        draw_detections(attacked, victim.detect(attacked), highlight=baseline_class),
    )
    plot_attack_example(
        draw_detections(clean, victim.detect(clean), highlight=baseline_class),
        draw_detections(attacked, victim.detect(attacked), highlight=baseline_class),
        baseline_conf,
        attacked_conf,
        figures_dir / "stage1_best_attack.png",
        title=f"Stage 1 best attack ({best_method}), action = "
        f"x={best_action[0]:.0f} y={best_action[1]:.0f} "
        f"size={best_action[2]:.0f} rot={best_action[3]:.0f}deg",
    )

    print("\n" + format_table(summaries))
    print(f"\nbest attack: {best_method}, confidence {baseline_conf:.3f} -> {attacked_conf:.3f}")
    print(f"results -> {results_dir}")
    print(f"figures -> {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
