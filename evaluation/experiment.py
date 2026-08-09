"""The Stage 2 / Stage 3 experiment, written once.

Both physics stages ask exactly the same questions -- Random vs Greedy vs PPO,
with and without obstacles -- so they share this driver.  That the same function
runs both is itself one of the project's results (prompt.md Q5).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from agents.base import AttackAgent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from environments.base import BaseEnvironment
from rendering.image_renderer import save_rgb

from .metrics import save_records, summarize
from .plots import (
    draw_detections,
    plot_attack_example,
    plot_confidence_curves,
    plot_method_comparison,
    plot_training_curve,
)
from .report import markdown_table, save_json
from .runner import run_episodes

__all__ = ["run_physics_stage"]

EnvFactory = Callable[[bool], BaseEnvironment]


def _build_agent(
    method: str,
    *,
    seed: int,
    env_factory: EnvFactory,
    obstacles: bool,
    ppo_cfg: dict[str, Any],
    model_path: Path,
) -> tuple[AttackAgent, dict[str, list] | None]:
    if method == "random":
        return RandomAgent(seed=seed), None
    if method == "greedy":
        return GreedyAgent(seed=seed), None
    if method == "ppo":
        from agents.ppo_agent import PPOAgent

        total = int(ppo_cfg.get("total_timesteps", 12_000))
        kwargs = {
            k: v for k, v in ppo_cfg.items() if k not in {"total_timesteps", "progress"}
        }
        print(f"    training PPO for {total} environment steps ...")
        started = time.time()
        agent = PPOAgent.train(
            env_factory=lambda: env_factory(obstacles),
            total_timesteps=total,
            seed=seed,
            **kwargs,
        )
        print(f"    trained in {time.time() - started:.1f}s")
        agent.save(model_path)
        return agent, agent.curve.to_dict()
    raise ValueError(f"unknown method: {method!r}")


def run_physics_stage(
    *,
    stage: str,
    env_factory: EnvFactory,
    seed: int,
    eval_episodes: int,
    ppo_cfg: dict[str, Any],
    results_dir: Path,
    figures_dir: Path,
    variants: Sequence[str] = ("no_obstacle", "obstacle"),
    methods: Sequence[str] = ("random", "greedy", "ppo"),
    demo_episodes: int = 3,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    all_records = []
    all_summaries: list[dict[str, Any]] = []
    training_curves: dict[str, dict[str, list]] = {}

    for variant in variants:
        obstacles = variant == "obstacle"
        print(f"\n-- {stage} / {variant} --")
        env = env_factory(obstacles)
        env.reset(seed=seed)
        baseline = env.pop_telemetry()
        baseline_conf = float(baseline["baseline_confidence"])
        baseline_class = baseline["baseline_class"]
        print(f"   clean scene: {baseline_class} @ {baseline_conf:.3f}")

        confidence_curves: dict[str, list[list[float]]] = {}
        best_demo: tuple[float, np.ndarray] | None = None
        best_demo_method = ""

        for method in methods:
            agent, curve = _build_agent(
                method,
                seed=seed,
                env_factory=env_factory,
                obstacles=obstacles,
                ppo_cfg=ppo_cfg,
                model_path=results_dir / f"ppo_{variant}.zip",
            )
            if curve is not None:
                training_curves[variant] = curve
                plot_training_curve(
                    curve,
                    figures_dir / f"{stage}_{variant}_ppo_training.png",
                    title=f"{stage} / {variant}: PPO training",
                )

            started = time.time()
            records, traces = run_episodes(
                env, agent, eval_episodes, method=method, variant=variant, seed=seed
            )
            elapsed = time.time() - started
            all_records.extend(records)

            summary = summarize(records)
            summary["label"] = f"{method} ({variant})"
            summary["seconds"] = elapsed
            all_summaries.append(summary)
            confidence_curves[method] = [t.confidences for t in traces]

            print(
                f"   {method:<7} {eval_episodes:>3} eps in {elapsed:6.1f}s | "
                f"success {summary['attack_success_rate']:.2f} | "
                f"conf {baseline_conf:.3f} -> {summary['mean_best_confidence']:.3f} | "
                f"reward {summary['mean_reward']:+.3f} | "
                f"cost {summary['mean_movement_cost']:.1f}"
            )

            # A few extra episodes *with frames* so we can show a real attack.
            if demo_episodes > 0:
                _, demo_traces = run_episodes(
                    env,
                    agent,
                    demo_episodes,
                    method=method,
                    variant=variant,
                    seed=seed + 10_000,
                    collect_frames=True,
                )
                for trace in demo_traces:
                    frame = trace.best_frame()
                    if frame is None or not trace.confidences:
                        continue
                    conf = float(min(trace.confidences))
                    if best_demo is None or conf < best_demo[0]:
                        best_demo = (conf, frame)
                        best_demo_method = method

        plot_confidence_curves(
            confidence_curves,
            figures_dir / f"{stage}_{variant}_confidence.png",
            baseline=baseline_conf,
            title=f"{stage} / {variant}: victim confidence during an episode",
        )
        plot_method_comparison(
            [s for s in all_summaries if s["variant"] == variant],
            figures_dir / f"{stage}_{variant}_comparison.png",
            title=f"{stage} / {variant}",
        )

        if best_demo is not None:
            conf, frame = best_demo
            clean = env.clean_image()
            victim_detect = env.victim_report  # experimenter-facing, sealed off from agents
            save_rgb(results_dir / f"best_attack_{variant}.png", frame)
            plot_attack_example(
                draw_detections(clean, victim_detect(clean), highlight=baseline_class),
                draw_detections(frame, victim_detect(frame), highlight=baseline_class),
                baseline_conf,
                conf,
                figures_dir / f"{stage}_{variant}_best_attack.png",
                title=f"{stage} / {variant}: best attack by {best_demo_method}",
            )

    save_records(results_dir / "episodes.json", all_records)
    save_json(results_dir / "summary.json", all_summaries)
    save_json(results_dir / "training_curves.json", training_curves)
    (results_dir / "summary.md").write_text(markdown_table(all_summaries), encoding="utf-8")

    if len(variants) > 1:
        plot_method_comparison(
            all_summaries,
            figures_dir / f"{stage}_obstacle_effect.png",
            title=f"{stage}: with vs without obstacles",
        )

    return {
        "records": all_records,
        "summaries": all_summaries,
        "training_curves": training_curves,
    }
