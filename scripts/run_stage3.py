"""Stage 3 experiment: the same agents, the same reward, the same experiment
driver -- now in the 3D physics environment.

    uv run python scripts/run_stage3.py [--episodes 20] [--timesteps 15000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(line_buffering=True)  # keep progress visible when redirected

from configs.loader import build_stage3_env, build_victim, load_config, resolve  # noqa: E402
from evaluation.experiment import run_physics_stage  # noqa: E402
from evaluation.report import format_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=["random", "greedy", "ppo"])
    parser.add_argument("--variants", nargs="+", default=["no_obstacle", "obstacle"])
    parser.add_argument("--demo-episodes", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    stage_cfg = cfg["stage3"]
    ppo_cfg = dict(stage_cfg["ppo"])
    if args.timesteps is not None:
        ppo_cfg["total_timesteps"] = args.timesteps

    print("== Stage 3: Physics3DEnvironment ==")
    victim = build_victim(cfg)
    print(f"victim: {victim.name}")

    def env_factory(obstacles: bool):
        return build_stage3_env(cfg, victim, obstacles=obstacles)

    probe = env_factory(False)
    print(f"action space    :\n{probe.action_space().describe()}")
    print(f"observation     :\n{probe.observation_space().describe()}")

    out = run_physics_stage(
        stage="stage3",
        env_factory=env_factory,
        seed=seed,
        eval_episodes=args.episodes or int(stage_cfg["eval_episodes"]),
        ppo_cfg=ppo_cfg,
        results_dir=resolve(cfg["output"]["results_dir"]) / "stage3",
        figures_dir=resolve(cfg["output"]["figures_dir"]),
        variants=tuple(args.variants),
        methods=tuple(args.methods),
        demo_episodes=args.demo_episodes,
    )

    print("\n" + format_table(out["summaries"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
