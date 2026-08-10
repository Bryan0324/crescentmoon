"""End-to-end health check: does this project actually run?

Fast (no YOLO, no training): builds all three environments on the deterministic
stub victim, walks the Environment API, verifies the access rules, runs a short
episode with every agent, and reports what is present on disk.

    uv run python scripts/check_project.py            # fast structural check
    uv run python scripts/check_project.py --with-yolo  # also load real YOLO
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(line_buffering=True)  # keep progress visible when redirected

from agents.greedy_agent import GreedyAgent  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from configs.loader import load_config, resolve  # noqa: E402
from environments.image_env import ImageEnvConfig, ImageEnvironment  # noqa: E402
from environments.physics2d_env import Physics2DEnvConfig, Physics2DEnvironment  # noqa: E402
from environments.physics3d_env import Physics3DEnvConfig, Physics3DEnvironment  # noqa: E402
from environments.sealed import EnvironmentAccessError, seal  # noqa: E402
from evaluation.runner import run_episodes  # noqa: E402
from models.stub_victim import ColorBlobVictim  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
_failures: list[str] = []


def check(name: str, fn):
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - this is the reporting boundary
        _failures.append(f"{name}: {exc}")
        print(f"[{FAIL}] {name}\n         {type(exc).__name__}: {exc}")
        if "--traceback" in sys.argv:
            traceback.print_exc()
        return None
    print(f"[{PASS}] {name}" + (f" — {detail}" if detail else ""))
    return detail


# ----------------------------------------------------------------------
def _tiny_cutout(tmp: Path) -> Path:
    path = tmp / "check_cutout.png"
    size = (56, 110)
    sprite = Image.new("RGBA", size, (0, 200, 0, 0))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse([3, 3, size[0] - 3, size[1] - 3], fill=255)
    sprite.putalpha(mask)
    sprite.save(path)
    return path


def build_environments(tmp: Path):
    cutout = _tiny_cutout(tmp)
    return {
        "ImageEnvironment": ImageEnvironment(
            ImageEnvConfig(
                target_sprite_path=cutout,
                render_size=160,
                obs_size=32,
                target_class="target",
                target_center=(80.0, 88.0),
                target_height=80,
                patch_min_frac=0.25,
                patch_max_frac=0.6,
            ),
            victim=ColorBlobVictim(),
        ),
        "Physics2DEnvironment": Physics2DEnvironment(
            Physics2DEnvConfig(
                render_size=160,
                obs_size=32,
                target_class="target",
                target_center=(80.0, 88.0),
                target_height=80,
                max_steps=6,
                spawn_center=(35.0, 130.0),
                spawn_jitter=4.0,
                obstacles=[(60.0, 100.0, 100.0, 145.0)],
            ),
            victim=ColorBlobVictim(),
        ),
        "Physics3DEnvironment": Physics3DEnvironment(
            Physics3DEnvConfig(
                render_size=160, obs_size=32, target_class="target", max_steps=6
            ),
            victim=ColorBlobVictim(),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-yolo", action="store_true", help="also load the real victim")
    parser.add_argument("--traceback", action="store_true")
    args = parser.parse_args()

    print("CrescentMoon project check\n" + "=" * 60)

    loaded: dict = {}

    def _load_cfg() -> str:
        loaded["cfg"] = load_config()
        return f"{len(loaded['cfg'])} sections, victim={loaded['cfg']['victim']['kind']}"

    check("config loads", _load_cfg)
    cfg = loaded.get("cfg")
    tmp = PROJECT_ROOT / "results" / "_check"
    tmp.mkdir(parents=True, exist_ok=True)

    envs = check("all three environments construct", lambda: build_environments(tmp))
    if envs is None:
        print("\ncannot continue without environments")
        return 1
    print()

    for name, env in envs.items():
        space = env.action_space()

        check(
            f"{name}: reset/observe/step honour the API",
            lambda env=env: (
                _api_roundtrip(env),
                f"action_dim={env.action_space().n}, obs={list(env.observe())}",
            )[1],
        )
        check(
            f"{name}: illegal actions are rejected, not obeyed",
            lambda env=env, space=space: _rejects_illegal(env, space),
        )
        check(f"{name}: internals are sealed from agents", lambda env=env: _sealed(env))
        check(
            f"{name}: random + greedy agents complete episodes",
            lambda env=env: _agents_run(env),
        )
        print()

    # --------------------------------------------------------------
    print("optional components")
    for label, importer in [
        ("gymnasium adapter", lambda: __import__("environments.gym_adapter", fromlist=["x"])),
        ("stable-baselines3", lambda: __import__("stable_baselines3")),
        ("ultralytics", lambda: __import__("ultralytics")),
        ("torch", lambda: __import__("torch")),
    ]:
        check(f"import {label}", lambda importer=importer: getattr(importer(), "__name__", "ok"))

    if args.with_yolo and cfg:
        check("real YOLO victim loads and detects", lambda: _yolo_smoke(cfg, tmp))

    # --------------------------------------------------------------
    print("\nassets and results on disk")
    if cfg:
        for label, path in [
            ("source photo", resolve(cfg["assets"]["source_photo"])),
            ("target sprite", resolve(cfg["assets"]["target_sprite"])),
            ("results dir", resolve(cfg["output"]["results_dir"])),
            ("figures dir", resolve(cfg["output"]["figures_dir"])),
        ]:
            state = "present" if path.exists() else "missing"
            print(f"  {label:<14} {state:<8} {path}")

    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


# ----------------------------------------------------------------------
def _api_roundtrip(env) -> None:
    obs = env.reset(seed=0)
    assert env.observation_space().contains(obs), "reset() observation off-spec"
    action = env.action_space().sample(np.random.default_rng(0))
    result = env.step(action)
    assert len(result) == 5, "step() must return 5 items"
    assert env.observation_space().contains(result.observation), "step() observation off-spec"
    assert isinstance(result.reward, float), "reward must be a float"


def _rejects_illegal(env, space) -> str:
    env.reset(seed=0)
    _, _, _, _, info = env.step(space.high * 500.0 + 500.0)
    assert info["action_valid"] is False, "an out-of-range action was accepted"
    try:
        env.step(np.zeros(space.n + 2))
    except ValueError:
        return "out-of-range clipped, wrong shape raises"
    raise AssertionError("a wrongly-shaped action was accepted")


def _sealed(env) -> str:
    api = seal(env)
    api.reset(seed=0)
    blocked = 0
    for attribute in ("_victim", "_renderer", "_baseline_conf", "render_human", "pop_telemetry"):
        try:
            getattr(api, attribute)
        except EnvironmentAccessError:
            blocked += 1
        else:
            raise AssertionError(f"agent could reach env.{attribute}")
    return f"{blocked}/5 privileged attributes blocked"


def _agents_run(env) -> str:
    out = []
    for agent in (RandomAgent(seed=0), GreedyAgent(seed=0)):
        records, _ = run_episodes(env, agent, 2, seed=0)
        assert len(records) == 2
        drop = max(r.confidence_drop for r in records)
        out.append(f"{agent.name} drop {drop:+.3f}")
    return ", ".join(out)


def _yolo_smoke(cfg, tmp: Path) -> str:
    from configs.loader import build_victim
    from rendering.image_renderer import load_rgb

    victim = build_victim(cfg)
    photo = resolve(cfg["assets"]["source_photo"])
    if not photo.exists():
        return f"{victim.name} loaded (no source photo yet -- run scripts/prepare_assets.py)"
    target_class = cfg["victim"].get("target_class")
    detections = victim.detect(load_rgb(photo))
    best = detections.best(target_class)
    if best is None:
        return f"{victim.name}: {len(detections)} detections, no '{target_class}' found"
    return f"{victim.name}: {len(detections)} detections, best {target_class} = {best.confidence:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
