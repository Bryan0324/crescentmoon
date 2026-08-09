"""Run the whole project end to end, in the order the stages must be built.

    uv run python scripts/run_all.py                 # everything
    uv run python scripts/run_all.py --skip stage3   # stop after 2D
    uv run python scripts/run_all.py --quick         # tiny budgets, for a smoke run

Each step is a separate process, so a failure in stage 3 does not throw away
the stage 1/2 results already written to disk.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def run(name: str, argv: list[str]) -> bool:
    print("\n" + "=" * 70)
    print(f"== {name}: {' '.join(argv)}")
    print("=" * 70, flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, *argv], cwd=PROJECT_ROOT)
    ok = result.returncode == 0
    print(f"-- {name}: {'ok' if ok else 'FAILED'} in {time.time() - started:.1f}s")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", nargs="*", default=[], help="step names to skip")
    parser.add_argument("--quick", action="store_true", help="tiny budgets (smoke run)")
    args = parser.parse_args()

    quick_stage1 = ["--episodes", "40"] if args.quick else []
    quick_physics = ["--episodes", "4", "--timesteps", "1500"] if args.quick else []

    steps = [
        ("check", [str(SCRIPTS / "check_project.py")]),
        ("assets", [str(SCRIPTS / "prepare_assets.py")]),
        ("stage1", [str(SCRIPTS / "run_stage1.py"), *quick_stage1]),
        ("stage2", [str(SCRIPTS / "run_stage2.py"), *quick_physics]),
        ("stage3", [str(SCRIPTS / "run_stage3.py"), *quick_physics]),
        ("report", [str(SCRIPTS / "make_report.py")]),
    ]

    failed = []
    for name, argv in steps:
        if name in args.skip:
            print(f"\n-- {name}: skipped")
            continue
        if not run(name, argv):
            failed.append(name)
            if name in {"check", "assets"}:
                print("\naborting: the project cannot run without this step")
                return 1

    print("\n" + "=" * 70)
    if failed:
        print("failed steps:", ", ".join(failed))
        return 1
    print("all steps completed -- see docs/RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
