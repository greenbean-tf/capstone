#!/usr/bin/env python3
"""Generate a synthetic object_poses.json for ColorSortBlocks (Advanced level).

Same logic as generate_synthetic_poses.py but avoids all THREE basket positions
instead of just the single storage box used in Entry level.

Basket positions (fixed in scene):
  green_basket: (0.65, -0.55)  — right side
  blue_basket:  (0.35, -0.55)  — front/middle
  red_basket:   (0.05, -0.55)  — left side

Usage
-----
    python scripts/datagen/generate_synthetic_poses_advanced.py \\
        --num_episodes 1200 \\
        --seed 42 \\
        --output data/synthetic_advanced/object_poses.json

Then pass to generate.py:

    python scripts/datagen/generate.py \\
        --task HCIS-ColorSortBlocks-SingleArm-v0 \\
        --object_poses data/synthetic_advanced/object_poses.json ...
"""

import argparse
import json
import math
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Scene constants (from color_sort_blocks_env_cfg.py)
# ---------------------------------------------------------------------------

ANCHOR_X: float = 0.35
ANCHOR_Y: float = 0.0

ROBOT_X: float = 0.35
ROBOT_Y: float = -0.74

# All three basket positions that blocks must stay away from.
BASKET_POSITIONS: list[tuple[float, float]] = [
    (0.65, -0.55),  # green_basket
    (0.35, -0.55),  # blue_basket
    (0.05, -0.55),  # red_basket
]

OBJECT_NAMES: tuple[str, ...] = ("green_block", "blue_block", "red_block")

# ---------------------------------------------------------------------------
# Workspace bounds (same as Entry level)
# ---------------------------------------------------------------------------

WORKSPACE_X_MIN: float = 0.05
WORKSPACE_X_MAX: float = 0.60
# All baskets sit at Y=-0.55; basket +Y outer wall at -0.447 m (measured from USD).
# Y_MIN=-0.36: block edge is ~5 cm from basket outer wall, enough for arm approach.
# Y_MAX=-0.15: just within robot reach (MAX_REACH=0.58 m from Y=-0.74 → -0.16 m);
# the reach check filters edge cases, giving ~0.21 m of usable Y range.
WORKSPACE_Y_MIN: float = -0.36
WORKSPACE_Y_MAX: float = -0.15

MIN_REACH: float = 0.30
MAX_REACH: float = 0.58

MIN_BLOCK_SPACING: float = 0.15

MAX_ATTEMPTS: int = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _world_to_tvec(wx: float, wy: float) -> list[float]:
    return [wx - ANCHOR_X, wy - ANCHOR_Y, 0.0]


def _sample_position(
    rng: random.Random,
    placed: list[tuple[float, float]],
) -> tuple[float, float]:
    for _ in range(MAX_ATTEMPTS):
        x = rng.uniform(WORKSPACE_X_MIN, WORKSPACE_X_MAX)
        y = rng.uniform(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)

        reach = _dist2d(x, y, ROBOT_X, ROBOT_Y)
        if reach < MIN_REACH or reach > MAX_REACH:
            continue

        if any(_dist2d(x, y, px, py) < MIN_BLOCK_SPACING for px, py in placed):
            continue

        return x, y

    raise RuntimeError(
        f"Could not place a block after {MAX_ATTEMPTS} attempts — "
        "workspace bounds or spacing constraints may be too tight."
    )


def _generate_episode(rng: random.Random, idx: int) -> dict:
    placed: list[tuple[float, float]] = []
    objects = []
    for name in OBJECT_NAMES:
        x, y = _sample_position(rng, placed)
        placed.append((x, y))
        objects.append(
            {
                "object_name": name,
                "rvec": [0.0, 0.0, 0.0],
                "tvec": _world_to_tvec(x, y),
            }
        )
    return {
        "video_name": f"synthetic_{idx:05d}",
        "episode_range": [0, 100],
        "objects": objects,
        "status": "full",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic object_poses.json for ColorSortBlocks (Advanced).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic_advanced/object_poses.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    episodes = [_generate_episode(rng, i) for i in range(args.num_episodes)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(episodes, indent=2))

    print(f"Wrote {len(episodes)} synthetic episodes → {out}")
    print(
        f"Workspace  X=[{WORKSPACE_X_MIN}, {WORKSPACE_X_MAX}]  "
        f"Y=[{WORKSPACE_Y_MIN}, {WORKSPACE_Y_MAX}]  (world frame)"
    )


if __name__ == "__main__":
    main()
