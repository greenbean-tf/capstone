#!/usr/bin/env python3
"""Generate a synthetic object_poses.json for ToyBlocksCollection.

Produces random block positions within the Franka robot's reachable workspace
so the state-machine has a high chance of succeeding without needing real
camera footage.

Coordinate system
-----------------
The loader applies:  world_x = ANCHOR_X + tvec[0]  (with anchor_yaw = 0)
                     world_y = ANCHOR_Y + tvec[1]
Inverse:             tvec[0] = world_x - ANCHOR_X
                     tvec[1] = world_y - ANCHOR_Y

``rvec`` is ignored because ``use_fixed_yaw=True`` in the task config.

Usage
-----
    python scripts/datagen/generate_synthetic_poses.py \\
        --num_episodes 200 \\
        --output data/synthetic/object_poses.json

Then pass the file to generate.py::

    python scripts/datagen/generate.py \\
        --task HCIS-ToyBlocksCollection-SingleArm-v0 \\
        --num_envs 1 --device cuda --enable_cameras \\
        --record --use_lerobot_recorder \\
        --lerobot_dataset_repo_id ${HF_USER}/<repo> \\
        --object_poses data/synthetic/object_poses.json
"""

import argparse
import json
import math
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Scene constants (from toy_blocks_collection_env_cfg.py)
# ---------------------------------------------------------------------------

ANCHOR_X: float = 0.35   # anchor_world_pose[0]
ANCHOR_Y: float = 0.0    # anchor_world_pose[1]

ROBOT_X: float = 0.35    # robot base x
ROBOT_Y: float = -0.74   # robot base y

BOX_X: float = 0.65      # storage_box init_state x (fixed)
BOX_Y: float = -0.55     # storage_box init_state y (fixed)

OBJECT_NAMES: tuple[str, ...] = ("green_block", "blue_block", "red_block")

# ---------------------------------------------------------------------------
# Workspace bounds (world frame)
# Table surface (counter_right_main_group) bounding box: X=[0.003, 0.703], Y=[-0.677, -0.027]
# Y_MIN set to -0.65 to keep a 2.7 cm margin from the table edge at Y=-0.677.
# Day-1 failure confirmed: Y ≈ 0 is outside effective reach (FSM fails).
# ---------------------------------------------------------------------------

WORKSPACE_X_MIN: float = 0.05
WORKSPACE_X_MAX: float = 0.60
WORKSPACE_Y_MIN: float = -0.65  # table Y min is -0.677; -0.65 keeps a 2.7 cm margin
WORKSPACE_Y_MAX: float = -0.28

# Franka reach window measured from robot base (x, y)
MIN_REACH: float = 0.15   # avoid base singularity
MAX_REACH: float = 0.62   # IK reliability limit

MIN_BLOCK_SPACING: float = 0.12   # min centre-to-centre distance between blocks
MIN_DIST_FROM_BOX: float = 0.20   # min distance from storage box centre

MAX_ATTEMPTS: int = 10_000        # rejection sampling limit per block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _world_to_tvec(wx: float, wy: float) -> list[float]:
    """Convert world XY → anchor-frame tvec (tvec[2] is ignored by loader)."""
    return [wx - ANCHOR_X, wy - ANCHOR_Y, 0.0]


def _sample_position(
    rng: random.Random,
    placed: list[tuple[float, float]],
) -> tuple[float, float]:
    """Rejection-sample one valid world (x, y) for a block."""
    for _ in range(MAX_ATTEMPTS):
        x = rng.uniform(WORKSPACE_X_MIN, WORKSPACE_X_MAX)
        y = rng.uniform(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)

        reach = _dist2d(x, y, ROBOT_X, ROBOT_Y)
        if reach < MIN_REACH or reach > MAX_REACH:
            continue

        if _dist2d(x, y, BOX_X, BOX_Y) < MIN_DIST_FROM_BOX:
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
                "rvec": [0.0, 0.0, 0.0],  # ignored: use_fixed_yaw=True
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
        description="Generate synthetic object_poses.json for ToyBlocksCollection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100,
        help="Number of episodes (rows) to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output", type=str, default="data/synthetic/object_poses.json",
        help="Destination path for the generated file.",
    )
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
    print(
        f"Constraints  reach=[{MIN_REACH}, {MAX_REACH}]m  "
        f"block_spacing>={MIN_BLOCK_SPACING}m  box_clearance>={MIN_DIST_FROM_BOX}m"
    )


if __name__ == "__main__":
    main()
