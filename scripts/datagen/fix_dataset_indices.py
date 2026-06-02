#!/usr/bin/env python3
"""Fix dataset_from_index / dataset_to_index in episodes metadata.

delete_episodes.py re-indexes the data parquet correctly but does NOT update
dataset_from_index / dataset_to_index in the episodes metadata. lerobot uses
these columns to map episode → global frame range. Stale values cause:

    IndexError: Invalid key: 690247 is out of bounds for size 688500

This script recomputes correct values from the data parquet (ground truth).

Usage
-----
    python3 scripts/datagen/fix_dataset_indices.py \
        --dataset_path ~/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset
"""

import argparse
from pathlib import Path

import pandas as pd


def load_parquet_dir(directory: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(directory.glob("file-*.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def save_parquet_dir(df: pd.DataFrame, directory: Path) -> None:
    existing = sorted(directory.glob("file-*.parquet"))
    n_files = len(existing)
    chunk = max(1, (len(df) + n_files - 1) // n_files)
    for i, fpath in enumerate(existing):
        part = df.iloc[i * chunk: (i + 1) * chunk]
        part.to_parquet(fpath, index=False)
        print(f"  Wrote {len(part)} rows → {fpath.name}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset_path", type=str,
        default="/root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    # -----------------------------------------------------------------------
    # Ground truth: per-episode frame counts from data parquet
    # -----------------------------------------------------------------------
    data_df = load_parquet_dir(dataset_path / "data" / "chunk-000")
    ep_frames = (
        data_df.groupby("episode_index")["frame_index"]
        .count()
        .rename("n_frames")
        .sort_index()
    )
    total_frames = ep_frames.sum()
    print(f"Data parquet: {len(ep_frames)} episodes, {total_frames} frames")

    # -----------------------------------------------------------------------
    # Compute correct from/to indices
    # -----------------------------------------------------------------------
    correct_from = {}
    correct_to = {}
    cursor = 0
    for ep_idx, n in ep_frames.items():
        correct_from[ep_idx] = cursor
        correct_to[ep_idx] = cursor + n
        cursor += n

    # -----------------------------------------------------------------------
    # Load episodes metadata and compare
    # -----------------------------------------------------------------------
    ep_meta_dir = dataset_path / "meta" / "episodes" / "chunk-000"
    ep_meta = load_parquet_dir(ep_meta_dir)
    ep_meta = ep_meta.sort_values("episode_index").reset_index(drop=True)

    fixes = 0
    for _, row in ep_meta.iterrows():
        ep = int(row["episode_index"])
        cur_from = int(row["dataset_from_index"])
        cur_to = int(row["dataset_to_index"])
        exp_from = correct_from[ep]
        exp_to = correct_to[ep]

        if cur_from != exp_from or cur_to != exp_to:
            fixes += 1
            if fixes <= 10:
                print(
                    f"  ep {ep}: from {cur_from}→{exp_from}  to {cur_to}→{exp_to}"
                )

    if fixes > 10:
        print(f"  ... and {fixes - 10} more")

    print(f"\nTotal fixes needed: {fixes}")

    if fixes == 0:
        print("Already correct — nothing to do.")
        return

    if args.dry_run:
        print("[dry-run] Would apply fixes. Re-run without --dry_run to write.")
        return

    # Apply fixes
    ep_meta["dataset_from_index"] = ep_meta["episode_index"].map(correct_from)
    ep_meta["dataset_to_index"] = ep_meta["episode_index"].map(correct_to)

    print("\nWriting fixed episodes metadata...")
    save_parquet_dir(ep_meta, ep_meta_dir)
    print("Done.")


if __name__ == "__main__":
    main()
