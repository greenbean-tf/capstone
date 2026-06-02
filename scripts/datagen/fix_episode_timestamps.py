#!/usr/bin/env python3
"""Fix from_timestamp/to_timestamp in episodes metadata after video trimming.

When fix_videos_after_delete.py trims a video file, it removes frames from
the beginning of the file. But the episodes metadata parquet still has the
old from_timestamp/to_timestamp values (which include the deleted frames'
duration as an offset). This causes FrameTimestampError during training.

This script recomputes correct from_timestamp/to_timestamp for all episodes
based on the actual frame counts in the data parquet (which are correct).

Usage
-----
    python3 scripts/datagen/fix_episode_timestamps.py \
        --dataset_path ~/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_parquet_dir(directory: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(directory.glob("file-*.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def save_parquet_dir(df: pd.DataFrame, directory: Path) -> None:
    """Write back to the same number of files with same approximate sizes."""
    existing = sorted(directory.glob("file-*.parquet"))
    n_files = len(existing)
    total = len(df)
    chunk = max(1, (total + n_files - 1) // n_files)
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
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would change without writing.",
    )
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    fps = json.loads((dataset_path / "meta" / "info.json").read_text()).get("fps", 30.0)
    print(f"fps = {fps}")

    # -----------------------------------------------------------------------
    # Load data parquet — per-episode frame counts (source of truth)
    # -----------------------------------------------------------------------
    data_df = load_parquet_dir(dataset_path / "data" / "chunk-000")
    ep_frames = (
        data_df.groupby("episode_index")["frame_index"]
        .count()
        .rename("n_frames")
    )  # index = episode_index

    # -----------------------------------------------------------------------
    # Load episodes metadata
    # -----------------------------------------------------------------------
    ep_meta_dir = dataset_path / "meta" / "episodes" / "chunk-000"
    ep_meta = load_parquet_dir(ep_meta_dir)
    ep_meta = ep_meta.sort_values("episode_index").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Identify camera video keys that have from/to timestamp columns
    # -----------------------------------------------------------------------
    camera_keys = []
    for col in ep_meta.columns:
        if col.endswith("/from_timestamp"):
            key = col.replace("/from_timestamp", "")
            camera_keys.append(key)
    print(f"Camera keys with timestamps: {camera_keys}")

    # -----------------------------------------------------------------------
    # For each camera key, recompute from_timestamp and to_timestamp
    # grouped by file_index (each file is a concatenated video)
    # -----------------------------------------------------------------------
    ep_meta_fixed = ep_meta.copy()
    total_fixes = 0

    for cam_key in camera_keys:
        file_idx_col = f"{cam_key}/file_index"
        from_col = f"{cam_key}/from_timestamp"
        to_col = f"{cam_key}/to_timestamp"

        if file_idx_col not in ep_meta.columns:
            print(f"  WARNING: {file_idx_col} not in metadata — skipping")
            continue

        print(f"\n=== {cam_key} ===")

        for file_idx, group in ep_meta.groupby(file_idx_col):
            group = group.sort_values("episode_index")
            ep_indices = group["episode_index"].tolist()

            # Recompute cumulative timestamps from frame counts
            cumulative_s = 0.0
            for row_i, ep_idx in enumerate(ep_indices):
                n = int(ep_frames.get(ep_idx, 0))
                if n == 0:
                    print(f"  WARNING: episode {ep_idx} has 0 frames in parquet!")
                    continue

                correct_from = cumulative_s
                correct_to = cumulative_s + n / fps
                cumulative_s = correct_to

                current_from = float(group.loc[group["episode_index"] == ep_idx, from_col].iloc[0])
                current_to = float(group.loc[group["episode_index"] == ep_idx, to_col].iloc[0])

                delta_from = abs(correct_from - current_from)
                delta_to = abs(correct_to - current_to)

                if delta_from > 0.01 or delta_to > 0.01:
                    print(
                        f"  file-{int(file_idx):03d}  ep {ep_idx}: "
                        f"from {current_from:.3f}→{correct_from:.3f}  "
                        f"to {current_to:.3f}→{correct_to:.3f}  "
                        f"(Δfrom={delta_from:.3f}s)"
                    )
                    total_fixes += 1
                    ep_mask = ep_meta_fixed["episode_index"] == ep_idx
                    ep_meta_fixed.loc[ep_mask, from_col] = correct_from
                    ep_meta_fixed.loc[ep_mask, to_col] = correct_to

    print(f"\nTotal timestamp fixes: {total_fixes}")

    if total_fixes == 0:
        print("No fixes needed — timestamps are already correct.")
        return

    if args.dry_run:
        print("\n[dry-run] Would write fixes. Re-run without --dry_run to apply.")
        return

    print("\nWriting fixed episodes metadata...")
    save_parquet_dir(ep_meta_fixed, ep_meta_dir)
    print("Done. Upload the dataset to HuggingFace to apply the fix remotely.")


if __name__ == "__main__":
    main()
