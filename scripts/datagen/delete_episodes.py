#!/usr/bin/env python3
"""Delete specific episodes from a local lerobot dataset cache.

Removes the specified episode indices, re-numbers all subsequent episodes,
updates the global frame index, renames video files, and patches info.json.

Usage
-----
    # Delete a single episode (0-indexed)
    python scripts/datagen/delete_episodes.py \
        --dataset_path /root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset \
        --episodes 175

    # Delete a range of episodes (inclusive)
    python scripts/datagen/delete_episodes.py \
        --dataset_path /root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset \
        --episodes 175 176 177
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def delete_episodes(dataset_path: Path, episodes_to_delete: list[int]) -> None:
    episodes_to_delete = sorted(set(episodes_to_delete))
    print(f"Deleting episodes: {episodes_to_delete}")

    data_dir = dataset_path / "data" / "chunk-000"
    meta_ep_dir = dataset_path / "meta" / "episodes" / "chunk-000"
    info_path = dataset_path / "meta" / "info.json"

    # ------------------------------------------------------------------
    # Step 1: Read all data frames, count deleted frames
    # ------------------------------------------------------------------
    all_frames = []
    for f in sorted(data_dir.glob("file-*.parquet")):
        all_frames.append(pd.read_parquet(f))
    df = pd.concat(all_frames, ignore_index=True)

    frames_to_delete = df["episode_index"].isin(episodes_to_delete).sum()
    print(f"Frames to delete: {frames_to_delete}")

    # ------------------------------------------------------------------
    # Step 2: Filter and re-index
    # ------------------------------------------------------------------
    df = df[~df["episode_index"].isin(episodes_to_delete)].copy()

    # Build episode_index remapping: old → new
    all_old = sorted(df["episode_index"].unique().tolist())
    old_to_new = {old: new for new, old in enumerate(all_old)}
    df["episode_index"] = df["episode_index"].map(old_to_new)

    # Recalculate global "index" column
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    df["index"] = df.index

    # ------------------------------------------------------------------
    # Step 3: Write data parquet files back (one file per original file)
    # ------------------------------------------------------------------
    # Distribute rows back proportionally into the same number of files
    parquet_files = sorted(data_dir.glob("file-*.parquet"))
    n_files = len(parquet_files)
    chunk_size = max(1, len(df) // n_files + 1)
    for i, f in enumerate(parquet_files):
        chunk = df.iloc[i * chunk_size: (i + 1) * chunk_size]
        chunk.to_parquet(f, index=False)
        print(f"  Wrote {len(chunk)} rows → {f.name}")

    # ------------------------------------------------------------------
    # Step 4: Update meta/episodes parquet files
    # ------------------------------------------------------------------
    all_meta = []
    for f in sorted(meta_ep_dir.glob("file-*.parquet")):
        all_meta.append(pd.read_parquet(f))
    meta_df = pd.concat(all_meta, ignore_index=True)

    meta_df = meta_df[~meta_df["episode_index"].isin(episodes_to_delete)].copy()
    meta_df["episode_index"] = meta_df["episode_index"].map(old_to_new)
    meta_df = meta_df.sort_values("episode_index").reset_index(drop=True)

    meta_files = sorted(meta_ep_dir.glob("file-*.parquet"))
    n_meta = len(meta_files)
    m_chunk = max(1, len(meta_df) // n_meta + 1)
    for i, f in enumerate(meta_files):
        chunk = meta_df.iloc[i * m_chunk: (i + 1) * m_chunk]
        chunk.to_parquet(f, index=False)
        print(f"  Wrote {len(chunk)} episode rows → {f.name}")

    # ------------------------------------------------------------------
    # Step 5: Rename video files
    # ------------------------------------------------------------------
    video_root = dataset_path / "videos"
    if video_root.exists():
        for video_key_dir in video_root.iterdir():
            if not video_key_dir.is_dir():
                continue
            for chunk_dir in video_key_dir.iterdir():
                if not chunk_dir.is_dir():
                    continue
                # Delete target episodes' videos
                for ep in episodes_to_delete:
                    vid = chunk_dir / f"episode_{ep:06d}.mp4"
                    if vid.exists():
                        vid.unlink()
                        print(f"  Deleted {vid.name}")
                # Rename remaining videos according to remapping
                # Sort in ascending order to avoid overwriting
                existing = sorted(chunk_dir.glob("episode_*.mp4"))
                for vid in existing:
                    old_idx = int(vid.stem.replace("episode_", ""))
                    if old_idx in old_to_new:
                        new_idx = old_to_new[old_idx]
                        if new_idx != old_idx:
                            new_path = chunk_dir / f"episode_{new_idx:06d}.mp4"
                            vid.rename(new_path)
                            print(f"  Renamed {vid.name} → {new_path.name}")

    # ------------------------------------------------------------------
    # Step 6: Update info.json
    # ------------------------------------------------------------------
    info = json.loads(info_path.read_text())
    info["total_episodes"] -= len(episodes_to_delete)
    info["total_frames"] -= int(frames_to_delete)
    info_path.write_text(json.dumps(info, indent=2))
    print(f"\nDone. info.json updated: "
          f"episodes={info['total_episodes']}, frames={info['total_frames']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete specific episodes from a local lerobot dataset cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="/root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset",
        help="Path to the local lerobot dataset cache.",
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", required=True,
        help="Episode indices to delete (0-indexed). E.g. --episodes 175 176",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    delete_episodes(dataset_path, args.episodes)


if __name__ == "__main__":
    main()
