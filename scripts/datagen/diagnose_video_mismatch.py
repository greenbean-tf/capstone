#!/usr/bin/env python3
"""Diagnose video/parquet timestamp mismatches in a lerobot dataset.

Identifies which episodes are in each video file, checks for timestamp
mismatches caused by incorrect video trimming, and outputs the exact
episodes to delete to fix the dataset.

Usage
-----
    python3 scripts/datagen/diagnose_video_mismatch.py \
        --dataset_path /root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def ffprobe_frames(video_path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return int(result.stdout.strip()) if result.stdout.strip() else 0


def ffprobe_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip()) if result.stdout.strip() else 0.0


def load_parquet_dir(directory: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(directory.glob("file-*.parquet")):
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset_path", type=str,
        default="/root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset",
    )
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    # -----------------------------------------------------------------------
    # 1. info.json
    # -----------------------------------------------------------------------
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    fps = info.get("fps", 30.0)
    print("=" * 60)
    print("info.json")
    print("=" * 60)
    print(f"  fps               : {fps}")
    print(f"  total_episodes    : {info.get('total_episodes')}")
    print(f"  total_frames      : {info.get('total_frames')}")
    print(f"  chunks_size       : {info.get('chunks_size', 'N/A')}")
    print(f"  video_chunk_size  : {info.get('video_chunk_size', 'N/A')}")
    print()

    # -----------------------------------------------------------------------
    # 2. Data parquet — episodes summary
    # -----------------------------------------------------------------------
    data_dir = dataset_path / "data" / "chunk-000"
    df = load_parquet_dir(data_dir)
    print("=" * 60)
    print(f"Data parquet: {len(df)} frames, columns:")
    print("=" * 60)
    print(" ", list(df.columns))
    print()

    # Per-episode: first/last global index, frame count, max timestamp
    ep_summary = (
        df.groupby("episode_index")
        .agg(
            first_global=("index", "min"),
            last_global=("index", "max"),
            n_frames=("index", "count"),
            max_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    print(f"Total episodes in parquet: {len(ep_summary)}")
    print()

    # -----------------------------------------------------------------------
    # 3. Episodes metadata parquet
    # -----------------------------------------------------------------------
    ep_meta_dir = dataset_path / "meta" / "episodes" / "chunk-000"
    ep_meta = load_parquet_dir(ep_meta_dir)
    print("=" * 60)
    print("Episodes metadata parquet columns:")
    print("=" * 60)
    print(" ", list(ep_meta.columns))
    if not ep_meta.empty:
        print()
        print("Sample (first 3 rows):")
        print(ep_meta.head(3).to_string())
    print()

    # -----------------------------------------------------------------------
    # 4. Video files — frame counts and durations
    # -----------------------------------------------------------------------
    video_root = dataset_path / "videos"
    camera_dirs = sorted(
        [d for d in video_root.iterdir() if d.is_dir()]
    ) if video_root.exists() else []

    print("=" * 60)
    print("Video files")
    print("=" * 60)

    # We'll use the first camera for the episode→file mapping
    primary_camera = None
    video_file_info = []  # list of (file_path, n_frames, duration)

    for camera_dir in camera_dirs:
        for chunk_dir in sorted(camera_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            video_files = sorted(chunk_dir.glob("file-*.mp4"))
            print(f"\n  {camera_dir.name}/{chunk_dir.name}/  ({len(video_files)} files)")

            cumulative = 0
            cam_info = []
            for vf in video_files:
                n = ffprobe_frames(vf)
                d = ffprobe_duration(vf)
                print(f"    {vf.name}: {n:>7} frames  {d:>8.3f}s")
                cam_info.append((vf, n, d))
                cumulative += n
            print(f"    TOTAL: {cumulative} frames  (parquet has {len(df)} frames)")

            if primary_camera is None:
                primary_camera = camera_dir.name
                video_file_info = cam_info

    print()

    # -----------------------------------------------------------------------
    # 5. Map episodes → video files
    #    Strategy: accumulate frame counts across video files and match
    #    against episode global frame ranges.
    # -----------------------------------------------------------------------
    if not video_file_info:
        print("No video files found — cannot check mapping.")
        return

    print("=" * 60)
    print(f"Episode → video file mapping  (camera: {primary_camera})")
    print("=" * 60)

    # Build cumulative frame boundaries for video files
    vid_cumulative = [0]
    for _, n, _ in video_file_info:
        vid_cumulative.append(vid_cumulative[-1] + n)

    def frame_to_video_idx(global_frame: int) -> int:
        for i in range(len(video_file_info)):
            if vid_cumulative[i] <= global_frame < vid_cumulative[i + 1]:
                return i
        return len(video_file_info) - 1  # clamp to last

    # Group episodes by video file
    ep_summary["video_idx"] = ep_summary["first_global"].apply(frame_to_video_idx)

    # -----------------------------------------------------------------------
    # 6. Check for mismatches: expected end timestamp vs actual video duration
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Mismatch check — episodes whose frames exceed video duration")
    print("=" * 60)

    bad_episodes = []

    for vid_idx, (vf, n_frames, duration) in enumerate(video_file_info):
        eps_in_file = ep_summary[ep_summary["video_idx"] == vid_idx]
        if eps_in_file.empty:
            continue

        # Compute expected total frames from parquet for this video file
        expected_frames = eps_in_file["n_frames"].sum()
        actual_frames = n_frames

        # Compute where each episode starts within the video file (in frames)
        # = first_global - vid_cumulative[vid_idx]
        eps_in_file = eps_in_file.copy()
        eps_in_file["file_frame_start"] = eps_in_file["first_global"] - vid_cumulative[vid_idx]
        eps_in_file["file_frame_end"] = eps_in_file["last_global"] - vid_cumulative[vid_idx]
        eps_in_file["expected_ts_end"] = eps_in_file["file_frame_end"] / fps

        # Check if any episode's expected end exceeds actual video duration
        over = eps_in_file[eps_in_file["expected_ts_end"] > duration + 0.1]

        status = "OK"
        if expected_frames != actual_frames:
            status = f"FRAME MISMATCH: parquet={expected_frames} video={actual_frames}"
        if not over.empty:
            status = f"TIMESTAMP OVERFLOW: {len(over)} episodes exceed duration {duration:.2f}s"

        print(f"\n  {vf.name}  ({n_frames} frames, {duration:.3f}s)")
        print(f"    episodes {int(eps_in_file['episode_index'].min())}–"
              f"{int(eps_in_file['episode_index'].max())}  "
              f"({len(eps_in_file)} eps, {expected_frames} parquet frames)")
        print(f"    status: {status}")

        if not over.empty:
            print(f"    OVER-range episodes:")
            for _, row in over.iterrows():
                print(f"      ep {int(row['episode_index'])}: "
                      f"file_frames [{int(row['file_frame_start'])}, {int(row['file_frame_end'])}] "
                      f"expected_ts_end={row['expected_ts_end']:.3f}s > video={duration:.3f}s")
                bad_episodes.append(int(row["episode_index"]))

    # -----------------------------------------------------------------------
    # 7. Summary and fix commands
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not bad_episodes:
        print("  No timestamp mismatches found!")
        print("  If training still fails, the issue may be intermittent. Re-download and retry.")
    else:
        print(f"  {len(bad_episodes)} episodes have timestamps that exceed the video duration:")
        print(f"  Episodes: {bad_episodes}")
        print()
        print("  FIX: Delete these episodes and re-upload")
        print()
        eps_str = " ".join(str(e) for e in bad_episodes)
        print("  Step 1: Delete from parquet")
        print(f"    python3 scripts/datagen/delete_episodes.py \\")
        print(f"        --dataset_path {dataset_path} \\")
        print(f"        --episodes {eps_str}")
        print()
        print("  Step 2: Re-upload")
        print(f"    hf upload greenbeanleo/toyblock_synth_dataset \\")
        print(f"        {dataset_path}/ \\")
        print(f"        --repo-type dataset")
        print()
        print("  NOTE: Also consider deleting the entire affected video file(s)")
        print("  if the above does not fully resolve the mismatch.")


if __name__ == "__main__":
    main()
