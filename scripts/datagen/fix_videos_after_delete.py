#!/usr/bin/env python3
"""Fix video files after delete_episodes.py removed episodes from parquet.

The deleted episodes' frames still exist in the video files.
This script removes them using ffmpeg.

Assumes:
  - Episodes were appended in order (new episodes come after all original episodes)
  - Original dataset had ORIGINAL_FRAME_COUNT frames before datagen added new ones

Usage
-----
    python scripts/datagen/fix_videos_after_delete.py \
        --dataset_path /root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset \
        --deleted_frames 3586 \
        --original_frames 351099
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def get_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def trim_video_remove_range(
    input_path: Path,
    output_path: Path,
    remove_start: int,
    remove_end: int,  # exclusive
    fps: float = 30.0,
) -> None:
    """Remove frames [remove_start, remove_end) from a video file."""
    total_frames = get_frame_count(input_path)
    print(f"  {input_path.name}: {total_frames} frames, removing [{remove_start}, {remove_end})")

    segments = []
    if remove_start > 0:
        segments.append((0, remove_start))
    if remove_end < total_frames:
        segments.append((remove_end, total_frames))

    if not segments:
        print(f"  Nothing to keep in {input_path.name} — deleting entire file.")
        output_path.write_bytes(b"")
        return

    if len(segments) == 1 and segments[0] == (0, total_frames):
        print(f"  {input_path.name} unchanged.")
        import shutil
        shutil.copy2(input_path, output_path)
        return

    with tempfile.TemporaryDirectory() as tmp:
        part_files = []
        for i, (start, end) in enumerate(segments):
            start_t = start / fps
            duration = (end - start) / fps
            part = Path(tmp) / f"part_{i}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(start_t),
                    "-i", str(input_path),
                    "-t", str(duration),
                    "-c", "copy",
                    str(part),
                ],
                check=True, capture_output=True,
            )
            part_files.append(part)

        if len(part_files) == 1:
            import shutil
            shutil.copy2(part_files[0], output_path)
        else:
            # Write concat list
            concat_list = Path(tmp) / "concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{p}'" for p in part_files)
            )
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_list),
                    "-c", "copy",
                    str(output_path),
                ],
                check=True, capture_output=True,
            )

    new_count = get_frame_count(output_path)
    print(f"  → {output_path.name}: {new_count} frames after trim")


def fix_videos(dataset_path: Path, deleted_frames: int, original_frames: int) -> None:
    """
    Remove `deleted_frames` frames starting at position `original_frames`
    from all video files in the dataset.
    """
    delete_start = original_frames          # first frame to delete (global)
    delete_end = original_frames + deleted_frames  # exclusive

    video_root = dataset_path / "videos"
    if not video_root.exists():
        print("No videos/ directory found — nothing to do.")
        return

    for camera_dir in sorted(video_root.iterdir()):
        if not camera_dir.is_dir():
            continue
        for chunk_dir in sorted(camera_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue

            print(f"\nProcessing {camera_dir.name}/{chunk_dir.name}/")
            video_files = sorted(chunk_dir.glob("file-*.mp4"))

            # Build cumulative frame counts to map global→local positions
            frame_counts = []
            for vf in video_files:
                frame_counts.append(get_frame_count(vf))
            cumulative = [0]
            for c in frame_counts:
                cumulative.append(cumulative[-1] + c)

            total_video_frames = cumulative[-1]
            print(f"  Total frames across {len(video_files)} files: {total_video_frames}")
            print(f"  Removing global frames [{delete_start}, {delete_end})")

            for i, vf in enumerate(video_files):
                file_start = cumulative[i]
                file_end = cumulative[i + 1]

                # Overlap between this file and the delete range
                overlap_start = max(file_start, delete_start)
                overlap_end = min(file_end, delete_end)

                if overlap_start >= overlap_end:
                    print(f"  {vf.name}: no overlap, skipping")
                    continue

                # Local indices within this file
                local_del_start = overlap_start - file_start
                local_del_end = overlap_end - file_start

                tmp_out = vf.with_suffix(".tmp.mp4")
                trim_video_remove_range(vf, tmp_out, local_del_start, local_del_end)
                tmp_out.replace(vf)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="/root/.cache/huggingface/lerobot/greenbeanleo/toyblock_synth_dataset",
    )
    parser.add_argument(
        "--deleted_frames", type=int, required=True,
        help="Number of frames that were deleted from the parquet (shown in delete_episodes.py output).",
    )
    parser.add_argument(
        "--original_frames", type=int, required=True,
        help="Total frames in the dataset BEFORE the deleted episodes were added "
             "(i.e. the HuggingFace dataset's original frame count).",
    )
    args = parser.parse_args()

    fix_videos(
        Path(args.dataset_path),
        args.deleted_frames,
        args.original_frames,
    )
    print("\nDone. You can now upload the dataset.")


if __name__ == "__main__":
    main()
