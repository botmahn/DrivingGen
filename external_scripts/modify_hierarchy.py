#!/usr/bin/env python3

import argparse
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


def run_ffmpeg_decode(video_path: Path, images_dir: Path, fps: int, overwrite: bool):
    images_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for p in images_dir.glob("*.png"):
            p.unlink()

    output_pattern = images_dir / "%05d.png"

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(output_pattern),
    ]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert flat DrivingGen video outputs into nested folder hierarchy."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing flat .mp4 files.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output root folder where ego_condition/ will be created.",
    )
    parser.add_argument(
        "--condition",
        default="ego_condition",
        help="Condition folder name. Default: ego_condition",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Model folder name, e.g. ltx-video-13bx, wan2.2-5bx.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS used for decoding frames. Default: 10",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing video.mp4 and decoded frames.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move videos instead of copying them. Default is copy.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    assert input_dir.exists(), f"Input dir does not exist: {input_dir}"

    videos = sorted(input_dir.glob("*.mp4"))

    if len(videos) == 0:
        raise RuntimeError(f"No .mp4 files found in {input_dir}")

    for src_video in tqdm(videos, desc="Converting videos"):
        video_stem = src_video.stem

        out_dir = (
            output_root
            / args.condition
            / video_stem
            / args.model_name
            / "folder_run"
        )

        images_dir = out_dir / "images"
        out_video = out_dir / "video.mp4"

        out_dir.mkdir(parents=True, exist_ok=True)

        if out_video.exists() and not args.overwrite:
            print(f"Skipping existing video: {out_video}")
        else:
            if args.move:
                shutil.move(str(src_video), str(out_video))
            else:
                shutil.copy2(src_video, out_video)

        run_ffmpeg_decode(
            video_path=out_video,
            images_dir=images_dir,
            fps=args.fps,
            overwrite=args.overwrite,
        )

    print("Done.")


if __name__ == "__main__":
    main()
