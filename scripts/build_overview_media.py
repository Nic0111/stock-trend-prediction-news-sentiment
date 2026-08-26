"""Build the active 1080p60 clips and 4K scene posters for Overview.

The repository keeps only the runtime 1080p60 clips and poster tiers. Original
source videos are supplied from outside the repository when a rebuild is
needed, so an old low-resolution media tree cannot be mistaken for a runtime
fallback or accidentally shipped with the frontend.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_overview_media.py \\
        --source-video-dir C:\\path\\to\\source-clips
    .venv\\Scripts\\python.exe scripts\\build_overview_media.py \\
        --source-video-dir C:\\path\\to\\source-clips --force

FFmpeg is resolved from --ffmpeg, PATH, or imageio-ffmpeg (installed in the
project virtual environment when available).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageFilter


SCENES = (
    "welcome-exchange",
    "market-observatory",
    "prediction-workshop",
    "newsroom-garden",
    "comparison-bridge",
    "learning-pavilion",
)


def resolve_ffmpeg(explicit: str | None) -> str:
    if explicit:
        return explicit
    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - only used on developer machines
        raise SystemExit(
            "FFmpeg is required. Install it, or install imageio-ffmpeg in the project "
            "environment with: python -m pip install imageio-ffmpeg"
        ) from exc


def encode_60fps(
    ffmpeg: str,
    source: Path,
    destination: Path,
    force: bool,
    interpolation: str,
    timeout_seconds: float,
) -> None:
    if destination.exists() and destination.stat().st_size > 2_000_000 and not force:
        print(f"skip video (already exists): {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    interpolation_filter = (
        "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:"
        "vsbmc=1:scd=fdiff"
        if interpolation == "mci"
        else "minterpolate=fps=60:mi_mode=blend"
    )
    vf = f"{interpolation_filter},scale=1920:1080:flags=lanczos,unsharp=5:5:0.35:5:5:0"
    temporary = destination.with_name(f"{destination.stem}.part{destination.suffix}")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(source),
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "60",
        "-g",
        "12",
        "-keyint_min",
        "12",
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    print(f"encode 60 fps: {source.name}")
    try:
        subprocess.run(command, check=True, timeout=timeout_seconds)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_poster(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"skip poster (already exists): {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB").resize((3840, 2160), Image.Resampling.LANCZOS)
        sharpened = rgb.filter(ImageFilter.UnsharpMask(radius=1.1, percent=35, threshold=3))
        sharpened.save(destination, "WEBP", quality=86, method=6)
    print(f"build 4K poster: {destination.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, default=Path("frontend/public/media/overview-world/v2"))
    parser.add_argument(
        "--source-video-dir",
        type=Path,
        required=True,
        help="External directory containing the original scene MP4s; it is not stored in the repository.",
    )
    parser.add_argument("--ffmpeg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scene", choices=SCENES, action="append", help="Build only the selected scene(s).")
    parser.add_argument("--interpolation", choices=("mci", "blend"), default="mci")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300,
        help="Abort an individual encode after this many seconds (default: 300).",
    )
    args = parser.parse_args()

    media_root = args.media_root.resolve()
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    source_video_dir = args.source_video_dir.resolve()
    output_video_dir = media_root / "video-1080-60"
    source_poster_dir = media_root / "poster"
    output_poster_dir = media_root / "poster-4k"

    missing = []
    selected_scenes = tuple(args.scene) if args.scene else SCENES
    for name in selected_scenes:
        source_video = source_video_dir / f"{name}.mp4"
        source_poster = source_poster_dir / f"{name}.webp"
        if not source_video.exists():
            missing.append(str(source_video))
        if not source_poster.exists():
            missing.append(str(source_poster))
    if missing:
        print("Missing Overview assets:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        return 2

    output_video_dir.mkdir(parents=True, exist_ok=True)
    for partial in output_video_dir.glob("*.part.mp4"):
        partial.unlink(missing_ok=True)

    for name in selected_scenes:
        encode_60fps(
            ffmpeg,
            source_video_dir / f"{name}.mp4",
            output_video_dir / f"{name}.mp4",
            args.force,
            args.interpolation,
            args.timeout_seconds,
        )
        build_poster(source_poster_dir / f"{name}.webp", output_poster_dir / f"{name}.webp", args.force)
    print("Overview media build complete. The runtime selects 1080p60 only when the browser reports it can decode smoothly; a seek watchdog prevents stalls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
