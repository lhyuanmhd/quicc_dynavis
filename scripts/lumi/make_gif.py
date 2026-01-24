#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image

# Example:
# Ek_1.0e-05_q2.0_Ra2.0e+03_run2_0005_snapshots.png
PAT = re.compile(r".*_run(?P<run>\d+)_(?P<tag>\d{4})_snapshots\.png$")


def collect_frames(fig_dir: Path) -> List[Tuple[int, int, Path]]:
    frames: List[Tuple[int, int, Path]] = []
    for p in fig_dir.glob("*_snapshots.png"):
        m = PAT.match(p.name)
        if not m:
            continue
        run = int(m.group("run"))
        tag = int(m.group("tag"))
        frames.append((run, tag, p))
    frames.sort(key=lambda x: (x[0], x[1]))
    return frames


def load_and_prepare(
    path: Path,
    resize_max: Optional[int],
    quantize: bool,
) -> Image.Image:
    im = Image.open(path).convert("RGBA")

    # Optional downscale to keep GIF size manageable
    if resize_max is not None:
        w, h = im.size
        scale = min(resize_max / float(w), resize_max / float(h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)

    # GIF is palette-based; quantize helps reduce size
    if quantize:
        im = im.convert("P", palette=Image.ADAPTIVE, colors=256)

    return im


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make an animated GIF from snapshot figures ordered by run index then tag (no ffmpeg)."
    )
    ap.add_argument("--case-dir", default=".", help="Case directory (default: .)")
    ap.add_argument("--figures-dir", default=None, help="Figures directory (default: <case-dir>/figures)")
    ap.add_argument("--out", default=None, help="Output GIF (default: <case-dir>/movies/snapshots.gif)")
    ap.add_argument("--fps", type=float, default=5.0, help="Playback speed (frames per second, default: 5)")
    ap.add_argument("--resize-max", type=int, default=1280,
                    help="Max width/height for frames (default: 1280). Use 0 to disable resize.")
    ap.add_argument("--no-quantize", action="store_true",
                    help="Disable palette quantization (may increase size).")
    ap.add_argument("--dry-run", action="store_true", help="Print final order only; do not create GIF.")
    args = ap.parse_args()

    case_dir = Path(args.case_dir).resolve()
    fig_dir = Path(args.figures_dir).resolve() if args.figures_dir else (case_dir / "figures")
    if not fig_dir.is_dir():
        raise FileNotFoundError(f"Figures directory not found: {fig_dir}")

    frames = collect_frames(fig_dir)
    if not frames:
        raise FileNotFoundError(
            f"No matching frames found in {fig_dir}\nExpected: *_runX_YYYY_snapshots.png"
        )

    print("========================================")
    print(f"[INFO] figures_dir: {fig_dir}")
    print(f"[INFO] frames     : {len(frames)}")
    print(f"[INFO] first      : {frames[0][2].name}")
    print(f"[INFO] last       : {frames[-1][2].name}")
    print("========================================")

    if args.dry_run:
        for run, tag, p in frames:
            print(f"run{run:02d} tag{tag:04d}  {p.name}")
        return

    out_gif = Path(args.out).resolve() if args.out else (case_dir / "movies" / "snapshots.gif")
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    resize_max = None if args.resize_max == 0 else args.resize_max
    quantize = not args.no_quantize

    duration_ms = int(round(1000.0 / float(args.fps)))  # per-frame duration

    # Load frames (may take time and memory if there are many / very large PNGs)
    images: List[Image.Image] = []
    for run, tag, p in frames:
        im = load_and_prepare(p, resize_max=resize_max, quantize=quantize)
        images.append(im)

    # Save animated GIF
    first, rest = images[0], images[1:]
    first.save(
        out_gif,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )

    print(f"[OK] GIF written: {out_gif}")
    print(f"[INFO] fps={args.fps}, duration_ms/frame={duration_ms}, resize_max={args.resize_max}, quantize={quantize}")


if __name__ == "__main__":
    main()

