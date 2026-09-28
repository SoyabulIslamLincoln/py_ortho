"""Thermal imagery: radiometric R-JPEG decoding and colour palettes.

DJI thermal cameras (M30T, M3T, M4T, H20T, ...) store the raw 16-bit sensor image in the
JPEG's APP3 segments. The pipeline mosaics those raw values (not the palette colours, which
must never be blended) and applies a palette only to the final product. A raw-value raster
is written alongside, so the palette can be changed later without re-processing:

    python -m orthomosaic.thermal recolor ortho_thermal.tif -o ortho_iron.tif --palette iron
    python -m orthomosaic.thermal list
"""
from __future__ import annotations

import argparse
import logging
import math
import struct
from typing import Optional, Sequence, Union

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_PALETTE = "rainbow"

# Colour stops (evenly spaced from cold to hot), interpolated to a 256-entry lookup table.
PALETTES = {
    "rainbow": [(0, 0, 135), (0, 0, 255), (0, 160, 255), (0, 255, 200), (60, 255, 0), (255, 255, 0),
                (255, 150, 0), (255, 40, 0), (180, 0, 0)],
    "rainbow_hc": [(0, 0, 0), (40, 0, 120), (0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 255, 0),
                   (255, 128, 0), (255, 0, 0), (255, 0, 255), (255, 255, 255)],
    "iron": [(0, 0, 0), (32, 0, 110), (100, 0, 150), (165, 0, 150), (215, 40, 90), (240, 100, 20),
             (250, 160, 0), (255, 215, 60), (255, 255, 230)],
    "white_hot": [(0, 0, 0), (255, 255, 255)],
    "black_hot": [(255, 255, 255), (0, 0, 0)],
    "arctic": [(0, 0, 40), (0, 30, 120), (20, 90, 190), (90, 170, 230), (200, 225, 245), (255, 230, 150),
               (255, 170, 30), (255, 110, 0)],
    "lava": [(0, 0, 0), (40, 0, 60), (110, 0, 80), (180, 20, 30), (230, 80, 0), (255, 160, 0),
             (255, 230, 100), (255, 255, 255)],
    "hot_metal": [(0, 0, 0), (120, 0, 0), (230, 50, 0), (255, 160, 0), (255, 240, 120), (255, 255, 255)],
    "medical": [(0, 0, 0), (0, 0, 180), (0, 170, 255), (0, 200, 0), (255, 255, 0), (255, 120, 0),
                (230, 0, 0), (255, 255, 255)],
    "green_hot": [(0, 0, 0), (0, 90, 0), (40, 200, 40), (200, 255, 200)],
}
ALIASES = {"ironbow": "iron", "iron_red": "iron", "ironred": "iron", "whitehot": "white_hot",
           "blackhot": "black_hot", "grayscale": "white_hot", "greyscale": "white_hot", "gray": "white_hot",
           "rainbow_high_contrast": "rainbow_hc", "hotmetal": "hot_metal", "fusion": "lava"}

PaletteSpec = Union[str, Sequence[Sequence[int]], np.ndarray]


def palette_names() -> list[str]:
    return sorted(PALETTES)


def palette_lut(palette: PaletteSpec = DEFAULT_PALETTE) -> np.ndarray:
    """(256, 3) uint8 lookup table from a palette name, a list of RGB stops, or a (256, 3) table."""
    if isinstance(palette, str):
        key = ALIASES.get(palette.lower().replace("-", "_").replace(" ", "_"), palette.lower())
        if key not in PALETTES:
            raise ValueError(f"Unknown palette '{palette}'. Available: {', '.join(palette_names())}")
        stops = np.asarray(PALETTES[key], np.float64)
    else:
        stops = np.asarray(palette, np.float64)
        if stops.shape == (256, 3):
            return np.clip(stops, 0, 255).astype(np.uint8)
    if stops.ndim != 2 or stops.shape[1] != 3 or len(stops) < 2:
        raise ValueError("a custom palette must be a list of at least two (R, G, B) stops")
    x = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, stops[:, c]) for c in range(3)], 1).round().astype(np.uint8)


def apply_palette(gray: np.ndarray, palette: PaletteSpec = DEFAULT_PALETTE) -> np.ndarray:
    """uint8 index image (..., ) -> (..., 3) RGB."""
    return palette_lut(palette)[np.asarray(gray, np.uint8)]


def colorize_rgba(rgba: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Replace the (gray) RGB of an RGBA block by palette colours; transparent stays transparent."""
    out = rgba.copy()
    out[..., :3] = lut[rgba[..., 0]]
    out[rgba[..., 3] == 0, :3] = 0
    return out


# --------------------------------------------------------------------------
# radiometric R-JPEG
# --------------------------------------------------------------------------

def _app3_segments(path: str, read_data: bool):
    """Walk JPEG markers up to the scan; return (total APP3 payload length, payload or None, size)."""
    total, chunks = 0, []
    with open(path, "rb") as fh:
        if fh.read(2) != b"\xff\xd8":
            return 0, None
        while True:
            hdr = fh.read(4)
            if len(hdr) < 4 or hdr[0] != 0xFF:
                break
            marker, length = hdr[1], struct.unpack(">H", hdr[2:4])[0]
            if marker == 0xDA:          # start of scan: metadata is over
                break
            if marker == 0xE3:
                if read_data:
                    chunks.append(fh.read(length - 2))
                else:
                    fh.seek(length - 2, 1)
                total += length - 2
            else:
                fh.seek(length - 2, 1)
    return total, (b"".join(chunks) if read_data else None)


def raw_thermal_shape(path: str, width: int, height: int) -> Optional[tuple[int, int]]:
    """(h, w) of the raw 16-bit thermal image embedded in a DJI R-JPEG, or None."""
    total, _ = _app3_segments(path, read_data=False)
    if total < 2 * 64 * 64 or total % 2:
        return None
    n = total // 2
    for k in (1, 2, 4):                        # raw is the JPEG size or an integer fraction of it
        if width % k == 0 and height % k == 0 and (width // k) * (height // k) == n:
            return height // k, width // k
    return None


def read_raw_thermal(path: str, shape: tuple[int, int]) -> np.ndarray:
    _, data = _app3_segments(path, read_data=True)
    h, w = shape
    return np.frombuffer(data[:2 * h * w], "<u2").reshape(h, w)


def raw_to_gray(raw: np.ndarray, value_range: tuple[float, float], size: tuple[int, int]) -> np.ndarray:
    """Normalise raw sensor values to uint8 with the survey-wide range and resize to (w, h)."""
    from PIL import Image
    lo, hi = value_range
    g = (raw.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6))
    im = Image.fromarray(np.clip(g, 0, 255).astype(np.float32), "F")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR if size[0] >= im.size[0] else Image.BOX)
    return np.clip(np.asarray(im) + 0.5, 0, 255).astype(np.uint8)


def survey_range(frames, sample: int = 40, pct=(0.5, 99.5)) -> tuple[float, float]:
    """Common raw-value range over a sample of the survey (so every image uses the same scale)."""
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).round().astype(int)
    vals = []
    for i in sorted(set(idx.tolist())):
        fr = frames[i]
        raw = read_raw_thermal(fr.path, fr.raw_shape)
        vals.append(raw[::4, ::4].ravel())
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, pct)
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------

def legend(path: str, palette: PaletteSpec, value_range: tuple[float, float], label: str = "raw sensor value"):
    """Small colour-bar PNG with the value range."""
    from PIL import Image, ImageDraw
    lut = palette_lut(palette)
    W, H, bar = 360, 70, 22
    img = Image.new("RGB", (W, H), "white")
    strip = np.repeat(lut[np.linspace(0, 255, W - 40).astype(int)][None], bar, 0)
    img.paste(Image.fromarray(strip), (20, 10))
    d = ImageDraw.Draw(img)
    lo, hi = value_range
    for frac in (0.0, 0.5, 1.0):
        x = 20 + frac * (W - 41)
        d.line([(x, 10 + bar), (x, 16 + bar)], fill="black")
        txt = f"{lo + frac * (hi - lo):.0f}"
        d.text((x - 4 * len(txt) / 2 * 1.5, 18 + bar), txt, fill="black")
    d.text((20, H - 14), f"{label} ({palette if isinstance(palette, str) else 'custom'})", fill="black")
    img.save(path)


def recolor(thermal_tif: str, out_tif: str, palette: PaletteSpec = DEFAULT_PALETTE,
            value_range: Optional[tuple[float, float]] = None, legend_png: Optional[str] = None) -> str:
    """Re-render a raw-value thermal GeoTIFF (from build_orthomosaic / build_3d) with another palette."""
    from .geotiff import GeoTIFFWriter, read_geotiff
    val, geo = read_geotiff(thermal_tif)
    valid = np.isfinite(val)
    if value_range is None:
        value_range = tuple(float(x) for x in np.percentile(val[valid], [0.5, 99.5])) if valid.any() else (0.0, 1.0)
    lo, hi = value_range
    gray = np.clip((np.where(valid, val, lo) - lo) * (255.0 / max(hi - lo, 1e-6)) + 0.5, 0, 255).astype(np.uint8)
    rgba = np.zeros(val.shape + (4,), np.uint8)
    rgba[..., :3] = palette_lut(palette)[gray]
    rgba[..., 3] = valid * 255
    rgba[~valid, :3] = 0
    H, W = val.shape
    w = GeoTIFFWriter(out_tif, W, H, tile=512, **geo)
    for ty in range(0, H, 512):
        for tx in range(0, W, 512):
            blk = rgba[ty:ty + 512, tx:tx + 512]
            if blk[..., 3].any():
                w.write_tile(tx // 512, ty // 512, w.compress_tile(blk))
    w.close()
    if legend_png:
        legend(legend_png, palette, value_range)
    return out_tif


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m orthomosaic.thermal", description="Thermal palettes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the available palettes")
    r = sub.add_parser("recolor", help="re-colour a *_thermal.tif raw-value raster with another palette")
    r.add_argument("thermal_tif")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--palette", default=DEFAULT_PALETTE)
    r.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"), help="value range (default: auto)")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        print("\n".join(palette_names()))
        return
    recolor(a.thermal_tif, a.output, a.palette, tuple(a.range) if a.range else None,
            legend_png=a.output.rsplit(".", 1)[0] + "_legend.png")
    print(a.output)


if __name__ == "__main__":
    main()
