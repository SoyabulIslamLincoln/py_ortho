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


CONTRASTS = ("linear", "equalize", "clahe")


def _equalize_lut(hist: np.ndarray) -> np.ndarray:
    """256-entry index remap that flattens a histogram (identity when it is empty)."""
    cdf = np.cumsum(hist, dtype=np.float64)
    if cdf[-1] <= 0:
        return np.arange(256, dtype=np.float32)
    lo = cdf[np.flatnonzero(hist)[0]]
    return np.clip((cdf - lo) * (255.0 / max(cdf[-1] - lo, 1.0)), 0, 255).astype(np.float32)


def apply_contrast(gray: np.ndarray, valid: Optional[np.ndarray] = None, contrast: str = "linear",
                   clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Re-map a uint8 index image before the palette: linear (unchanged), equalize (global
    histogram equalisation) or clahe (contrast-limited adaptive equalisation, `tiles` x `tiles`
    grid, OpenCV-style clip limit). Statistics use only `valid` pixels; display only - the
    raw-value raster is never touched."""
    gray = np.asarray(gray, np.uint8)
    contrast = contrast.lower()
    if contrast not in CONTRASTS:
        raise ValueError(f"Unknown contrast '{contrast}'. Available: {', '.join(CONTRASTS)}")
    if valid is None:
        valid = np.ones(gray.shape, bool)
    if contrast == "linear" or not valid.any():
        return gray
    if contrast == "equalize":
        lut = _equalize_lut(np.bincount(gray[valid], minlength=256))
        out = gray.copy()
        out[valid] = (lut[gray[valid]] + 0.5).astype(np.uint8)
        return out

    H, W = gray.shape
    ny, nx = min(tiles, H), min(tiles, W)
    ys = np.linspace(0, H, ny + 1).round().astype(int)
    xs = np.linspace(0, W, nx + 1).round().astype(int)
    fallback = _equalize_lut(np.bincount(gray[valid], minlength=256))
    luts = np.empty((ny, nx, 256), np.float32)
    for i in range(ny):
        for j in range(nx):
            g = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]][valid[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]]
            if g.size == 0:
                luts[i, j] = fallback
                continue
            hist = np.bincount(g, minlength=256).astype(np.float64)
            limit = max(clip_limit * g.size / 256.0, 1.0)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0       # clip and redistribute evenly
            cdf = np.cumsum(hist)
            luts[i, j] = np.clip(cdf * (255.0 / cdf[-1]), 0, 255)
    # bilinear blend of the four nearest tile mappings (tile centres), in row strips to bound memory
    cy = (ys[:-1] + ys[1:]) / 2.0
    cx = (xs[:-1] + xs[1:]) / 2.0
    fx = np.interp(np.arange(W) + 0.5, cx, np.arange(nx))
    x0 = np.floor(fx).astype(int)
    x1 = np.minimum(x0 + 1, nx - 1)
    wx = (fx - x0).astype(np.float32)
    out = gray.copy()
    for r0 in range(0, H, 512):
        r1 = min(r0 + 512, H)
        fy = np.interp(np.arange(r0, r1) + 0.5, cy, np.arange(ny))
        y0 = np.floor(fy).astype(int)
        y1 = np.minimum(y0 + 1, ny - 1)
        wy = (fy - y0).astype(np.float32)[:, None]
        g = gray[r0:r1]
        top = luts[y0[:, None], x0, g] * (1 - wx) + luts[y0[:, None], x1, g] * wx
        bot = luts[y1[:, None], x0, g] * (1 - wx) + luts[y1[:, None], x1, g] * wx
        blk = np.clip(top * (1 - wy) + bot * wy + 0.5, 0, 255).astype(np.uint8)
        v = valid[r0:r1]
        out[r0:r1][v] = blk[v]
    return out


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


def raw_to_gray(raw: np.ndarray, value_range: tuple[float, float], size: tuple[int, int],
                flat: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalise raw sensor values to uint8 with the survey-wide range and resize to (w, h).
    `flat`: optional (h, w) additive correction in gray levels (see flat_field)."""
    from PIL import Image
    lo, hi = value_range
    g = (raw.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6))
    im = Image.fromarray(np.clip(g, 0, 255).astype(np.float32), "F")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR if size[0] >= im.size[0] else Image.BOX)
    g = np.asarray(im)
    if flat is not None:
        g = g + flat
    return np.clip(g + 0.5, 0, 255).astype(np.uint8)


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


def _flat_terms(px, py, width: int, height: int) -> np.ndarray:
    """Flat-field basis at full-resolution pixel positions: radial falloff plus a planar and a
    saddle tilt across the sensor (coordinates scaled by the half width)."""
    s = 0.5 * width
    x, y = np.broadcast_arrays((np.asarray(px, np.float64) - 0.5 * width) / s,
                               (np.asarray(py, np.float64) - 0.5 * height) / s)
    r2 = x * x + y * y
    return np.stack([r2, r2 * r2, x, y, x * x - y * y, x * y], -1)


def flat_field(coeffs, width: int, height: int, size: tuple[int, int]) -> np.ndarray:
    """(h, w) float32 additive correction (gray levels, zero mean) for a frame resampled to `size`."""
    w, h = size
    xs = (np.arange(w) + 0.5) * (width / w)
    ys = (np.arange(h) + 0.5) * (height / h)
    v = _flat_terms(xs[None, :], ys[:, None], width, height) @ np.asarray(coeffs, np.float64)
    return (v - v.mean()).astype(np.float32)


def solve_offsets(used, pairs, size: Optional[tuple[int, int]] = None, prior: float = 8.0,
                  max_offset: float = 32.0, max_flat: float = 32.0, iters: int = 10, per_pair: int = 200):
    """Remove frame-to-frame drift and the sensor's flat-field error from a radiometric thermal block.

    Uncooled thermal cores drift between flat-field corrections (a per-frame offset) and read
    the image borders differently from the centre (a fixed pattern across the sensor). In a
    mosaic both show as blocks: where two source photos meet, the same surface reads a few
    levels apart. Gains are never touched (they would rescale temperatures). Every tie point
    gives ``c_i + o_i + V(p_i) = c_j + o_j + V(p_j)``; the per-frame offsets ``o`` and, when
    `size` = (width, height) is given, one flat-field surface ``V`` shared by all frames are fitted
    by Huber-reweighted least squares. The offsets have zero mean (the survey's absolute level
    is kept), a weak prior anchors poorly connected frames, and both are bounded.

    Returns ``(offsets, flat)``: ``{frame_index: float32 (3,)}`` usable as ``biases`` and the
    flat-field coefficients for :func:`flat_field` (None when not solved or implausible)."""
    local = {g: k for k, g in enumerate(used)}
    n = len(used)
    I, J, M, Fi, Fj = [], [], [], [], []
    rng = np.random.default_rng(0)
    for p in pairs:
        if p.i in local and p.j in local and len(p.ci) and len(p.ci) == len(p.cj):
            sel = np.arange(len(p.ci))
            if len(sel) > per_pair:
                sel = np.sort(rng.choice(sel, per_pair, replace=False))
            I.append(np.full(len(sel), local[p.i]))
            J.append(np.full(len(sel), local[p.j]))
            M.append(p.cj[sel, 0].astype(np.float64) - p.ci[sel, 0].astype(np.float64))
            if size is not None:
                Fi.append(p.pi[sel])
                Fj.append(p.pj[sel])
    if n == 0 or not I:
        return {}, None
    ii, jj, m = np.concatenate(I), np.concatenate(J), np.concatenate(M)
    F = (_flat_terms(*np.concatenate(Fi).T, *size) - _flat_terms(*np.concatenate(Fj).T, *size)
         if size is not None else np.zeros((len(m), 0)))
    nv = F.shape[1]
    N = n + nv
    w = np.ones(len(m))
    x = np.zeros(N)
    for _ in range(iters):
        # residual per tie point: o_i - o_j + (V(p_i) - V(p_j)) - (c_j - c_i)
        A = np.zeros((N, N))
        rhs = np.zeros(N)
        np.add.at(A, (ii, ii), w)
        np.add.at(A, (jj, jj), w)
        np.add.at(A, (ii, jj), -w)
        np.add.at(A, (jj, ii), -w)
        np.add.at(rhs, ii, w * m)
        np.add.at(rhs, jj, -w * m)
        if nv:
            Fw = F * w[:, None]
            C = np.zeros((n, nv))
            np.add.at(C, ii, Fw)
            np.add.at(C, jj, -Fw)
            A[:n, n:] += C
            A[n:, :n] += C.T
            A[n:, n:] += F.T @ Fw + 1e-3 * np.eye(nv)
            rhs[n:] += Fw.T @ m
        A[:n, :n] += np.eye(n) / prior ** 2 + 1.0  # weak prior + sum(o) = 0
        x = np.linalg.solve(A, rhs)
        r = np.abs(x[ii] - x[jj] + F @ x[n:] - m)
        k = 1.5 * max(np.median(r), 0.5)           # Huber: down-weight parallax/occlusion/mismatches
        w = np.minimum(1.0, k / np.maximum(r, 1e-9))
    o = np.clip(x[:n] - x[:n].mean(), -max_offset, max_offset)
    flat = x[n:] if nv else None
    if flat is not None:
        span = np.ptp(flat_field(flat, *size, (64, max(1, round(64 * size[1] / size[0])))))
        if not np.isfinite(span) or span > max_flat:
            log.warning("Thermal flat-field fit implausible (%.1f gray levels across the frame): not applied", span)
            flat = None
        else:
            log.info("Thermal flat field: %.1f gray levels across the sensor", span)
    log.info("Thermal drift: per-image offsets %.1f..%.1f gray levels (p5..p95), tie-point mismatch p90 %.1f -> %.1f",
             *np.percentile(o, [5, 95]), np.percentile(np.abs(m), 90), np.percentile(r, 90))
    return {g: np.full(3, o[k], np.float32) for g, k in local.items()}, flat


# --------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------

def legend(path: str, palette: PaletteSpec, value_range: tuple[float, float], label: str = "raw sensor value",
           contrast: str = "linear"):
    """Small colour-bar PNG with the value range (a non-linear contrast only marks the ends)."""
    from PIL import Image, ImageDraw
    lut = palette_lut(palette)
    W, H, bar = 360, 70, 22
    img = Image.new("RGB", (W, H), "white")
    strip = np.repeat(lut[np.linspace(0, 255, W - 40).astype(int)][None], bar, 0)
    img.paste(Image.fromarray(strip), (20, 10))
    d = ImageDraw.Draw(img)
    lo, hi = value_range
    for frac in ((0.0, 0.5, 1.0) if contrast == "linear" else (0.0, 1.0)):
        x = 20 + frac * (W - 41)
        d.line([(x, 10 + bar), (x, 16 + bar)], fill="black")
        txt = f"{lo + frac * (hi - lo):.0f}"
        d.text((x - 4 * len(txt) / 2 * 1.5, 18 + bar), txt, fill="black")
    name = palette if isinstance(palette, str) else "custom"
    if contrast != "linear":
        name += f", {contrast}: non-linear scale"
    d.text((20, H - 14), f"{label} ({name})", fill="black")
    img.save(path)


def recolor(thermal_tif: str, out_tif: str, palette: PaletteSpec = DEFAULT_PALETTE,
            value_range: Optional[tuple[float, float]] = None, legend_png: Optional[str] = None,
            contrast: str = "linear") -> str:
    """Re-render a raw-value thermal GeoTIFF (from build_orthomosaic / build_3d) with another palette."""
    from .geotiff import GeoTIFFWriter, read_geotiff
    val, geo = read_geotiff(thermal_tif)
    valid = np.isfinite(val)
    if value_range is None:
        value_range = tuple(float(x) for x in np.percentile(val[valid], [0.5, 99.5])) if valid.any() else (0.0, 1.0)
    lo, hi = value_range
    gray = np.clip((np.where(valid, val, lo) - lo) * (255.0 / max(hi - lo, 1e-6)) + 0.5, 0, 255).astype(np.uint8)
    rgba = np.zeros(val.shape + (4,), np.uint8)
    rgba[..., :3] = palette_lut(palette)[apply_contrast(gray, valid, contrast)]
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
        legend(legend_png, palette, value_range, contrast=contrast)
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
    r.add_argument("--contrast", choices=CONTRASTS, default="linear",
                   help="linear (auto range), equalize (histogram) or clahe (local contrast)")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        print("\n".join(palette_names()))
        return
    recolor(a.thermal_tif, a.output, a.palette, tuple(a.range) if a.range else None,
            legend_png=a.output.rsplit(".", 1)[0] + "_legend.png", contrast=a.contrast)
    print(a.output)


if __name__ == "__main__":
    main()
