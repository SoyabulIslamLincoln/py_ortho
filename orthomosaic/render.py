"""Block-wise orthomosaic rendering into a streaming GeoTIFF."""
from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

from .geotiff import GeoTIFFWriter
from .imageio import load_rgb

log = logging.getLogger(__name__)


class ImageCache:
    """Thread-safe LRU cache with a byte budget; concurrent requests for the
    same key wait for a single load instead of decoding twice."""

    def __init__(self, loader: Callable, sizeof: Callable, budget_bytes: int):
        self.loader, self.sizeof, self.budget = loader, sizeof, budget_bytes
        self._data: OrderedDict = OrderedDict()
        self._loading: dict = {}
        self._size = 0
        self._lock = threading.Lock()
        self.loads = 0

    def get(self, key):
        while True:
            with self._lock:
                if key in self._data:
                    self._data.move_to_end(key)
                    return self._data[key]
                ev = self._loading.get(key)
                owner = ev is None
                if owner:
                    ev = self._loading[key] = threading.Event()
            if not owner:
                ev.wait()
                continue
            try:
                value = self.loader(key)
            except BaseException:
                with self._lock:
                    self._loading.pop(key).set()
                raise
            with self._lock:
                self._data[key] = value
                self._size += self.sizeof(value)
                self.loads += 1
                while self._size > self.budget and len(self._data) > 1:
                    _, old = self._data.popitem(last=False)
                    self._size -= self.sizeof(old)
                self._loading.pop(key).set()
            return value


def render(frames, affines: dict, gains: dict, backend, out_path: str, gsd: float,
           epsg: Optional[int], origin_offset: tuple[float, float], render_scale: float = 1.0,
           block: int = 1024, tile: int = 512, mode: int = 0, power: float = 2.0,
           workers: int = 4, cache_mb: int = 1024, preview_path: Optional[str] = None,
           max_pixels: float = 4e10, post=None, thermal=None):
    """post: optional callable applied to every rendered RGBA block.
    thermal: dict(lut, range, raw_path) -> the blended gray is raw thermal; colour it with the
    palette and also write the raw values (float32, nodata NaN) to raw_path."""
    ids = sorted(affines)
    # world footprints
    boxes = {}
    for i in ids:
        f = frames[i]
        corners = np.array([[-0.5, -0.5], [f.width - 0.5, -0.5], [f.width - 0.5, f.height - 0.5],
                            [-0.5, f.height - 0.5]])
        w = corners @ affines[i][:, :2].T + affines[i][:, 2]
        boxes[i] = (w[:, 0].min(), w[:, 0].max(), w[:, 1].min(), w[:, 1].max())
    minX = math.floor(min(b[0] for b in boxes.values()) / gsd) * gsd
    maxX = math.ceil(max(b[1] for b in boxes.values()) / gsd) * gsd
    minY = math.floor(min(b[2] for b in boxes.values()) / gsd) * gsd
    maxY = math.ceil(max(b[3] for b in boxes.values()) / gsd) * gsd
    W = int(round((maxX - minX) / gsd))
    H = int(round((maxY - minY) / gsd))
    if W * H > max_pixels:
        raise ValueError(f"Output would be {W}x{H} pixels; increase the output resolution (GSD) value")
    log.info("Output raster %d x %d px at %.4f units/px", W, H, gsd)

    block = max(tile, (block // tile) * tile)
    geo = dict(epsg=epsg, origin=(minX + origin_offset[0], maxY + origin_offset[1]), pixel_size=gsd) \
        if epsg is not None else dict(epsg=None, origin=(minX, maxY), pixel_size=gsd)
    writer = GeoTIFFWriter(out_path, W, H, tile=tile, **geo)
    raw_writer = None
    if thermal is not None:
        from .thermal import colorize_rgba
        raw_writer = GeoTIFFWriter(thermal["raw_path"], W, H, tile=tile, kind="float32", nodata=float("nan"), **geo)
        t_lo, t_hi = thermal["range"]
        t_lut = thermal["lut"]

    # world -> loaded-image pixel transforms
    inv = {}
    for i in ids:
        P = np.vstack([affines[i], [0, 0, 1]])
        f = frames[i]
        lw = max(1, int(round(f.width * render_scale)))
        lh = max(1, int(round(f.height * render_scale)))
        sx, sy = lw / f.width, lh / f.height
        S = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1]])
        inv[i] = S @ np.linalg.inv(P)

    cache = ImageCache(lambda i: backend.upload(load_rgb(frames[i], render_scale)),
                       backend.image_nbytes, cache_mb * 1024 * 1024)

    pf = max(1, math.ceil(max(W, H) / 2048)) if preview_path else 0
    preview = np.zeros((math.ceil(H / pf), math.ceil(W / pf), 4), np.uint8) if pf else None

    bx_n, by_n = math.ceil(W / block), math.ceil(H / block)
    order = []
    for by in range(by_n):  # serpentine traversal keeps neighbouring images in cache
        row = range(bx_n) if by % 2 == 0 else range(bx_n - 1, -1, -1)
        order.extend((bx, by) for bx in row)

    def block_hits(bx, by):
        x0, y0 = bx * block, by * block
        bw, bh = min(block, W - x0), min(block, H - y0)
        wx0, wx1 = minX + x0 * gsd, minX + (x0 + bw) * gsd
        wy1, wy0 = maxY - y0 * gsd, maxY - (y0 + bh) * gsd
        return [i for i in ids if boxes[i][0] < wx1 and boxes[i][1] > wx0
                and boxes[i][2] < wy1 and boxes[i][3] > wy0]

    def render_block(bx, by):
        """Warp + blend every overlapping image into one block (runs on the compute backend)."""
        x0, y0 = bx * block, by * block
        bw, bh = min(block, W - x0), min(block, H - y0)
        hits = block_hits(bx, by)
        rgba = None
        if hits:
            B = np.array([[gsd, 0, minX + (x0 + 0.5) * gsd], [0, -gsd, maxY - (y0 + 0.5) * gsd], [0, 0, 1]])
            acc = backend.new_block(bh, bw)
            for i in hits:
                M = (inv[i] @ B)[:2]
                backend.warp_accumulate(acc, cache.get(i), M, gains.get(i, np.ones(3, np.float32)), power, mode)
            rgba = backend.finalize(acc, mode)
            if not rgba[..., 3].any():
                rgba = None
            else:
                if post is not None:
                    rgba = post(rgba)
        return x0, y0, bw, bh, rgba

    def compress_block(x0, y0, bw, bh, rgba):
        out = []
        if rgba is not None and raw_writer is not None:
            raw = np.where(rgba[..., 3] > 0, t_lo + rgba[..., 0].astype(np.float32) * ((t_hi - t_lo) / 255.0),
                           np.nan).astype(np.float32)
            for ty in range(0, bh, tile):
                for tx in range(0, bw, tile):
                    blk = raw[ty:ty + tile, tx:tx + tile]
                    if np.isfinite(blk).any():
                        out.append(("raw", (x0 + tx) // tile, (y0 + ty) // tile, raw_writer.compress_tile(blk)))
            rgba = colorize_rgba(rgba, t_lut)
        if rgba is not None:
            for ty in range(0, bh, tile):
                for tx in range(0, bw, tile):
                    t = rgba[ty:ty + tile, tx:tx + tile]
                    if t[..., 3].any():
                        out.append(((x0 + tx) // tile, (y0 + ty) // tile, writer.compress_tile(t)))
        return x0, y0, rgba, out

    def do_block(bx, by):
        return compress_block(*render_block(bx, by))

    def consume(result):
        x0, y0, rgba, tiles = result
        for t in tiles:
            if t[0] == "raw":
                raw_writer.write_tile(t[1], t[2], t[3])
            else:
                writer.write_tile(*t)
        if preview is not None and rgba is not None:
            ox, oy = (-x0) % pf, (-y0) % pf
            sub = rgba[oy::pf, ox::pf]
            py, px = (y0 + oy) // pf, (x0 + ox) // pf
            preview[py:py + sub.shape[0], px:px + sub.shape[1]] = sub

    done = 0
    total = len(order)
    step = max(1, total // 20)
    if backend.parallel_blocks and workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            pending = []
            it = iter(order)
            for bx, by in it:
                pending.append(ex.submit(do_block, bx, by))
                if len(pending) >= 2 * workers:
                    consume(pending.pop(0).result())
                    done += 1
                    if done % step == 0:
                        log.info("  rendered %d/%d blocks", done, total)
            for fut in pending:
                consume(fut.result())
    else:
        # GPU backends render blocks one at a time; overlap the CPU work around them:
        # decode (and upload) the images of the next blocks and Deflate finished tiles in threads
        from collections import deque
        prefetched = set()
        with ThreadPoolExecutor(max(2, workers)) as ex:
            pending = deque()
            for k, (bx, by) in enumerate(order):
                for nb in order[k + 1:k + 3]:
                    for i in block_hits(*nb):
                        if i not in prefetched:
                            prefetched.add(i)
                            ex.submit(cache.get, i)
                pending.append(ex.submit(compress_block, *render_block(bx, by)))
                while len(pending) > 2 * workers:
                    consume(pending.popleft().result())
                done += 1
                if done % step == 0:
                    log.info("  rendered %d/%d blocks", done, total)
            while pending:
                consume(pending.popleft().result())
    writer.close()
    if raw_writer is not None:
        raw_writer.close()
    log.info("Image decodes during render: %d (for %d images)", cache.loads, len(ids))

    if preview is not None:
        from PIL import Image
        im = Image.fromarray(preview, "RGBA")
        if preview_path.lower().endswith((".jpg", ".jpeg")):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            im = bg
        im.save(preview_path)
    return dict(width=W, height=H, bounds=(minX + origin_offset[0], minY + origin_offset[1],
                                           maxX + origin_offset[0], maxY + origin_offset[1]))
