"""Minimal streaming writer for tiled, Deflate-compressed (Geo)TIFF / BigTIFF.

Tiles can be written in any order as they are produced; the IFD is appended
at the end and the header patched, so the full raster never sits in memory.
"""
from __future__ import annotations

import struct
import zlib
from typing import Optional

import numpy as np

_SHORT, _LONG, _DOUBLE, _LONG8 = 3, 4, 12, 16


class GeoTIFFWriter:
    def __init__(self, path: str, width: int, height: int, tile: int = 512,
                 epsg: Optional[int] = None, origin: Optional[tuple[float, float]] = None,
                 pixel_size: Optional[float] = None, level: int = 6, bigtiff: Optional[bool] = None):
        if tile % 16:
            raise ValueError("tile size must be a multiple of 16")
        self.path, self.width, self.height, self.tile = path, width, height, tile
        self.epsg, self.origin, self.pixel_size, self.level = epsg, origin, pixel_size, level
        self.tiles_across = (width + tile - 1) // tile
        self.tiles_down = (height + tile - 1) // tile
        n = self.tiles_across * self.tiles_down
        if bigtiff is None:
            bigtiff = width * height * 4 > 3_500_000_000
        self.big = bigtiff
        self.offsets = np.zeros(n, np.uint64)
        self.counts = np.zeros(n, np.uint64)
        self.f = open(path, "wb")
        if self.big:
            self.f.write(b"II" + struct.pack("<HHHQ", 43, 8, 0, 0))
        else:
            self.f.write(b"II" + struct.pack("<HI", 42, 0))

    # -- tile data -----------------------------------------------------------
    def compress_tile(self, rgba: np.ndarray) -> bytes:
        """Pad a (<=tile, <=tile, 4) block to a full tile and Deflate it.
        Thread-safe / GIL-releasing, so call it from worker threads."""
        t = self.tile
        if rgba.shape[0] != t or rgba.shape[1] != t:
            full = np.zeros((t, t, 4), np.uint8)
            full[:rgba.shape[0], :rgba.shape[1]] = rgba
            rgba = full
        return zlib.compress(np.ascontiguousarray(rgba).tobytes(), self.level)

    def write_tile(self, tx: int, ty: int, data: bytes):
        i = ty * self.tiles_across + tx
        self.offsets[i] = self.f.tell()
        self.counts[i] = len(data)
        self.f.write(data)
        if self.f.tell() & 1:
            self.f.write(b"\0")

    # -- IFD -----------------------------------------------------------------
    def close(self):
        # Tiles never written (entirely outside coverage) share one empty tile.
        missing = self.counts == 0
        if missing.any():
            empty = self.compress_tile(np.zeros((self.tile, self.tile, 4), np.uint8))
            off = self.f.tell()
            self.f.write(empty)
            self.offsets[missing] = off
            self.counts[missing] = len(empty)
        if self.f.tell() & 1:
            self.f.write(b"\0")

        entries = [
            (256, _LONG, [self.width]),
            (257, _LONG, [self.height]),
            (258, _SHORT, [8, 8, 8, 8]),
            (259, _SHORT, [8]),               # Adobe Deflate
            (262, _SHORT, [2]),               # RGB
            (277, _SHORT, [4]),
            (284, _SHORT, [1]),               # chunky
            (317, _SHORT, [1]),               # no predictor
            (322, _LONG, [self.tile]),
            (323, _LONG, [self.tile]),
            (324, _LONG8 if self.big else _LONG, self.offsets.tolist()),
            (325, _LONG8 if self.big else _LONG, self.counts.tolist()),
            (338, _SHORT, [2]),               # unassociated alpha
            (339, _SHORT, [1, 1, 1, 1]),
        ]
        if self.pixel_size is not None and self.origin is not None:
            entries.append((33550, _DOUBLE, [self.pixel_size, self.pixel_size, 0.0]))
            entries.append((33922, _DOUBLE, [0.0, 0.0, 0.0, self.origin[0], self.origin[1], 0.0]))
            keys = [1, 1, 0, 0]
            geo = [(1024, 0, 1, 1),            # GTModelType = projected
                   (1025, 0, 1, 1)]            # RasterType = PixelIsArea
            if self.epsg:
                geo.append((3072, 0, 1, int(self.epsg)))
            keys[3] = len(geo)
            for g in geo:
                keys.extend(g)
            entries.append((34735, _SHORT, keys))
        entries.sort(key=lambda e: e[0])
        self._write_ifd(entries)
        self.f.close()

    def _write_ifd(self, entries):
        big = self.big
        fmt_count = "<Q" if big else "<H"
        entry_size = 20 if big else 12
        inline = 8 if big else 4
        sizes = {_SHORT: 2, _LONG: 4, _DOUBLE: 8, _LONG8: 8}
        codes = {_SHORT: "H", _LONG: "I", _DOUBLE: "d", _LONG8: "Q"}

        ifd_pos = self.f.tell()
        head = 8 if big else 2
        tail = 8 if big else 4
        data_pos = ifd_pos + head + entry_size * len(entries) + tail
        blob = bytearray()
        body = bytearray(struct.pack(fmt_count, len(entries)))
        for tag, typ, vals in entries:
            payload = struct.pack("<%d%s" % (len(vals), codes[typ]), *vals)
            hdr = struct.pack("<HHQ" if big else "<HHI", tag, typ, len(vals))
            if len(payload) <= inline:
                body += hdr + payload.ljust(inline, b"\0")
            else:
                off = data_pos + len(blob)
                body += hdr + struct.pack("<Q" if big else "<I", off)
                blob += payload
                if len(blob) & 1:
                    blob += b"\0"
        body += b"\0" * tail
        self.f.write(body)
        self.f.write(blob)
        if not big and ifd_pos > 0xFFFFFFFF:
            raise OverflowError("Output exceeds 4 GB; rerun with bigtiff=True")
        self.f.seek(8 if big else 4)
        self.f.write(struct.pack("<Q" if big else "<I", ifd_pos))
