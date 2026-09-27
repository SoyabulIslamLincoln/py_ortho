"""Image discovery, EXIF/GPS parsing and (reduced-resolution) decoding."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_TAG_GPS_IFD = 0x8825
_TAG_EXIF_IFD = 0x8769
_TAG_ORIENTATION = 0x0112
_TAG_FOCAL = 0x920A
_TAG_FOCAL35 = 0xA405


@dataclass
class Frame:
    path: str
    width: int
    height: int
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    focal_mm: Optional[float] = None
    focal35_mm: Optional[float] = None
    orientation: int = 1
    # filled by the pipeline
    E: Optional[float] = None
    N: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def _rat(v) -> float:
    try:
        return float(v)
    except TypeError:
        return float(v[0]) / float(v[1])


def _dms(vals, ref) -> Optional[float]:
    if vals is None:
        return None
    d, m, s = (_rat(x) for x in vals)
    out = d + m / 60.0 + s / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors="ignore")
    if ref and str(ref).strip().upper() in ("S", "W"):
        out = -out
    return out


def list_images(folder: str) -> list[str]:
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS and not f.startswith(".")]
    return files


def read_frame(path: str) -> Frame:
    with Image.open(path) as im:
        w, h = im.size
        exif = im.getexif()
        orientation = int(exif.get(_TAG_ORIENTATION, 1) or 1)
        if orientation in (5, 6, 7, 8):
            w, h = h, w
        fr = Frame(path=path, width=w, height=h, orientation=orientation)
        try:
            gps = exif.get_ifd(_TAG_GPS_IFD)
        except Exception:
            gps = {}
        if gps:
            fr.lat = _dms(gps.get(2), gps.get(1))
            fr.lon = _dms(gps.get(4), gps.get(3))
            if gps.get(6) is not None:
                alt = _rat(gps.get(6))
                ref = gps.get(5, 0)
                if isinstance(ref, bytes):
                    ref = ref[0] if ref else 0
                fr.alt = -alt if ref == 1 else alt
        try:
            ex = exif.get_ifd(_TAG_EXIF_IFD)
            if ex.get(_TAG_FOCAL) is not None:
                fr.focal_mm = _rat(ex[_TAG_FOCAL])
            if ex.get(_TAG_FOCAL35) is not None:
                fr.focal35_mm = float(ex[_TAG_FOCAL35])
        except Exception:
            pass
    return fr


def load_rgb(frame: Frame, scale: float = 1.0) -> np.ndarray:
    """Decode an image as uint8 RGB at approximately `scale` of full resolution.

    Uses JPEG DCT-domain downscaling (Image.draft) when possible, which makes
    reduced-resolution decoding several times faster and lighter on memory.
    Output size is exactly round(width*scale) x round(height*scale).
    """
    tw = max(1, int(round(frame.width * scale)))
    th = max(1, int(round(frame.height * scale)))
    with Image.open(frame.path) as im:
        if scale < 1.0 and im.format == "JPEG":
            dw, dh = (tw, th) if frame.orientation not in (5, 6, 7, 8) else (th, tw)
            im.draft("RGB", (dw, dh))
        if frame.orientation != 1:
            im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        if im.size != (tw, th):
            im = im.resize((tw, th), Image.BILINEAR if scale >= 0.5 else Image.BOX)
        return np.asarray(im, dtype=np.uint8).copy()
