"""Thermal: palettes, radiometric R-JPEG decoding, recolouring of the raw-value raster."""
import io
import os
import struct
import sys
import tempfile

import numpy as np
from PIL import Image

try:
    import orthomosaic._core  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import thermal  # noqa: E402
from orthomosaic.geotiff import GeoTIFFWriter  # noqa: E402
from orthomosaic.imageio import load_rgb, read_frame  # noqa: E402


def _rjpeg(path, raw, size):
    """A JPEG with the raw uint16 image split over APP3 segments, like DJI R-JPEGs."""
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 0, 120)).save(buf, "JPEG")
    jpg = buf.getvalue()
    data = raw.astype("<u2").tobytes()
    app3 = b"".join(b"\xff\xe3" + struct.pack(">H", len(c) + 2) + c
                    for c in (data[i:i + 65000] for i in range(0, len(data), 65000)))
    with open(path, "wb") as fh:
        fh.write(jpg[:2] + app3 + jpg[2:])


def test_palettes():
    assert thermal.DEFAULT_PALETTE == "rainbow"
    for name in thermal.palette_names():
        lut = thermal.palette_lut(name)
        assert lut.shape == (256, 3) and lut.dtype == np.uint8
    assert (thermal.palette_lut("white_hot")[[0, 255]] == [[0, 0, 0], [255, 255, 255]]).all()
    assert (thermal.palette_lut("ironbow") == thermal.palette_lut("iron")).all()        # alias
    custom = thermal.palette_lut([(0, 0, 255), (255, 0, 0)])
    assert tuple(custom[0]) == (0, 0, 255) and tuple(custom[255]) == (255, 0, 0)
    try:
        thermal.palette_lut("nope")
        raise AssertionError("unknown palette accepted")
    except ValueError:
        pass


def test_rjpeg_decode_and_load():
    raw = (17000 + np.arange(64 * 80).reshape(64, 80) % 2000).astype(np.uint16)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t_T.JPG")
        _rjpeg(p, raw, (160, 128))                     # raw is half the JPEG size, like the M4T
        fr = read_frame(p)
        assert fr.raw_shape == (64, 80)
        assert (thermal.read_raw_thermal(p, fr.raw_shape) == raw).all()
        fr.thermal_range = (17000.0, 19000.0)
        img = load_rgb(fr, 1.0)
        assert img.shape == (128, 160, 3) and (img[..., 0] == img[..., 2]).all()
        exp = np.clip((raw[10, 10] - 17000) / 2000 * 255, 0, 255)
        assert abs(int(img[21, 21, 0]) - exp) <= 3


def test_recolor_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "t.tif"), os.path.join(d, "c.tif")
        v = np.linspace(100, 200, 64 * 64, dtype=np.float32).reshape(64, 64)
        v[:5] = np.nan
        w = GeoTIFFWriter(src, 64, 64, tile=64, kind="float32", nodata=float("nan"), epsg=32646,
                          origin=(1000.0, 2000.0), pixel_size=0.1)
        w.write_tile(0, 0, w.compress_tile(v))
        w.close()
        thermal.recolor(src, dst, "iron", value_range=(100.0, 200.0))
        out = Image.open(dst)
        a = np.asarray(out)
        assert a.shape == (64, 64, 4) and (a[:5, :, 3] == 0).all() and (a[5:, :, 3] == 255).all()
        assert tuple(a[-1, -1, :3]) == tuple(thermal.palette_lut("iron")[255])
        assert out.tag_v2[33550][0] == 0.1 and 32646 in out.tag_v2[34735]


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
