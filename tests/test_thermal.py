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
from orthomosaic.imageio import load_rgb, load_thermal, read_frame  # noqa: E402


def _iirp_app4(ambient=20.0, distance=5.0, emissivity=0.95, humidity=0.5, reflected=20.0, gain=1):
    """A DJI iirp APP4 measurement header (magic, tag, 5 floats at 0x20, gain at 0x80)."""
    head = bytearray(0x100)
    head[:8] = b"\xff\xd2\xd1\xffiirp"
    struct.pack_into("<5f", head, 0x20, ambient, distance, emissivity, humidity, reflected)
    head[0x80] = gain
    return bytes(head)


def _rjpeg(path, raw, size, app4=None):
    """A JPEG with the raw uint16 image split over APP3 segments, like DJI R-JPEGs."""
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 0, 120)).save(buf, "JPEG")
    jpg = buf.getvalue()
    data = raw.astype("<u2").tobytes()
    app3 = b"".join(b"\xff\xe3" + struct.pack(">H", len(c) + 2) + c
                    for c in (data[i:i + 65000] for i in range(0, len(data), 65000)))
    if app4 is not None:
        app3 += b"\xff\xe4" + struct.pack(">H", len(app4) + 2) + app4
    with open(path, "wb") as fh:
        fh.write(jpg[:2] + app3 + jpg[2:])


def _celsius_to_raw(t):
    """Inverse of the iirp count encoding at emissivity 1 / no atmosphere: T0 = (raw >> 2) / 16 K."""
    return (np.rint((np.asarray(t, np.float64) + 273.15) * 16).astype(np.int64) << 2).astype(np.uint16)


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
        # the mosaic loader keeps one unquantised float channel, unclipped beyond the range
        g = load_thermal(fr, 1.0)
        assert g.shape == (128, 160) and g.dtype == np.float32
        centre = 0.25 * (g[0::2, 0::2] + g[1::2, 0::2] + g[0::2, 1::2] + g[1::2, 1::2])   # = raw centre (bilinear x2)
        err = np.abs(thermal.gray_to_values(centre, fr.thermal_range) - raw)
        assert np.median(err) < 0.5 < 2000 / 255                  # 8-bit gray would be ~8 counts per step
        assert len(np.unique(g)) > 256


def test_celsius_conversion():
    """iirp counts -> °C: the stored measurement parameters are read, emissivity 1 / no atmosphere
    gives the blackbody temperature, and the correction inverts the radiometric forward model."""
    t = np.array([[-10.0, 0.0, 21.3], [35.7, 60.0, 120.0]])
    raw = _celsius_to_raw(t)
    p = dict(emissivity=1.0, distance=0.0, humidity=50.0, reflected=20.0, ambient=20.0)
    assert np.abs(thermal.raw_to_celsius(raw, p) - t).max() <= 0.032          # 1/16 K counts
    assert abs(thermal.atmospheric_transmission(0.0, 50, 20) - 1.0) < 1e-9
    taus = [thermal.atmospheric_transmission(d, 50, 20) for d in (5, 30, 100)]
    assert 1.0 > taus[0] > taus[1] > taus[2] > 0.8
    # forward model: what the camera sees of a 45 °C, e=0.9 roof at 30 m with a 15 °C sky reflection
    e, tau, K = 0.9, thermal.atmospheric_transmission(30.0, 60.0, 18.0), 273.15
    w = e * tau * (45 + K) ** 4 + (1 - e) * tau * (15 + K) ** 4 + (1 - tau) * (18 + K) ** 4
    seen = _celsius_to_raw(np.array([w ** 0.25 - K]))
    p = dict(emissivity=e, distance=30.0, humidity=60.0, reflected=15.0, ambient=18.0)
    assert abs(float(thermal.raw_to_celsius(seen, p)[0]) - 45.0) < 0.1
    with tempfile.TemporaryDirectory() as d:
        a, b, c = (os.path.join(d, n) for n in ("a_T.JPG", "b_T.JPG", "c_T.JPG"))
        r = _celsius_to_raw(np.full((64, 80), 25.0))
        _rjpeg(a, r, (160, 128), _iirp_app4(ambient=17.5, distance=5.0, emissivity=0.95, humidity=0.5, reflected=17.5))
        _rjpeg(b, r, (160, 128), _iirp_app4(gain=3))                       # 'high-high gain': not measurable
        _rjpeg(c, r, (160, 128))                                           # no measurement header
        m = thermal.read_measurement_params(a)
        assert m == dict(emissivity=np.float32(0.95), distance=5.0, humidity=50.0, reflected=17.5, ambient=17.5)
        assert thermal.read_measurement_params(b) is None and thermal.read_measurement_params(c) is None
        # survey setup: per-image parameters + option overrides, range in °C
        from types import SimpleNamespace
        frames = [read_frame(a), read_frame(a)]
        opt = SimpleNamespace(thermal_units="celsius", emissivity=None, thermal_distance=12.0,
                              humidity=None, reflected_temp=None)
        rng, units = thermal.setup_survey(frames, opt)
        assert units == "°C" and frames[0].thermal_cal["distance"] == 12.0 and frames[0].thermal_cal["ambient"] == 17.5
        assert 25.0 < rng[0] <= rng[1] < 25.6                     # e < 1 against a cooler surround
        assert thermal.measurement_summary(frames)["distance"] == 12.0
        mixed = [read_frame(a), read_frame(c)]
        rng, units = thermal.setup_survey(mixed, opt)
        assert units == "raw" and mixed[0].thermal_cal is None and rng[0] > 10000
        # newer M4T/M3TD firmware tags APP4 "hirp" (14-bit grays + per-camera calibration in APP5):
        # recognised, not converted, the survey stays in raw counts
        h = os.path.join(d, "h_T.JPG")
        app4 = bytearray(0x100)
        app4[:8] = b"\xff\xd2\xd1\xffhirp"
        _rjpeg(h, np.full((64, 80), 5000, np.uint16), (160, 128), bytes(app4))
        assert [thermal.rjpeg_format(x) for x in (a, b, c, h)] == ["iirp", "iirp-highgain", "unknown", "hirp"]
        assert thermal.read_measurement_params(h) is None
        hf = [read_frame(h), read_frame(h)]
        assert hf[0].raw_shape == (64, 80)                        # still mosaicked radiometrically
        rng, units = thermal.setup_survey(hf, opt)
        assert units == "raw" and rng == (5000.0, 5000.0) and hf[0].thermal_cal is None


def test_carrier_blends_exactly():
    """The 2D renderer's 16-bit carrier survives the uint8 warp/blend kernels: the blend of two
    frames decodes to the blend of their float values, far finer than 8-bit gray."""
    from orthomosaic.backend import CPUBackend
    rng = np.random.default_rng(2)
    ga = rng.uniform(-40, 300, (40, 50)).astype(np.float32)
    gb = rng.uniform(-40, 300, (40, 50)).astype(np.float32)
    assert np.abs(thermal.carrier_to_gray(thermal.gray_to_carrier(ga).astype(np.float32)) - ga).max() < 0.005
    be = CPUBackend()
    blk = be.new_block(40, 50)
    M = np.array([1.0, 0, 0, 0, 1.0, 0])
    for g in (ga, gb):
        be.warp_accumulate(blk, be.upload(thermal.gray_to_carrier(g)), M, np.ones(3, np.float32), 2.0, 0)
    vals, cov = be.finalize_float(blk, 0)
    inner = (slice(1, -1), slice(1, -1))                  # border pixels carry zero feather weight
    assert cov[inner].all()
    assert np.abs(thermal.carrier_to_gray(vals)[inner] - 0.5 * (ga + gb)[inner]).max() < 0.01


def test_true_orthophoto_thermal_is_float_celsius():
    """End to end through the true orthophoto: two nadir thermal frames of a ground plane with a
    linear temperature field; the value output reproduces it in °C with sub-gray-level error."""
    from types import SimpleNamespace
    from orthomosaic.backend import CPUBackend
    from orthomosaic.camera import Intrinsics
    from orthomosaic.ortho import true_orthophoto
    temp = lambda x, y: 12.0 + 2.1 * x + 1.3 * y                 # °C, about -10..35 over the scene
    f, Z = 200.0, 30.0
    R = np.diag([1.0, -1.0, -1.0])                               # nadir: image x = east, y = south
    Cs = [np.array([-2.0, 0.0, Z]), np.array([2.5, 0.4, Z])]
    with tempfile.TemporaryDirectory() as d:
        frames = []
        for k, C in enumerate(Cs):
            u = (np.arange(80) * 2 + 0.5)                       # raw pixel centres in full-res pixels
            v = (np.arange(64) * 2 + 0.5)
            X = C[0] + (u[None, :] - 79.5) / f * Z
            Y = C[1] - (v[:, None] - 63.5) / f * Z
            p = os.path.join(d, f"f{k}_T.JPG")
            _rjpeg(p, _celsius_to_raw(temp(X, Y)), (160, 128), _iirp_app4(distance=0.0, emissivity=1.0))
            frames.append(read_frame(p))
        opt = SimpleNamespace(thermal_units="celsius", emissivity=None, thermal_distance=None,
                              humidity=None, reflected_temp=None)
        rng, units = thermal.setup_survey(frames, opt)
        assert units == "°C"
        gsd, minX, maxY = 0.1, -6.0, 5.0
        dsm = np.zeros((100, 120), np.float32)
        ar = SimpleNamespace(frames=frames, backend=CPUBackend(), thermal_range=rng, workers=1)
        rec = SimpleNamespace(used=[0, 1], R=np.stack([R, R]), C=np.stack(Cs),
                              intr=[Intrinsics(160, 128, f)], cam_group=np.zeros(2, np.int32))
        rgba, src, cnt, tmp = true_orthophoto(ar, rec, dsm, minX, maxY, gsd)
        val = thermal.gray_to_values(np.load(os.path.join(tmp, "value.npy")), rng)
        ok = rgba[..., 3] > 0
        assert ok.mean() > 0.95 and np.isfinite(val[ok]).all() and np.isnan(val[~ok]).all()
        xs = minX + (np.arange(120) + 0.5) * gsd
        ys = maxY - (np.arange(100) + 0.5) * gsd
        err = np.abs(val - temp(xs[None, :], ys[:, None]))[ok]
        step8 = (rng[1] - rng[0]) / 255                          # what 8-bit gray could resolve
        assert err.max() < 0.08 < 0.5 * step8, (err.max(), step8)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


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


def test_solve_offsets_removes_frame_drift_and_flat_field():
    """Per-frame drift and a shared sensor flat field are recovered from tie points (zero mean,
    robust to a bad pair), and the correction reaches the decoded thermal frames."""
    from types import SimpleNamespace
    rng = np.random.default_rng(4)
    n, size = 30, (640, 512)
    true_flat = np.array([3.0, -1.0, 2.5, 0.0, 0.5, 0.0])          # radial falloff + left-right tilt
    used = list(range(100, 100 + n))                   # frame indices need not start at 0
    drift = rng.normal(0, 4, n)
    drift -= drift.mean()
    pairs = []
    for a in range(n):
        for b in range(a + 1, a + 5):                  # sparser overlap than a real flight (~16)
            if b < n:
                k = 120
                scene = rng.uniform(40, 200, k)
                pi = rng.uniform(0, 1, (k, 2)) * size
                pj = rng.uniform(0, 1, (k, 2)) * size
                vi = thermal._flat_terms(*pi.T, *size) @ true_flat
                vj = thermal._flat_terms(*pj.T, *size) @ true_flat
                ci = np.clip(scene - drift[a] - vi + rng.normal(0, 1, k), 0, 255)
                cj = np.clip(scene - drift[b] - vj + rng.normal(0, 1, k), 0, 255)
                pairs.append(SimpleNamespace(i=used[a], j=used[b], pi=pi, pj=pj,
                                             ci=np.repeat(ci[:, None], 3, 1).astype(np.uint8),
                                             cj=np.repeat(cj[:, None], 3, 1).astype(np.uint8)))
    pairs[5].cj = np.clip(pairs[5].cj.astype(int) + 40, 0, 255).astype(np.uint8)   # occlusion/parallax outlier
    off, flat = thermal.solve_offsets(used, pairs, size)
    o = np.array([off[g][0] for g in used])
    assert set(off) == set(used) and all(v.shape == (3,) and v.dtype == np.float32 for v in off.values())
    assert abs(o.mean()) < 1e-3                      # the survey's absolute level is unchanged
    assert np.abs((o - drift) - (o - drift).mean()).max() < 1.0
    got, want = (thermal.flat_field(c, *size, (64, 52)) for c in (flat, true_flat))
    assert np.abs(got - want).max() < 1.0, np.abs(got - want).max()
    off2, flat2 = thermal.solve_offsets(used, pairs)          # offsets only
    assert flat2 is None and len(off2) == n
    assert thermal.solve_offsets(used, []) == ({}, None)
    # the decoder applies the flat field (zero mean: the frame average is unchanged)
    raw = np.full((64, 80), 1000, np.uint16)
    plain = thermal.values_to_gray(raw, (900.0, 1100.0), (80, 64))
    fixed = thermal.values_to_gray(raw, (900.0, 1100.0), (80, 64), thermal.flat_field(true_flat, 80, 64, (80, 64)))
    assert abs(fixed.astype(float).mean() - plain.astype(float).mean()) < 0.6 and fixed.std() > 0.5


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
