"""Writers for 3D outputs: PLY / LAS point clouds, OBJ and GLB textured meshes.
All pure Python + numpy (no PDAL / Open3D / trimesh)."""
from __future__ import annotations

import datetime
import io
import json
import struct
from typing import Optional

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# point clouds
# --------------------------------------------------------------------------

def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray, offset=(0.0, 0.0, 0.0), comment: str = ""):
    """Binary little-endian PLY. Coordinates are stored relative to `offset` (float32)."""
    n = len(xyz)
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ["ply", "format binary_little_endian 1.0",
              f"comment offset {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f}"]
    if comment:
        header.append(f"comment {comment}")
    header += [f"element vertex {n}", "property float x", "property float y", "property float z",
               "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        fh.write(rec.tobytes())


def write_las(path: str, xyz_abs: np.ndarray, rgb: np.ndarray, epsg: Optional[int] = None,
              scale: float = 0.001, classification: Optional[np.ndarray] = None):
    """LAS 1.2, point format 2 (XYZ + RGB), absolute coordinates, optional EPSG GeoKey VLR.
    classification: optional per-point ASPRS class (1 = unclassified, 2 = ground)."""
    n = len(xyz_abs)
    mins, maxs = xyz_abs.min(0), xyz_abs.max(0)
    offset = np.floor(mins)
    vlrs = b""
    if epsg:
        keys = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, int(epsg)]
        data = struct.pack("<%dH" % len(keys), *keys)
        vlrs = (struct.pack("<H", 0) + b"LASF_Projection".ljust(16, b"\0") + struct.pack("<HH", 34735, len(data))
                + b"GeoKeyDirectoryTag".ljust(32, b"\0") + data)
    header_size = 227
    offset_to_points = header_size + len(vlrs)
    today = datetime.date.today()
    hdr = bytearray()
    hdr += b"LASF" + struct.pack("<HH", 0, 0) + b"\0" * 16 + struct.pack("<BB", 1, 2)
    hdr += b"pyOrthomosaic".ljust(32, b"\0") + b"pyOrthomosaic".ljust(32, b"\0")
    hdr += struct.pack("<HHHIIBHI", today.timetuple().tm_yday, today.year, header_size, offset_to_points,
                       1 if epsg else 0, 2, 26, n)
    hdr += struct.pack("<5I", n, 0, 0, 0, 0)
    hdr += struct.pack("<3d", scale, scale, scale)
    hdr += struct.pack("<3d", *offset)
    hdr += struct.pack("<6d", maxs[0], mins[0], maxs[1], mins[1], maxs[2], mins[2])
    assert len(hdr) == header_size
    pts = np.zeros(n, dtype=[("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("I", "<u2"), ("ret", "u1"),
                             ("cls", "u1"), ("ang", "i1"), ("ud", "u1"), ("src", "<u2"),
                             ("R", "<u2"), ("G", "<u2"), ("B", "<u2")])
    q = np.round((xyz_abs - offset) / scale).astype(np.int64)
    pts["X"], pts["Y"], pts["Z"] = q[:, 0], q[:, 1], q[:, 2]
    pts["ret"] = 0b00001001          # return 1 of 1
    pts["cls"] = 1 if classification is None else classification   # ASPRS: 1 unclassified, 2 ground
    rgb16 = rgb.astype(np.uint16) * 257
    pts["R"], pts["G"], pts["B"] = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        fh.write(vlrs)
        fh.write(pts.tobytes())


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------

def grid_mesh(Z: np.ndarray, valid: np.ndarray, minX: float, maxY: float, cell: float):
    """Triangulate a height grid. Each quad is split along the diagonal with the smaller
    height difference (keeps roof ridges and walls sharp). Returns (V (n,3), F (m,3), UV (n,2))."""
    H, W = Z.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.nonzero(valid)
    idx[ys, xs] = np.arange(len(ys))
    V = np.column_stack([minX + (xs + 0.5) * cell, maxY - (ys + 0.5) * cell, Z[ys, xs]]).astype(np.float64)
    UV = np.column_stack([(xs + 0.5) / W, 1.0 - (ys + 0.5) / H])
    a, b = idx[:-1, :-1], idx[:-1, 1:]
    c, d = idx[1:, :-1], idx[1:, 1:]
    za, zb, zc, zd = Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]
    diag_ad = np.abs(za - zd) <= np.abs(zb - zc)
    tris = []
    for m, t in [(diag_ad, (a, c, d)), (diag_ad, (a, d, b)), (~diag_ad, (a, c, b)), (~diag_ad, (b, c, d))]:
        f = np.stack([t[0][m], t[1][m], t[2][m]], 1)
        tris.append(f[(f >= 0).all(1)])
    F = np.concatenate(tris)
    return V, F, UV


def write_obj(path: str, V: np.ndarray, F: np.ndarray, UV: np.ndarray, texture: Image.Image,
              offset=(0.0, 0.0, 0.0)):
    """Wavefront OBJ + MTL + JPEG texture (vertices relative to `offset`)."""
    import os
    base = os.path.splitext(path)[0]
    name = os.path.basename(base)
    texture.convert("RGB").save(base + "_texture.jpg", quality=90)
    with open(base + ".mtl", "w") as fh:
        fh.write(f"newmtl ortho\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd {name}_texture.jpg\n")
    Vl = V - np.asarray(offset)
    with open(path, "w") as fh:
        fh.write(f"# pyOrthomosaic mesh; add offset {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f} "
                 f"for absolute coordinates\nmtllib {name}.mtl\nusemtl ortho\n")
        np.savetxt(fh, Vl, fmt="v %.3f %.3f %.3f")
        np.savetxt(fh, UV, fmt="vt %.6f %.6f")
        f1 = F + 1
        np.savetxt(fh, np.column_stack([f1[:, 0], f1[:, 0], f1[:, 1], f1[:, 1], f1[:, 2], f1[:, 2]]),
                   fmt="f %d/%d %d/%d %d/%d")


def write_glb(path: str, V: np.ndarray, F: np.ndarray, UV: np.ndarray, texture: Image.Image,
              offset=(0.0, 0.0, 0.0)):
    """Binary glTF 2.0 with an embedded JPEG texture. glTF is Y-up: (E, N, Z) -> (E, Z, -N)."""
    Vl = (V - np.asarray(offset)).astype(np.float32)
    pos = np.column_stack([Vl[:, 0], Vl[:, 2], -Vl[:, 1]]).astype(np.float32)
    # glTF UV origin is top-left
    uv = np.column_stack([UV[:, 0], 1.0 - UV[:, 1]]).astype(np.float32)
    ind = F.astype(np.uint32).ravel()
    buf = io.BytesIO()
    texture.convert("RGB").save(buf, "JPEG", quality=90)
    jpg = buf.getvalue()

    chunks, views = [], []

    def add(data: bytes, target=None):
        off = sum(len(c) for c in chunks)
        chunks.append(data + b"\0" * (-len(data) % 4))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    v_pos = add(pos.tobytes(), 34962)
    v_uv = add(uv.tobytes(), 34962)
    v_ind = add(ind.tobytes(), 34963)
    v_img = add(jpg)
    gltf = {
        "asset": {"version": "2.0", "generator": "pyOrthomosaic"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
                                                 "roughnessFactor": 1.0}, "doubleSided": True}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": v_img, "mimeType": "image/jpeg"}],
        "accessors": [
            {"bufferView": v_pos, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": v_uv, "componentType": 5126, "count": len(uv), "type": "VEC2"},
            {"bufferView": v_ind, "componentType": 5125, "count": len(ind), "type": "SCALAR"},
        ],
        "bufferViews": views,
        "extras": {"offset": list(map(float, offset)), "axes": "x=East, y=Up, z=-North"},
    }
    binary = b"".join(chunks)
    gltf["buffers"] = [{"byteLength": len(binary)}]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binary), 0x004E4942) + binary)
