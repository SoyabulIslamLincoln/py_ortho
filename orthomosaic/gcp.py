"""Ground Control Points (GCPs): surveyed world points marked in the images.

GCPs anchor the whole photogrammetric block to real-world coordinates. Each GCP has a
surveyed position and one or more image "marks" (the pixel where it appears). In the bundle
adjustment a GCP is a 3D point whose world position is pulled toward the survey value with a
tight prior, while its image marks tie the cameras to it. A handful of well-spread GCPs take a
GPS-only block (metre-level) to survey-level absolute accuracy and remove doming.

File format (the WebODM / Pix4D GCP list, which most survey tools export):

    EPSG:32646                      # or:  WGS84 / +proj=longlat ... ; header line, coordinate system
    geo_x geo_y geo_z im_x im_y image_name [gcp_name]
    230012.5 2635008.1 12.3 2456.0 1810.5 DJI_0001_V.JPG GCP1
    ...

One row per (GCP, image) mark. Columns are whitespace- or comma-separated. Lines starting with
'#' are comments. If the header is a lat/lon system, coordinates are projected to the survey's
UTM zone automatically.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GCP:
    name: str
    world: np.ndarray                       # (3,) surveyed X, Y, Z in the survey CRS (UTM metres)
    marks: dict = field(default_factory=dict)   # image basename -> (u, v) full-res pixel


@dataclass
class GCPSet:
    gcps: list                              # list[GCP]
    epsg: Optional[int] = None
    is_latlon: bool = False

    def __len__(self):
        return len(self.gcps)

    @property
    def n_marks(self) -> int:
        return sum(len(g.marks) for g in self.gcps)


def _parse_header(line: str) -> tuple[Optional[int], bool]:
    """Return (epsg, is_latlon) from a WebODM/Pix4D GCP header line."""
    s = line.strip()
    m = re.search(r"EPSG:\s*(\d+)", s, re.I)
    if m:
        epsg = int(m.group(1))
        return epsg, epsg in (4326, 4979)
    if re.search(r"longlat|wgs84|lat[/ _-]*lon", s, re.I):
        return 4326, True
    return None, False


def load_gcps(path: str) -> GCPSet:
    """Parse a GCP list file. Coordinates stay in the file's CRS (projected later if lat/lon)."""
    epsg, is_latlon = None, False
    gcps: dict[str, GCP] = {}
    with open(path) as fh:
        lines = [ln for ln in fh]
    start = 0
    if lines and ("EPSG" in lines[0].upper() or "PROJ" in lines[0].upper() or "WGS84" in lines[0].upper()
                  or "LONGLAT" in lines[0].upper()):
        epsg, is_latlon = _parse_header(lines[0])
        start = 1
    for ln in lines[start:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", s)
        if len(parts) < 6:
            log.warning("GCP line ignored (need >= 6 columns): %s", s)
            continue
        try:
            gx, gy, gz, ix, iy = (float(v) for v in parts[:5])
        except ValueError:
            continue
        image = parts[5]
        name = parts[6] if len(parts) > 6 else None
        # group marks by GCP identity: explicit name, else the shared coordinate
        key = name or f"{gx:.4f}_{gy:.4f}_{gz:.4f}"
        g = gcps.get(key)
        if g is None:
            g = gcps[key] = GCP(name or key, np.array([gx, gy, gz], np.float64))
        import os
        g.marks[os.path.basename(image)] = (ix, iy)
    gset = GCPSet(list(gcps.values()), epsg, is_latlon)
    log.info("Loaded %d GCP(s) with %d image mark(s)%s", len(gset), gset.n_marks,
             f" (EPSG:{epsg})" if epsg else "")
    return gset


def to_local(gset: GCPSet, zone: int, north: bool, origin: tuple[float, float]) -> dict:
    """GCP world coords -> the reconstruction's local ENU frame (UTM minus origin).
    Returns {name: (world_local (3,), {image: (u, v)})} for GCPs that have image marks."""
    from . import geo
    out = {}
    for g in gset.gcps:
        if not g.marks:
            continue
        if gset.is_latlon:
            # file stores lat, lon in geo_x, geo_y OR lon, lat -- WebODM uses geo_x=lon, geo_y=lat
            lon, lat, z = g.world
            E, N = geo.latlon_to_utm(lat, lon, zone, north)
            w = np.array([float(E) - origin[0], float(N) - origin[1], z])
        else:
            w = np.array([g.world[0] - origin[0], g.world[1] - origin[1], g.world[2]])
        out[g.name] = (w, dict(g.marks))
    return out
