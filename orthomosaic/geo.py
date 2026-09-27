"""WGS84 <-> UTM (transverse Mercator series, Snyder 1987). No PROJ needed."""
from __future__ import annotations

import numpy as np

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_K0 = 0.9996


def utm_zone(lat: float, lon: float) -> tuple[int, bool]:
    zone = int((lon + 180) // 6) + 1
    # Norway / Svalbard exceptions
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat < 84:
        if 0 <= lon < 9:
            zone = 31
        elif 9 <= lon < 21:
            zone = 33
        elif 21 <= lon < 33:
            zone = 35
        elif 33 <= lon < 42:
            zone = 37
    return min(max(zone, 1), 60), lat >= 0


def utm_epsg(zone: int, north: bool) -> int:
    return (32600 if north else 32700) + zone


def latlon_to_utm(lat, lon, zone: int, north: bool):
    """Vectorised forward projection. Returns (easting, northing) in metres."""
    lat = np.radians(np.asarray(lat, np.float64))
    lon = np.radians(np.asarray(lon, np.float64))
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    sin_p, cos_p, tan_p = np.sin(lat), np.cos(lat), np.tan(lat)
    N = _A / np.sqrt(1 - _E2 * sin_p ** 2)
    T = tan_p ** 2
    C = _EP2 * cos_p ** 2
    A = cos_p * (lon - lon0)
    e4, e6 = _E2 ** 2, _E2 ** 3
    M = _A * ((1 - _E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat
              - (3 * _E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * lat)
              + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * lat)
              - (35 * e6 / 3072) * np.sin(6 * lat))
    x = _K0 * N * (A + (1 - T + C) * A ** 3 / 6
                   + (5 - 18 * T + T ** 2 + 72 * C - 58 * _EP2) * A ** 5 / 120) + 500000.0
    y = _K0 * (M + N * tan_p * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                + (61 - 58 * T + T ** 2 + 600 * C - 330 * _EP2) * A ** 6 / 720))
    if not north:
        y = y + 10000000.0
    return x, y


def utm_to_latlon(x, y, zone: int, north: bool):
    """Vectorised inverse projection. Returns (lat, lon) in degrees."""
    x = np.asarray(x, np.float64) - 500000.0
    y = np.asarray(y, np.float64)
    if not north:
        y = y - 10000000.0
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    M = y / _K0
    mu = M / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    e1 = (1 - np.sqrt(1 - _E2)) / (1 + np.sqrt(1 - _E2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
          + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    sin_p, cos_p, tan_p = np.sin(p1), np.cos(p1), np.tan(p1)
    C1 = _EP2 * cos_p ** 2
    T1 = tan_p ** 2
    N1 = _A / np.sqrt(1 - _E2 * sin_p ** 2)
    R1 = _A * (1 - _E2) / (1 - _E2 * sin_p ** 2) ** 1.5
    D = x / (N1 * _K0)
    lat = p1 - (N1 * tan_p / R1) * (D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * _EP2) * D ** 4 / 24
                                    + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * _EP2 - 3 * C1 ** 2)
                                    * D ** 6 / 720)
    lon = lon0 + (D - (1 + 2 * T1 + C1) * D ** 3 / 6
                  + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * _EP2 + 24 * T1 ** 2) * D ** 5 / 120) / cos_p
    return np.degrees(lat), np.degrees(lon)
