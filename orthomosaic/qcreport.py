"""Processing quality report (Agisoft Metashape / Pix4D style) from a reconstruction.

`build_3d` writes a `report.json` next to its products. This module turns that report into a
self-contained HTML quality report: survey summary, camera calibration, a camera-location plot,
reprojection error, ground-control accuracy, digital-model statistics with embedded previews, and
the processing parameters.

    python -m orthomosaic.qcreport recon/report.json -o recon/quality_report.html
    python -m orthomosaic.qcreport recon/                       # finds report.json, writes quality_report.html

    from orthomosaic import build_3d, write_quality_report
    rep = build_3d("images/", "recon/", Options3D())
    write_quality_report(rep, "recon/quality_report.html", product_dir="recon/")
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import math
import os
from typing import Optional, Union

# ------------------------------------------------------------------ helpers


def _fmt(v, nd=2, dash="-"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return dash
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _embed(path: Optional[str]) -> Optional[str]:
    """Read an image file and return a base64 data URI, or None if unavailable."""
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode("ascii")


def _row(label, value, unit=""):
    u = f' <span class="u">{unit}</span>' if unit else ""
    return f"<tr><th>{label}</th><td>{value}{u}</td></tr>"


def _camera_plot(cameras: dict, gcp: Optional[dict] = None, size: int = 460) -> str:
    """Top-down SVG scatter of estimated camera centres, coloured by height."""
    if not cameras:
        return "<p class='muted'>No camera positions available.</p>"
    xs = [c["C"][0] for c in cameras.values()]
    ys = [c["C"][1] for c in cameras.values()]
    zs = [c["C"][2] for c in cameras.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    pad, W, H = 30, size, size
    spanx = max(maxx - minx, 1e-6)
    spany = max(maxy - miny, 1e-6)
    scale = min((W - 2 * pad) / spanx, (H - 2 * pad) / spany)

    def px(x, y):
        cx = pad + (x - minx) * scale
        cy = H - pad - (y - miny) * scale          # north up
        return cx, cy

    def color(z):
        t = (z - minz) / max(maxz - minz, 1e-6)     # blue (low) -> red (high)
        r, g, b = int(40 + 215 * t), int(80 + 60 * (1 - abs(2 * t - 1))), int(230 - 200 * t)
        return f"rgb({r},{g},{b})"

    dots = []
    for c in cameras.values():
        cx, cy = px(c["C"][0], c["C"][1])
        dots.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{color(c["C"][2])}" '
                    f'fill-opacity="0.85"/>')
    marks = []
    if gcp and gcp.get("per_gcp"):
        # GCP world error is a delta; the plot uses camera frame, so just annotate names near centroid
        pass
    return (f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" class="plot">'
            f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fbfbfd" stroke="#e3e3ea"/>'
            f'{"".join(dots)}{"".join(marks)}'
            f'<text x="{pad}" y="{H-8}" class="axis">{_fmt(minx,0)} m E</text>'
            f'<text x="{W-pad}" y="{H-8}" text-anchor="end" class="axis">{_fmt(maxx,0)} m E</text>'
            f'<text x="6" y="{H-pad}" class="axis" transform="rotate(-90 6 {H-pad})">{_fmt(miny,0)} m N</text>'
            f'</svg>'
            f'<div class="legend"><span>low {_fmt(minz,1)} m</span>'
            f'<span class="bar"></span><span>{_fmt(maxz,1)} m high</span></div>')


def _calibration_groups(cameras: dict):
    """Group cameras by identical sensor + calibration (Metashape 'Sensors' table)."""
    groups = {}
    for name, c in cameras.items():
        key = (c["width"], c["height"], round(c["f"], 1), round(c["k1"], 4), round(c["k2"], 4))
        groups.setdefault(key, {"cam": c, "count": 0})
        groups[key]["count"] += 1
    return list(groups.values())


# ------------------------------------------------------------------ report


def write_quality_report(report: Union[dict, str], out_html: str,
                         product_dir: Optional[str] = None, title: Optional[str] = None) -> str:
    """Render a quality report HTML from a build_3d report (dict, or path to report.json).

    product_dir: folder holding the previews (dsm_preview.png, orthophoto_preview.jpg, ...).
                 Defaults to the report.json's folder, or out_html's folder for a dict."""
    if isinstance(report, str):
        product_dir = product_dir or os.path.dirname(os.path.abspath(report))
        with open(report) as fh:
            report = json.load(fh)
    product_dir = product_dir or os.path.dirname(os.path.abspath(out_html))
    R = report
    title = title or f"Processing Quality Report — {os.path.basename(os.path.normpath(product_dir))}"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    sfm = R.get("sfm", {}) or {}
    dsm = R.get("dsm", {}) or {}
    dtm = R.get("dtm") or {}
    cams = R.get("cameras", {}) or {}
    opts = R.get("options", {}) or {}
    gcp = sfm.get("gcp") or {}

    # ---- survey summary
    area = None
    if dsm.get("bounds"):
        b = dsm["bounds"]
        area = abs((b[2] - b[0]) * (b[3] - b[1]))
    heights = [c["C"][2] for c in cams.values()] if cams else []
    mean_h = sum(heights) / len(heights) if heights else None
    crs = f"EPSG:{R['epsg']}" if R.get("georeferenced") and R.get("epsg") else "local (not georeferenced)"

    summary = "".join([
        _row("Images", f"{_fmt(R.get('images_used'))} used / {_fmt(R.get('images_total'))} total"),
        _row("Images skipped", _fmt(len(R.get("images_dropped", [])))),
        _row("Coordinate system", crs),
        _row("Ground area", _fmt(area, 0) if area else "-", "m²"),
        _row("DSM ground sampling", _fmt(dsm.get("gsd", 0) * 100, 1) if dsm.get("gsd") else "-", "cm/px"),
        _row("Mean camera height", _fmt(mean_h, 1), "m (above datum)"),
        _row("Height datum", R.get("z_datum") or sfm.get("z_datum") or "-"),
        _row("Dense points", _fmt(R.get("dense_points"))),
        _row("Compute backend", R.get("backend", "-")),
        _row("Processing time", _fmt((R.get("seconds") or 0) / 60.0, 1), "min"),
    ])

    # ---- reconstruction quality
    n_pts = sfm.get("points", 0)
    n_obs = sfm.get("observations", 0)
    recon = "".join([
        _row("Tie points", _fmt(n_pts)),
        _row("Projections (observations)", _fmt(n_obs)),
        _row("Mean track length", _fmt(n_obs / n_pts, 2) if n_pts else "-", "images/point"),
        _row("Verified image pairs", _fmt(sfm.get("pairs"))),
        _row("Reprojection error (RMS)", _fmt(sfm.get("rms_px"), 2), "px"),
    ])

    # ---- calibration table
    cal_rows = []
    for i, g in enumerate(_calibration_groups(cams)):
        c = g["cam"]
        cal_rows.append(
            f"<tr><td>Sensor {i+1}</td><td>{g['count']}</td><td>{c['width']}×{c['height']}</td>"
            f"<td>{_fmt(c['f'], 1)}</td><td>{_fmt(c['cx'], 1)}, {_fmt(c['cy'], 1)}</td>"
            f"<td>{_fmt(c['k1'], 5)}</td><td>{_fmt(c['k2'], 5)}</td></tr>")
    cal_table = ("<table class='grid'><thead><tr><th>Sensor</th><th>Images</th><th>Resolution</th>"
                 "<th>Focal (px)</th><th>Principal point (px)</th><th>k1</th><th>k2</th></tr></thead>"
                 f"<tbody>{''.join(cal_rows) or '<tr><td colspan=7>-</td></tr>'}</tbody></table>")

    # ---- GCP control
    gcp_html = ""
    if gcp.get("per_gcp"):
        rows = []
        for name, g in gcp["per_gcp"].items():
            e = g["world_error_m"]
            rows.append(f"<tr><td>{name}</td><td>{_fmt(e[0],3)}</td><td>{_fmt(e[1],3)}</td>"
                        f"<td>{_fmt(e[2],3)}</td><td>{_fmt(g['world_error_norm_m'],3)}</td>"
                        f"<td>{_fmt(g['n_marks'])}</td><td>{_fmt(g.get('reproj_rms_px'),2)}</td></tr>")
        s = gcp.get("summary", {})
        gcp_html = f"""
        <section><h2>Ground Control Points</h2>
        <table class="grid"><thead><tr><th>GCP</th><th>ΔX (m)</th><th>ΔY (m)</th><th>ΔZ (m)</th>
        <th>Error (m)</th><th>Marks</th><th>Reproj (px)</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table>
        <table class="kv"><tbody>
        {_row("Control points", _fmt(s.get('count')))}
        {_row("RMSE 3D", _fmt(s.get('rmse_3d_m'), 3), "m")}
        {_row("RMSE horizontal", _fmt(s.get('rmse_horizontal_m'), 3), "m")}
        {_row("RMSE vertical", _fmt(s.get('rmse_vertical_m'), 3), "m")}
        </tbody></table></section>"""

    # ---- digital models
    dm_rows = "".join([
        _row("DSM size", f"{_fmt(dsm.get('width'))} × {_fmt(dsm.get('height'))}", "px"),
        _row("DSM cell size", _fmt(dsm.get("gsd", 0) * 100, 1) if dsm.get("gsd") else "-", "cm"),
        _row("DSM confident cells", _fmt((dsm.get("confident_fraction") or 0) * 100, 1), "%"),
        _row("DSM elevation range", (f"{_fmt(dsm['z_range'][0],1)} – {_fmt(dsm['z_range'][1],1)}"
                                     if dsm.get("z_range") else "-"), "m"),
        _row("DTM ground cells", _fmt((dtm.get("ground_fraction") or 0) * 100, 1) if dtm else "-", "%"),
        _row("DTM elevation range", (f"{_fmt(dtm['z_range'][0],1)} – {_fmt(dtm['z_range'][1],1)}"
                                     if dtm.get("z_range") else "-"), "m"),
    ])
    ortho_img = _embed(os.path.join(product_dir, "orthophoto_preview.jpg"))
    dsm_img = _embed(os.path.join(product_dir, "dsm_preview.png"))
    dtm_img = _embed(os.path.join(product_dir, "dtm_preview.png"))
    figs = ""
    for src, cap in [(ortho_img, "Orthophoto"), (dsm_img, "Digital Surface Model"),
                     (dtm_img, "Digital Terrain Model")]:
        if src:
            figs += f'<figure><img src="{src}" alt="{cap}"/><figcaption>{cap}</figcaption></figure>'

    # ---- processing parameters
    def _v(x):
        return ", ".join(str(i) for i in x) if isinstance(x, list) else str(x)
    param_rows = "".join(_row(k, _v(v)) for k, v in sorted(opts.items()))

    # ---- thermal note
    thermal_html = ""
    if R.get("thermal"):
        t = R["thermal"]
        rng = t.get("raw_range") or [None, None]
        rng_txt = f"{_fmt(rng[0], 0)} – {_fmt(rng[1], 0)}"
        thermal_html = (f"<section><h2>Thermal</h2><table class='kv'><tbody>"
                        f"{_row('Palette', t.get('palette'))}"
                        f"{_row('Raw value range', rng_txt)}"
                        f"</tbody></table><p class='muted'>Raw sensor units (monotonic with temperature); "
                        f"radiometric-to-°C conversion is not applied.</p></section>")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --ink:#1a1a22; --muted:#6b6b78; --line:#e3e3ea; --accent:#2b6cb0; --bg:#fff; }}
  * {{ box-sizing:border-box; }}
  body {{ font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--ink);
         background:var(--bg); margin:0; padding:0 16px 64px; }}
  .wrap {{ max-width:900px; margin:0 auto; }}
  header {{ border-bottom:3px solid var(--accent); padding:28px 0 16px; margin-bottom:8px; }}
  header h1 {{ margin:0 0 4px; font-size:24px; }}
  header .sub {{ color:var(--muted); }}
  section {{ margin:28px 0; }}
  h2 {{ font-size:16px; border-bottom:1px solid var(--line); padding-bottom:6px; margin:0 0 12px;
        text-transform:uppercase; letter-spacing:.04em; color:var(--accent); }}
  table {{ border-collapse:collapse; width:100%; margin:0 0 8px; }}
  table.kv th {{ text-align:left; font-weight:500; color:var(--muted); width:45%; padding:5px 10px;
                 border-bottom:1px solid var(--line); vertical-align:top; }}
  table.kv td {{ padding:5px 10px; border-bottom:1px solid var(--line); }}
  table.grid th, table.grid td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line);
                                   font-variant-numeric:tabular-nums; }}
  table.grid thead th {{ color:var(--muted); font-weight:600; border-bottom:2px solid var(--line); }}
  .u {{ color:var(--muted); font-size:.85em; }}
  .muted {{ color:var(--muted); }}
  .two {{ display:flex; gap:24px; flex-wrap:wrap; align-items:flex-start; }}
  .two > * {{ flex:1 1 320px; }}
  .plot {{ border-radius:6px; }}
  .legend {{ display:flex; align-items:center; gap:8px; color:var(--muted); font-size:12px; margin-top:6px; }}
  .legend .bar {{ flex:1; height:8px; border-radius:4px;
                  background:linear-gradient(90deg, rgb(30,80,230), rgb(120,140,120), rgb(230,120,30)); }}
  .axis {{ fill:var(--muted); font-size:10px; }}
  figure {{ margin:0 0 16px; }}
  figure img {{ width:100%; border:1px solid var(--line); border-radius:6px; }}
  figcaption {{ color:var(--muted); font-size:12px; margin-top:4px; text-align:center; }}
  .figs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; }}
  footer {{ color:var(--muted); font-size:12px; border-top:1px solid var(--line); padding-top:12px; margin-top:40px; }}
  @media print {{ body {{ padding:0; }} section {{ break-inside:avoid; }} }}
</style></head>
<body><div class="wrap">
<header><h1>{title}</h1>
<div class="sub">Generated {now} · pyOrthomosaic quality report</div></header>

<section><h2>Survey Summary</h2><table class="kv"><tbody>{summary}</tbody></table></section>

<section><h2>Camera Locations</h2>
<div class="two">
  <div>{_camera_plot(cams, gcp)}</div>
  <div><table class="kv"><tbody>{recon}</tbody></table></div>
</div></section>

<section><h2>Camera Calibration</h2>{cal_table}
<p class="muted">Focal length is held at the EXIF value on pure-nadir flights (focal/depth ambiguity);
radial distortion (k1, k2) is self-calibrated in the bundle adjustment.</p></section>

{gcp_html}

<section><h2>Digital Models</h2>
<table class="kv"><tbody>{dm_rows}</tbody></table>
<div class="figs">{figs or "<p class='muted'>No previews found in the product folder.</p>"}</div>
</section>

{thermal_html}

<section><h2>Processing Parameters</h2><table class="kv"><tbody>{param_rows}</tbody></table></section>

<footer>pyOrthomosaic — native Cython/CUDA/Metal photogrammetry. This report is generated from
<code>report.json</code>.</footer>
</div></body></html>"""

    with open(out_html, "w") as fh:
        fh.write(html)
    return out_html


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m orthomosaic.qcreport",
                                 description="Quality report (HTML) from a build_3d report.json")
    ap.add_argument("report", help="path to report.json, or a reconstruction output folder")
    ap.add_argument("-o", "--output", default=None, help="output HTML (default: <dir>/quality_report.html)")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    path = a.report
    if os.path.isdir(path):
        product_dir = path
        path = os.path.join(path, "report.json")
    else:
        product_dir = os.path.dirname(os.path.abspath(path))
    if not os.path.isfile(path):
        raise SystemExit(f"report.json not found: {path}")
    out = a.output or os.path.join(product_dir, "quality_report.html")
    write_quality_report(path, out, product_dir=product_dir, title=a.title)
    print(out)


if __name__ == "__main__":
    main()
