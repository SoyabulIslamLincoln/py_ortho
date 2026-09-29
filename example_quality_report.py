"""Example: generate an Agisoft/Pix4D-style quality report from a pyOrthomosaic 3D run.

Three equivalent ways to get the report:

1. Straight after build_3d, from the returned report dict:

       from orthomosaic import build_3d, Options3D, write_quality_report
       rep = build_3d("images/", "recon/", Options3D())
       write_quality_report(rep, "recon/quality_report.html", product_dir="recon/")

2. Later, from a report.json that build_3d already wrote:

       from orthomosaic import write_quality_report
       write_quality_report("recon/report.json", "recon/quality_report.html")

3. From the command line, pointing at the reconstruction folder:

       python -m orthomosaic.qcreport recon/ -o recon/quality_report.html

This file shows how to wire option (1) into a make_3d.py-style driver: it reconstructs a folder and
writes the HTML report next to the products.
"""
import argparse
import webbrowser
from pathlib import Path

from orthomosaic import Options3D, build_3d, write_quality_report


def reconstruct_with_report(images, out_dir, open_browser=False, **opt_kwargs):
    out_dir = Path(out_dir)
    report = build_3d(str(images), str(out_dir), Options3D(**opt_kwargs))

    # ---- the one extra call: turn report.json into an HTML quality report
    html = out_dir / "quality_report.html"
    write_quality_report(report, str(html), product_dir=str(out_dir),
                         title=f"Quality Report — {out_dir.name}")
    print(f"quality report -> {html}")
    if open_browser:
        webbrowser.open(html.resolve().as_uri())
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reconstruct a folder and write a quality report")
    ap.add_argument("images", help="folder of nadir drone images")
    ap.add_argument("-o", "--out-dir", default="recon")
    ap.add_argument("--backend", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--open", action="store_true", help="open the report in a browser when done")
    a = ap.parse_args()
    reconstruct_with_report(a.images, a.out_dir, open_browser=a.open, backend=a.backend)
