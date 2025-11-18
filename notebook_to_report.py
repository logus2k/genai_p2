#!/usr/bin/env python3
"""
Strip text outputs from a Jupyter notebook while keeping charts (images),
then export to DOCX. Optional: export to HTML and Markdown, hide code cells.

Usage:
  python notebook_to_report.py path/to/notebook.ipynb [--out report_basename] [--no-code] [--html] [--markdown]

Requirements:
  pip install nbformat nbconvert
  # For DOCX:
  pip install pypandoc            # preferred
  # OR install system pandoc: https://pandoc.org/install.html

Examples:
  # Basic: just DOCX with code visible
  python notebook_to_report.py my_run.ipynb
  # DOCX without code
  python notebook_to_report.py my_run.ipynb --no-code
  # DOCX + HTML + Markdown
  python notebook_to_report.py my_run.ipynb --html --markdown
"""

import argparse
from pathlib import Path
import sys

import nbformat as nbf
from nbconvert import HTMLExporter, MarkdownExporter
from traitlets.config import Config


def keep_only_image_outputs(nb):
    """Remove text-only outputs, keep image (PNG/SVG) and optional rich HTML outputs."""
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        new_outputs = []
        for out in cell.get("outputs", []):
            otype = out.get("output_type")
            data = out.get("data", {})
            has_img = any(k in data for k in ("image/png", "image/svg+xml"))
            keep_html = "text/html" in data  # set False if you don't want rich HTML
            if otype in {"display_data", "execute_result"} and (has_img or keep_html):
                new_outputs.append(out)
            # drop streams/errors/plain text-only execute_results
        cell["outputs"] = new_outputs
        cell["execution_count"] = None
    return nb


def export_html(nb, out_html, hide_code: bool):
    c = Config()
    # Hide prompts; optionally hide code inputs entirely
    c.HTMLExporter.exclude_input_prompt = True
    c.HTMLExporter.exclude_output_prompt = True
    if hide_code:
        c.HTMLExporter.exclude_input = True
    html_exporter = HTMLExporter(config=c)
    body, _ = html_exporter.from_notebook_node(nb)
    Path(out_html).write_text(body, encoding="utf-8")


def export_markdown(nb, out_md, assets_dir: str, hide_code: bool):
    """
    Export Markdown and extract images to assets_dir (nbconvert writes files there).
    """
    c = Config()
    c.MarkdownExporter.exclude_input_prompt = True
    c.MarkdownExporter.exclude_output_prompt = True
    if hide_code:
        c.MarkdownExporter.exclude_input = True
    resources = {"output_files_dir": assets_dir}
    md_exporter = MarkdownExporter(config=c)
    body, resources = md_exporter.from_notebook_node(nb, resources=resources)
    Path(out_md).write_text(body, encoding="utf-8")

    # Extract images - nbconvert includes the assets_dir in the fname
    for fname, data in resources.get("outputs", {}).items():
        out_path = Path(out_md).parent / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)


def markdown_to_docx(md_path, docx_path):
    """Convert Markdown to DOCX using pypandoc if available; else shell pandoc."""
    try:
        import pypandoc  # type: ignore
        pypandoc.convert_file(
            md_path,
            "docx",
            outputfile=docx_path,
            extra_args=["--standalone", "--resource-path=."],
        )
        return
    except Exception as e:
        print("[info] pypandoc not available or failed:", e)

    import subprocess
    cmd = ["pandoc", str(md_path), "-o", str(docx_path), "--standalone", "--resource-path=."]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        print(
            "[error] pandoc not found. Install it or `pip install pypandoc`.\n"
            "Download: https://pandoc.org/install.html"
        )
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", help="Path to source .ipynb")
    ap.add_argument("--out", help="Base name for outputs (without extension)")
    ap.add_argument(
        "--no-code",
        action="store_true",
        help="Hide code cells in the exported outputs (markdown + charts only).",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="Also export to HTML format.",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="Also export to Markdown format.",
    )
    args = ap.parse_args()

    src_nb = Path(args.notebook).resolve()
    if not src_nb.exists():
        print(f"[error] Notebook not found: {src_nb}")
        sys.exit(1)

    base = args.out if args.out else src_nb.with_suffix("").name
    out_html = src_nb.parent / f"{base}_report.html"
    out_md = src_nb.parent / f"{base}_report.md"
    assets_dir = f"{base}_report_files"
    out_docx = src_nb.parent / f"{base}_report.docx"

    print("[1/3] Reading notebook…")
    nb = nbf.read(str(src_nb), as_version=4)

    print("[2/3] Stripping text outputs, keeping charts…")
    nb_clean = keep_only_image_outputs(nb)

    # Export HTML if requested
    if args.html:
        print(f"[3a/3] Exporting HTML → {out_html} (hide_code={args.no_code})")
        export_html(nb_clean, str(out_html), hide_code=args.no_code)

    # Always export Markdown (needed for DOCX), but only mention if explicitly requested
    if args.markdown:
        print(f"[3b/3] Exporting Markdown (and images) → {out_md} (hide_code={args.no_code})")
    export_markdown(nb_clean, str(out_md), assets_dir=assets_dir, hide_code=args.no_code)

    # Always export DOCX
    print(f"[3/3] Converting to DOCX → {out_docx}")
    markdown_to_docx(str(out_md), str(out_docx))

    # Clean up intermediate files if neither markdown nor html was requested
    if not args.markdown:
        print("[cleanup] Removing intermediate Markdown files…")
        out_md.unlink(missing_ok=True)
        import shutil
        assets_path = src_nb.parent / assets_dir
        if assets_path.exists():
            shutil.rmtree(assets_path)

    print("\nDone.")
    print("Outputs:")
    if args.html:
        print("  • HTML report     :", out_html)
    if args.markdown:
        print("  • Markdown        :", out_md, f"(assets in ./{assets_dir}/)")
    print("  • Word (.docx)    :", out_docx)


if __name__ == "__main__":
    main()
