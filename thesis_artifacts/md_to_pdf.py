#!/usr/bin/env python3
"""Render a Markdown file to a clean, print-friendly PDF via headless Chrome."""
import sys, subprocess, tempfile, os, html, markdown

src = sys.argv[1]
out = sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "Document"

with open(src) as f:
    md_text = f.read()

body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"],
)

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Georgia", "Iowan Old Style", "Palatino", serif;
  font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; max-width: 100%;
}
h1 {
  font-family: "Helvetica Neue", Arial, sans-serif; font-size: 21pt;
  color: #0b3d5c; border-bottom: 3px solid #0b3d5c; padding-bottom: 6px;
  margin: 0 0 4px 0; line-height: 1.15;
}
h2 {
  font-family: "Helvetica Neue", Arial, sans-serif; font-size: 14pt;
  color: #0b3d5c; background: #eef4f8; padding: 7px 10px; border-radius: 4px;
  margin: 22px 0 10px 0; border-left: 5px solid #0b3d5c;
  page-break-after: avoid;
}
h3 {
  font-family: "Helvetica Neue", Arial, sans-serif; font-size: 11.5pt;
  color: #14506e; margin: 16px 0 4px 0; padding-bottom: 2px;
  border-bottom: 1px dotted #b8cdd9; page-break-after: avoid;
}
/* keep a task block (h3 + its paragraphs) together when possible */
h3 + p, h3 + p + p { page-break-inside: avoid; }
p { margin: 4px 0 7px 0; }
strong { color: #08334d; }
code {
  font-family: "SF Mono", "Consolas", monospace; font-size: 9pt;
  background: #f3f3f0; padding: 1px 4px; border-radius: 3px; color: #9b2c2c;
}
table {
  border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9pt;
  page-break-inside: avoid;
}
th { background: #0b3d5c; color: #fff; text-align: left; padding: 6px 8px;
     font-family: "Helvetica Neue", Arial, sans-serif; }
td { border: 1px solid #cfd8dc; padding: 5px 8px; vertical-align: top; }
tr:nth-child(even) td { background: #f6f9fb; }
hr { border: none; border-top: 2px solid #d0dae1; margin: 18px 0; }
ul, ol { margin: 4px 0 8px 0; padding-left: 22px; }
li { margin: 2px 0; }
blockquote {
  border-left: 4px solid #c9b458; background: #fdfaf0; margin: 8px 0;
  padding: 6px 12px; color: #4a4326;
}
/* tier banners by emoji marker in headings handled inline */
.tier-a { border-left-color: #2e7d32 !important; }
"""

html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body>{body}</body></html>"""

with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as tf:
    tf.write(html_doc)
    html_path = tf.name

profile = tempfile.mkdtemp(prefix="chrome-pdf-")
cmd = [
    "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
    f"--user-data-dir={profile}",
    "--no-pdf-header-footer", "--print-to-pdf-no-header",
    f"--print-to-pdf={out}", f"file://{html_path}",
]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if not os.path.exists(out):
    sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
    sys.exit("PDF not produced")
print(f"wrote {out} ({os.path.getsize(out)} bytes)")
os.unlink(html_path)
