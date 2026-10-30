import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "reports" / "REPORT.md"
OUT = ROOT / "reports" / "REPORT.html"
FIGDIR = ROOT / "reports" / "figures"

text = MD.read_text(encoding="utf-8")

# Встраиваем PNG как base64 (чтобы HTML был самодостаточным)
def embed(match):
    alt, src = match.group(1), match.group(2)
    p = FIGDIR / Path(src).name
    if not p.exists():
        return match.group(0)
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

text = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", embed, text)

html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'DejaVu Sans', 'PT Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1b1b1b; max-width: 900px; margin: 0 auto; padding: 24px; }
h1 { font-size: 20pt; border-bottom: 3px solid #2b6cb0; padding-bottom: 6px; margin-top: 0; color: #1a365d; }
h2 { font-size: 14pt; margin-top: 26px; color: #1a365d; border-bottom: 1px solid #cbd5e0; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 18px; color: #2d3748; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #cbd5e0; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #edf2f7; font-weight: 600; }
tr:nth-child(even) td { background: #f7fafc; }
code { background: #edf2f7; padding: 1px 4px; border-radius: 3px; font-family: 'DejaVu Sans Mono', Consolas, monospace; font-size: 9pt; }
pre { background: #1a202c; color: #e2e8f0; padding: 12px; border-radius: 6px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; color: inherit; padding: 0; }
blockquote { border-left: 4px solid #4299e1; margin: 10px 0; padding: 4px 12px; background: #ebf8ff; color: #2a4365; }
figure { margin: 18px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; }
figcaption { font-size: 9pt; color: #4a5568; margin-top: 4px; }
.toc { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 16px; margin: 14px 0 22px; font-size: 9.5pt; }
.toc ul { list-style: none; padding-left: 10px; margin: 4px 0; }
.toc > ul { padding-left: 0; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 18px 0; }
@media print { body { padding: 0; max-width: none; } a { color: inherit; text-decoration: none; } }
"""

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Отчёт: vLLM inference test</title>
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
print("Откройте в браузере и нажмите Ctrl+P -> Сохранить как PDF")
