"""Patch ViewX section in how_use_statslibx.ipynb for v0.3.1 integration."""

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "how_use_statslibx.ipynb"

REPLACEMENTS = [
    (
        "| **viewx** | `HTML`, `Slides`, `Report`, `DataMatrix`, `to_report_data` |",
        "| **viewx** | `HTML`, `Presentation`, `Slide`, `Report`, `DataMatrix`, `to_report_data`, `.to_html()` |",
    ),
    (
        "Visualization and reporting bridge (`statslibx.viewx`). Requires `pip install viewx`.",
        "StatsLibX analiza; ViewX visualiza al exportar. **El análisis funciona sin ViewX.**\n\nInstala con: `pip install statslibx[viewx]`",
    ),
    (
        "    from statslibx import HTML, Slides, Report, DataMatrix, to_report_data\n",
        "    from statslibx import HTML, Presentation, Slide, Report, DataMatrix, to_report_data, VIEWX_AVAILABLE\n",
    ),
    (
        '    print("ViewX classes available: HTML, Slides, Report, DataMatrix")\n',
        '    print("ViewX available:", VIEWX_AVAILABLE)\n    print("Classes: HTML, Presentation, Slide, Report, DataMatrix")\n',
    ),
    (
        "Serialize `DescriptiveSummary`, `TestResult`, or `RegressionResult` to a dict for ViewX reports.",
        "Serialize results to a payload dict. Works **without** viewx installed.",
    ),
]

NEW_EXPORT_CELL = '''"""Export to HTML when ViewX is installed."""
from statslibx import DescriptiveStats, load_iris

df = load_iris()
summary = DescriptiveStats(df).summary()

try:
    path = summary.to_html(
        "notebook_viewx_export.html",
        theme="dark_enterprise",
        include_figures=True,
        data=df,
        show=False,
    )
    print("Exported:", path)
except ImportError as exc:
    print(exc)
'''

with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    src = cell.get("source", [])
    if isinstance(src, list):
        text = "".join(src)
        for old, new in REPLACEMENTS:
            if old in text:
                text = text.replace(old, new)
        cell["source"] = text.splitlines(keepends=True) if text else src

sources = ["".join(c.get("source", [])) for c in nb["cells"]]
if "notebook_viewx_export.html" not in "".join(sources):
    nb["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Exportar con `.to_html()`\n\n",
                "Genera un dashboard HTML desde un `DescriptiveSummary`. "
                "Si ViewX no está instalado, se captura el `ImportError`.\n",
            ],
        }
    )
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [line + "\n" for line in NEW_EXPORT_CELL.split("\n")],
        }
    )

with NB.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched", NB)
