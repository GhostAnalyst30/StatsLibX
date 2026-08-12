"""Append StatsLibX integration section to how_use_viewx.ipynb."""

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent.parent / "Libreria_Visual" / "how_use_viewx.ipynb"

MARKDOWN = """## Integración StatsLibX

StatsLibX analiza; ViewX visualiza al exportar. El payload `to_report_data()` se convierte con `from_report_payload()`.

Requiere: `pip install statslibx[viewx]`"""

CODE = '''from statslibx import DescriptiveStats, load_iris, to_report_data
from viewx import from_report_payload

df = load_iris()
summary = DescriptiveStats(df).summary()

# Opción directa en el resultado
try:
    path = summary.to_html("viewx_from_statslibx.html", data=df, show=False)
    print("Direct export:", path)
except ImportError as exc:
    print(exc)

# Pipeline manual
payload = to_report_data(summary, include_figures=True, data=df)
from_report_payload(payload, target="html", filename="viewx_payload.html")
print("Payload export complete")
'''

with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

text = "".join("".join(c.get("source", [])) for c in nb["cells"])
if "Integración StatsLibX" not in text:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [MARKDOWN + "\n"]})
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [line + "\n" for line in CODE.split("\n")],
        }
    )
    with NB.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Patched", NB)
else:
    print("Already patched", NB)
