"""Update ViewX Documentation_Page HTML files for v0.2.5 and Presentation/Slide API."""

from pathlib import Path

DOC = Path(__file__).resolve().parent.parent.parent / "Libreria_Visual" / "Documentation_Page"

for html in DOC.glob("*.html"):
    text = html.read_text(encoding="utf-8")
    text = text.replace("v0.2.4", "v0.2.5")
    text = text.replace(">Slides</a>", ">Presentation</a>")
    text = text.replace('class="active">Slides</a>', 'class="active">Presentation</a>')
    text = text.replace("<h3>Slides</h3>", "<h3>Presentation + Slide</h3>")
    text = text.replace("ViewX Slides permite", "ViewX Presentation (contenedor de Slide) permite")
    text = text.replace("from viewx.Slides import Slides", "from viewx import Presentation, Slide")
    text = text.replace("Slides(load_iris(), auto=True", "Presentation.auto_generate(load_iris(),")
    text = text.replace('<title>Slides — ViewX</title>', '<title>Presentation — ViewX</title>')
    text = text.replace("/ Slides</div>", "/ Presentation</div>")
    text = text.replace('<h1 class="page-title">Slides</h1>', '<h1 class="page-title">Presentation</h1>')
    if "from_report_payload" not in text and html.name == "index.html":
        insert = """
      <h3>StatsLibX integration</h3>
      <pre><code class="language-python">from statslibx import DescriptiveStats, load_iris
summary = DescriptiveStats(load_iris()).summary()
summary.to_html("report.html", theme="dark_enterprise", data=load_iris())</code></pre>
"""
        text = text.replace("</main>", insert + "\n</main>", 1)
    html.write_text(text, encoding="utf-8")
    print("Updated", html.name)
