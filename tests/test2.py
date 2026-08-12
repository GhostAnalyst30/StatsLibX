"""ViewX integration pipeline demo — statslibx analysis then ViewX export."""

from statslibx import DescriptiveStats, load_iris

data = load_iris()
summary = DescriptiveStats(data).summary()

try:
    path = summary.to_html(
        filename="Dashboard_Iris.html",
        theme="dark_enterprise",
        include_figures=True,
        data=data,
        show=False,
    )
    print(f"Dashboard exported: {path}")
except ImportError as exc:
    print(exc)
