"""
High-level ViewX renderers for statslibx result objects.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from ._availability import require_viewx
from .adapters import to_report_data
from .export import HTMLTheme, PresentationTheme

ReportTarget = Literal["html", "report", "presentation"]


def render_html(
    result: Any,
    filename: str = "report.html",
    theme: HTMLTheme = "dark_enterprise",
    include_figures: bool = True,
    data: Optional[Any] = None,
    show: bool = False,
    **kwargs: Any,
) -> str:
    """
    Render a statslibx result to an HTML dashboard using ViewX.

    Parameters
    ----------
    result : stats result object
        Any object supported by ``to_report_data()``.
    filename : str
        Output HTML path.
    theme : HTMLTheme
        ViewX dashboard theme.
    include_figures : bool
        Include Plotly figures in the payload.
    data : DataFrame, optional
        Source data for descriptive charts.
    show : bool
        Open browser after generation.
    **kwargs
        Forwarded to ``from_report_payload``.

    Returns
    -------
    str
        Generated file path.

    Raises
    ------
    ImportError
        If ViewX is not installed.
    """
    require_viewx("render_html")
    from viewx.shared.stats_payload import from_report_payload

    payload = to_report_data(result, include_figures=include_figures, data=data)
    return from_report_payload(
        payload,
        target="html",
        filename=filename,
        theme=theme,
        show=show,
        **kwargs,
    )


def render_report(
    result: Any,
    filename: str = "stats_report",
    outdir: str = "output",
    include_figures: bool = False,
    data: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """
    Render a statslibx result to a PDF report using ViewX.

    Parameters
    ----------
    result : stats result object
        Any object supported by ``to_report_data()``.
    filename : str
        Base filename without extension.
    outdir : str
        Output directory.
    include_figures : bool
        Embed figure images when possible.
    data : DataFrame, optional
        Source data for optional charts.
    **kwargs
        Forwarded to ``from_report_payload``.

    Returns
    -------
    str
        Path to generated PDF.

    Raises
    ------
    ImportError
        If ViewX is not installed.
    """
    require_viewx("render_report")
    from viewx.shared.stats_payload import from_report_payload

    payload = to_report_data(result, include_figures=include_figures, data=data)
    return from_report_payload(
        payload,
        target="report",
        filename=filename,
        outdir=outdir,
        **kwargs,
    )


def render_presentation(
    result: Any,
    filename: str = "presentation.html",
    theme: PresentationTheme = "dark",
    include_figures: bool = True,
    data: Optional[Any] = None,
    open_browser: bool = False,
    **kwargs: Any,
) -> str:
    """
    Render a statslibx result to an HTML slide deck using ViewX Presentation.

    Parameters
    ----------
    result : stats result object
        Any object supported by ``to_report_data()``.
    filename : str
        Output HTML path.
    theme : PresentationTheme
        Slide deck theme.
    include_figures : bool
        Include chart slides when figures are available.
    data : DataFrame, optional
        Source data for descriptive charts.
    open_browser : bool
        Open browser after export.
    **kwargs
        Forwarded to ``from_report_payload``.

    Returns
    -------
    str
        Path to generated presentation HTML.

    Raises
    ------
    ImportError
        If ViewX is not installed.
    """
    require_viewx("render_presentation")
    from viewx.shared.stats_payload import from_report_payload

    payload = to_report_data(result, include_figures=include_figures, data=data)
    return from_report_payload(
        payload,
        target="presentation",
        filename=filename,
        theme=theme,
        open_browser=open_browser,
        **kwargs,
    )


def render(
    result: Any,
    target: ReportTarget = "html",
    **kwargs: Any,
) -> str:
    """
    Dispatch renderer by target engine name.

    Parameters
    ----------
    result : stats result object
        Object to serialize and render.
    target : {'html', 'report', 'presentation'}
        ViewX engine to use.
    **kwargs
        Forwarded to the specific renderer.

    Returns
    -------
    str
        Path to generated artifact.
    """
    if target == "html":
        return render_html(result, **kwargs)
    if target == "report":
        return render_report(result, **kwargs)
    return render_presentation(result, **kwargs)
