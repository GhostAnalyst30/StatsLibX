"""
ViewX export mixin for statslibx result objects.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from ._availability import require_viewx

HTMLTheme = Literal[
    "corporate_blue",
    "dark_enterprise",
    "modern_green",
    "void_indigo",
    "glass_ocean",
    "cyberpunk_neon",
]

PresentationTheme = Literal["dark", "light", "neon", "ocean", "sunset", "corporate"]


class ViewXExportMixin:
    """
    Mixin that adds ViewX export methods to statslibx result objects.

    All export methods require ``viewx`` (``pip install statslibx[viewx]``).
    Core statslibx analysis works without ViewX installed.
    """

    def to_report_data(
        self,
        include_figures: bool = False,
        data: Optional[Any] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Serialize this result into a ViewX-friendly payload.

        Parameters
        ----------
        include_figures : bool
            When True, attach Plotly figures to the payload (may require ``data``).
        data : DataFrame, optional
            Source data used to build distribution/correlation figures.
        **kwargs
            Forwarded to the adapter serializer.

        Returns
        -------
        dict
            Payload with keys ``title``, ``sections``, ``tables``, optional ``valueboxes`` and ``figures``.
        """
        from .adapters import to_report_data

        return to_report_data(
            self,
            include_figures=include_figures,
            data=data,
            **kwargs,
        )

    def to_html(
        self,
        filename: str = "report.html",
        theme: HTMLTheme = "dark_enterprise",
        include_figures: bool = True,
        data: Optional[Any] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Export this result to an interactive HTML dashboard via ViewX.

        Parameters
        ----------
        filename : str
            Output HTML path.
        theme : HTML theme name supported by ViewX ThemeManager.
        include_figures : bool
            Include Plotly figures when available.
        data : DataFrame, optional
            Source data for descriptive figures.
        show : bool
            Open the generated file in a browser.
        **kwargs
            Extra options forwarded to the renderer.

        Returns
        -------
        str
            Path to the generated HTML file.

        Raises
        ------
        ImportError
            If ViewX is not installed.
        """
        require_viewx("exportación HTML")
        from .renderers import render_html

        return render_html(
            self,
            filename=filename,
            theme=theme,
            include_figures=include_figures,
            data=data,
            show=show,
            **kwargs,
        )

    def to_report(
        self,
        filename: str = "stats_report",
        outdir: str = "output",
        include_figures: bool = False,
        data: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Export this result to a PDF report via ViewX Report engine.

        Parameters
        ----------
        filename : str
            Base filename without extension.
        outdir : str
            Output directory for PDF and assets.
        include_figures : bool
            Embed figures as images when possible.
        data : DataFrame, optional
            Source data for optional figures.
        **kwargs
            Extra renderer options.

        Returns
        -------
        str
            Path to the generated PDF.

        Raises
        ------
        ImportError
            If ViewX is not installed.
        """
        require_viewx("exportación PDF")
        from .renderers import render_report

        return render_report(
            self,
            filename=filename,
            outdir=outdir,
            include_figures=include_figures,
            data=data,
            **kwargs,
        )

    def to_presentation(
        self,
        filename: str = "presentation.html",
        theme: PresentationTheme = "dark",
        include_figures: bool = True,
        data: Optional[Any] = None,
        open_browser: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Export this result to an HTML slide deck via ViewX Presentation + Slide.

        Parameters
        ----------
        filename : str
            Output HTML path for the deck.
        theme : Presentation theme name.
        include_figures : bool
            Include Plotly charts on dedicated slides when available.
        data : DataFrame, optional
            Source data for descriptive figures.
        open_browser : bool
            Open the deck after export.
        **kwargs
            Extra renderer options.

        Returns
        -------
        str
            Path to the generated presentation file.

        Raises
        ------
        ImportError
            If ViewX is not installed.
        """
        require_viewx("exportación de presentación")
        from .renderers import render_presentation

        return render_presentation(
            self,
            filename=filename,
            theme=theme,
            include_figures=include_figures,
            data=data,
            open_browser=open_browser,
            **kwargs,
        )
