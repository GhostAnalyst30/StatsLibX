import logging
import numpy as np
import pandas as pd
from typing import Any, Union, List, Optional, Literal, Tuple, Dict
import warnings
import os
from scipy import stats
from pathlib import Path

from ._stats_utils import (
    detect_outliers as _detect_outliers,
    check_normality as _check_normality,
    confidence_interval as _confidence_interval,
    analytic_ci as _analytic_ci,
    bootstrap_ci as _bootstrap_ci,
    cohens_d as _cohens_d,
    hedges_g as _hedges_g,
)

logger = logging.getLogger(__name__)


class UtilsStats:
    """
    UtilsStats
    A utility class for common statistical operations and visualization.
    This class provides methods for data validation, basic statistical analysis,
    and visualization of results. It also supports loading data directly from files.
    >>> # Load data from a file
    >>> data = utils.load_data("data.csv")
    >>> utils.check_normality(data, column='age')
    >>> # Analyze data from an array
    Methods:
    --------
    _setup_plotting_style():
        Configures default plotting styles for matplotlib.
    
    set_plot_backend(backend: Literal['matplotlib', 'seaborn', 'plotly']):
        Sets the default visualization backend.
    
    set_default_figsize(figsize: Tuple[int, int]):
        Sets the default figure size for plots.
    
    set_save_fig_options(save_fig: Optional[bool] = False, fig_format: str = 'png', 
                         fig_dpi: int = 300, figures_dir: str = 'figures'):
        Configures options for saving figures.
    
    load_data(path: Union[str, Path], **kwargs) -> pd.DataFrame:
        Loads data from a file in various formats (CSV, Excel, JSON, etc.).
    
    validate_dataframe(data: Union[pd.DataFrame, np.ndarray, list, str, Path]) -> pd.DataFrame:
        Validates and converts data to a DataFrame. Also accepts file paths.
    
    format_number(num: float, decimals: int = 6, scientific: bool = False) -> str:
        Formats a number with specified decimal places.
    
    check_normality(data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path], 
                    column: Optional[str] = None, alpha: float = 0.05) -> dict:
        Checks if the data follows a normal distribution using the Shapiro-Wilk test.
    
    calculate_confidence_intervals(data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path],
                                    column: Optional[str] = None, confidence_level: float = 0.95,
        Calculates confidence intervals for the mean using parametric or bootstrap methods.
    
    detect_outliers(data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path],
                    column: Optional[str] = None, method: Literal['iqr', 'zscore', 'isolation_forest'] = 'iqr',
        Detects outliers using different methods: 'iqr', 'zscore', or 'isolation_forest'.
    
    calculate_effect_size(data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path] = None, 
        Calculates the effect size between two groups using Cohen's d or Hedges' g.
    
    plot_distribution(data: Union[pd.DataFrame, pd.Series, np.ndarray, str, Path],
                      column: Optional[str] = None, plot_type: Literal['hist', 'kde', 'box', 'violin', 'all'] = 'hist',
                      bins: int = 30, figsize: Optional[Tuple[int, int]] = None,
                      save_fig: Optional[bool] = False, filename: Optional[str] = None, **kwargs):
        Plots the distribution of a variable using various plot types and backends.
    
    plot_correlation_matrix(data: Union[pd.DataFrame, str, Path],
                            filename: Optional[str] = None, **kwargs):
        Visualizes the correlation matrix using a heatmap.
    
    plot_scatter_matrix(data: Union[pd.DataFrame, str, Path],
                        filename: Optional[str] = None, **kwargs):
        Creates a scatter matrix (pairplot) for visualizing relationships between variables.
    
    plot_distribution_with_ci(data: Union[pd.DataFrame, pd.Series, np.ndarray, str, Path],
                              column: Optional[str] = None, confidence_level: float = 0.95,
                              ci_method: str = 'parametric', bins: int = 30,
                              filename: Optional[str] = None, **kwargs) -> plt.Figure:
        Plots the distribution of a variable with confidence intervals.
    
    get_descriptive_stats(data, column=None) -> dict:
        Returns a dictionary of descriptive statistics for the given data.
    
    help():
        Displays a complete help guide for the UtilsStats class.
    """
    
    
    def __init__(self) -> None:
        """Initialize the utility class."""
        self._plot_backend: str = 'seaborn'
        self._default_figsize: Tuple[int, int] = (12, 5)
        self._save_fig: bool = False
        self._fig_format: str = 'png'
        self._fig_dpi: int = 300
        self._figures_dir: str = 'figures'
        
        plt.style.use('default')
        self._setup_plotting_style()
    
    def _setup_plotting_style(self) -> None:
        """Configure default plotting styles."""
        plt.rcParams['figure.figsize'] = [self._default_figsize[0], self._default_figsize[1]]
        plt.rcParams['figure.dpi'] = self._fig_dpi
        plt.rcParams['savefig.dpi'] = self._fig_dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['lines.linewidth'] = 2
    
    def set_plot_backend(self, backend: Literal['matplotlib', 'seaborn', 'plotly']) -> None:
        """Set the default visualization backend."""
        self._plot_backend = backend
    
    def set_default_figsize(self, figsize: Tuple[int, int]) -> None:
        """Set the default figure size."""
        self._default_figsize = figsize
        plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
    
    def set_save_fig_options(self, save_fig: Optional[bool] = False, 
                            fig_format: str = 'png', 
                            fig_dpi: int = 300,
                            figures_dir: str = 'figures') -> None:
        """Configure figure saving options."""
        self._save_fig = save_fig
        self._fig_format = fig_format
        self._fig_dpi = fig_dpi
        self._figures_dir = figures_dir
    
    def _save_figure(self, fig, filename: str, **kwargs) -> None:
        """Save the figure when save_fig is enabled."""
        if self._save_fig:
            try:
                os.makedirs(self._figures_dir, exist_ok=True)
                filepath = os.path.join(self._figures_dir, f"{filename}.{self._fig_format}")
                
                fig.savefig(
                    filepath, 
                    format=self._fig_format,
                    dpi=self._fig_dpi,
                    bbox_inches='tight',
                    facecolor='white',
                    **kwargs
                )
                logger.info("Figure saved: %s", filepath)

            except Exception as e:
                logger.warning("Error saving figure: %s", e)

    # ============= DATA LOADING =============

    def load_data(self, path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from a file in multiple formats.

        Parameters
        ----------
        path : str or Path
            Path to the data file. Supported formats: CSV, Excel
            (.xlsx/.xls), text/TSV, JSON, Parquet, Feather.
        **kwargs
            Extra arguments forwarded to the pandas reader.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> utils = UtilsStats()
        >>> df = utils.load_data("data.csv")
        >>> df = utils.load_data("data.xlsx", sheet_name="Sheet1")
        """
        logger.info(f"Loading data from path: {path}")
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        
        ext = path.suffix.lower()
        
        try:
            if ext == ".csv":
                df = pd.read_csv(path, **kwargs)
                
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(path, **kwargs)
                
            elif ext in [".txt", ".tsv"]:
                df = pd.read_table(path, **kwargs)
                
            elif ext == ".json":
                df = pd.read_json(path, **kwargs)
                
            elif ext == ".parquet":
                df = pd.read_parquet(path, **kwargs)
                
            elif ext == ".feather":
                df = pd.read_feather(path, **kwargs)
                
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            logger.info("Data loaded from %s (shape=%s)", path, df.shape)
            return df

        except Exception as e:
            raise Exception(f"Error loading file {path}: {str(e)}")

    def _resolve_data(self, data: Union[pd.DataFrame, pd.Series, np.ndarray, list, str, Path],
                        column: Optional[str] = None) -> Tuple[Union[pd.DataFrame, pd.Series, np.ndarray], str]:
        """
        Resuelve el input de datos: si es una ruta, carga el archivo; si no, usa los datos directamente
        
        Returns:
        --------
        Tuple[data, data_source]
            - data: Los datos procesados
            - data_source: String indicando la fuente ('file' o 'memory')
        """
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.exists():
                df = self.load_data(path)
                if column is not None and column in df.columns:
                    return df[column], 'file'
                return df, 'file'
            else:
                raise FileNotFoundError(f"File does not exist: {path}")
        
        return data, 'memory'

    # ============= STATISTICAL ANALYSIS HELPERS =============

    def validate_dataframe(self, data: Union[pd.DataFrame, np.ndarray, list, str, Path]) -> pd.DataFrame:
        """
        Validate and coerce input (arrays, lists, file paths) to a
        pandas DataFrame.
        """
        data, source = self._resolve_data(data)
        
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame({'var': data})
            elif data.ndim == 2:
                return pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
            else:
                raise ValueError("Only 1D and 2D arrays are supported")
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def format_number(self, num: float, decimals: int = 6, scientific: bool = False) -> str:
        """Format a number with the given number of decimals."""
        if scientific and abs(num) < 0.001:
            return f"{num:.{decimals}e}"
        return f"{num:.{decimals}f}"

    def check_normality(self, 
                        data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path], 
                        column: Optional[str] = None,
                        alpha: float = 0.05) -> dict:
        """
        Test whether the data are normally distributed (Shapiro-Wilk).

        Parameters
        ----------
        data : Series, ndarray, DataFrame, str or Path
            Data to analyze, or a path to a data file.
        column : str, optional
            Column to analyze when data is a DataFrame or a file.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        dict
            Keys: is_normal, shapiro_statistic, shapiro_pvalue, alpha,
            interpretation.

        Examples
        --------
        >>> utils.check_normality("data.csv", column="age")
        >>> utils.check_normality(np.random.normal(0, 1, 100))
        """
        logger.info("Checking normality with Shapiro-Wilk test")
        data, source = self._resolve_data(data, column)
        
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("You must specify 'column' when data is a DataFrame")
            data = data[column]
        
        if isinstance(data, pd.Series):
            data_array = data.dropna().values
        else:
            data_array = np.array(data)
            data_array = data_array[~np.isnan(data_array)]
        
        result = _check_normality(data_array, method='shapiro')
        
        is_normal = result['p_value'] > alpha
        interpretation = 'Normal' if is_normal else 'Not normal'
        
        return {
            'is_normal': is_normal,
            'shapiro_statistic': result['statistic'],
            'shapiro_pvalue': result['p_value'],
            'alpha': alpha,
            'interpretation': interpretation
        }

    def calculate_confidence_intervals(self, 
                                        data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path],
                                        column: Optional[str] = None,
                                        confidence_level: float = 0.95,
                                        method: str = 'parametric') -> dict:
        """
        Confidence interval for the mean.

        Parameters
        ----------
        data : Series, ndarray, DataFrame, str or Path
            Data to analyze, or a path to a data file.
        column : str, optional
            Column to analyze.
        confidence_level : float, default 0.95
            Confidence level.
        method : {'parametric', 'bootstrap'}, default 'parametric'
            'parametric' uses the Student t interval; 'bootstrap' uses
            percentile resampling.

        Returns
        -------
        dict
            Keys: mean, std, n, confidence_level, ci_lower, ci_upper,
            margin_error, method.
        """
        logger.info(f"Calculating confidence intervals (method={method}, level={confidence_level})")
        data, source = self._resolve_data(data, column)
        
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("You must specify 'column' when data is a DataFrame")
            data = data[column]
        
        if isinstance(data, pd.Series):
            data_clean = data.dropna().values
        else:
            data_clean = np.array(data)
            data_clean = data_clean[~np.isnan(data_clean)]
        
        n = len(data_clean)
        mean = float(np.mean(data_clean))
        std = float(np.std(data_clean, ddof=1))
        
        ci_method = 'analytic' if method == 'parametric' else 'bootstrap'
        ci_result = _confidence_interval(data_clean, confidence=confidence_level, method=ci_method)
        
        ci_lower = ci_result['lower']
        ci_upper = ci_result['upper']
        margin_error = (ci_upper - ci_lower) / 2
        
        return {
            'mean': mean,
            'std': std,
            'n': n,
            'confidence_level': confidence_level,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'margin_error': margin_error,
            'method': method
        }

    def detect_outliers(self, 
                        data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path],
                        column: Optional[str] = None,
                        method: Literal['iqr', 'zscore', 'isolation_forest'] = 'iqr',
                        **kwargs) -> np.ndarray:
        """
        Detect outliers using different methods.

        Parameters
        ----------
        data : Series, ndarray, DataFrame, str or Path
            Data to analyze, or a path to a data file.
        column : str, optional
            Column to analyze.
        method : {'iqr', 'zscore', 'isolation_forest'}, default 'iqr'
            Detection method ('isolation_forest' requires scikit-learn).
        **kwargs
            'threshold' (iqr/zscore) or 'contamination'
            (isolation_forest).

        Returns
        -------
        numpy.ndarray of bool
            Mask over the NaN-free values; True marks an outlier.
        """
        logger.info(f"Detecting outliers (method={method})")
        data, source = self._resolve_data(data, column)
        
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("You must specify 'column' when data is a DataFrame")
            data = data[column]
        
        if isinstance(data, pd.Series):
            data = data.values
        
        data_clean = data[~np.isnan(data)]
        
        if method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            contamination = kwargs.get('contamination', 0.1)
            X = data_clean.reshape(-1, 1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            outliers = clf.fit_predict(X) == -1
        elif method in ('iqr', 'zscore'):
            threshold = kwargs.get('threshold', 3) if method == 'zscore' else 1.5
            outliers = _detect_outliers(data_clean, method=method, threshold=threshold)
        else:
            raise ValueError("Method must be 'iqr', 'zscore', or 'isolation_forest'")
        
        return outliers

    def calculate_effect_size(self, 
                            data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path] = None,                             
                            group1: Union[str, pd.Series, np.ndarray] = None, 
                            group2: Union[str, pd.Series, np.ndarray] = None,
                            method: Literal['cohen', 'hedges'] = 'cohen') -> dict:
        """
        Effect size between two groups (Cohen's d or Hedges' g), with a
        conventional magnitude interpretation.
        """
        logger.info(f"Calculating effect size (method={method})")

        if isinstance(data, pd.DataFrame):
            group1 = np.array(data[group1])
            group2 = np.array(data[group2])
        elif isinstance(data, (pd.Series, np.ndarray)) and group2 is not None:
            group1 = np.array(data)
            group2 = np.array(group2)
        else:
            group1 = np.array(group1)
            group2 = np.array(group2)
            
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]

        mean1, mean2 = float(np.mean(group1)), float(np.mean(group2))
        std1, std2 = float(np.std(group1, ddof=1)), float(np.std(group2, ddof=1))
        n1, n2 = len(group1), len(group2)

        if method == 'hedges':
            effect_size = _hedges_g(group1, group2)
        else:
            effect_size = _cohens_d(group1, group2)

        pooled_std = float(np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)))

        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            interpretation = "Negligible"
        elif abs_effect < 0.5:
            interpretation = "Small"
        elif abs_effect < 0.8:
            interpretation = "Medium"
        else:
            interpretation = "Large"
        
        return {
            'effect_size': effect_size,
            'method': method,
            'interpretation': interpretation,
            'mean_diff': mean1 - mean2,
            'pooled_std': pooled_std
        }


    # ============= VISUALIZATION =============

    def _plot_distribution_seaborn(self, data, plot_type: str, bins: int, figsize, title: str, **kwargs):
        """Seaborn implementation."""
        if plot_type == 'all':
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            sns.histplot(data, bins=bins, kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Histogram with KDE')
            
            sns.boxplot(y=data, ax=axes[0, 1])
            axes[0, 1].set_title('Box Plot')
            
            sns.violinplot(y=data, ax=axes[1, 0])
            axes[1, 0].set_title('Violin Plot')
            
            stats.probplot(data, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot')
            
            fig.suptitle(title, fontsize=16, y=1.00)
            plt.tight_layout()
            
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'hist':
                sns.histplot(data, bins=bins, kde=True, ax=ax, **kwargs)
            elif plot_type == 'kde':
                sns.kdeplot(data, ax=ax, **kwargs)
            elif plot_type == 'box':
                sns.boxplot(y=data, ax=ax, **kwargs)
            elif plot_type == 'violin':
                sns.violinplot(y=data, ax=ax, **kwargs)
            
            ax.set_title(title)
            plt.tight_layout()
        
        return fig

    def _plot_distribution_matplotlib(self, data, plot_type: str, bins: int, figsize, title: str, **kwargs):
        """Pure matplotlib implementation."""
        if plot_type == 'all':
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            axes[0, 0].hist(data, bins=bins, alpha=0.7, edgecolor='black', density=True)
            axes[0, 0].set_title('Histogram')
            axes[0, 0].set_ylabel('Density')
            
            axes[0, 1].boxplot(data)
            axes[0, 1].set_title('Box Plot')
            
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            axes[1, 0].plot(x_range, kde(x_range))
            axes[1, 0].fill_between(x_range, kde(x_range), alpha=0.3)
            axes[1, 0].set_title('KDE')
            axes[1, 0].set_ylabel('Density')
            
            stats.probplot(data, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot')
            
            fig.suptitle(title, fontsize=16)
            plt.tight_layout()
            
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'hist':
                ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, **kwargs)
                ax.set_ylabel('Frequency')
            elif plot_type == 'box':
                ax.boxplot(data, vert=True)
            elif plot_type == 'kde':
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                ax.plot(x_range, kde(x_range), **kwargs)
                ax.fill_between(x_range, kde(x_range), alpha=0.3)
                ax.set_ylabel('Density')
        
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
        
        return fig
    
    def _plot_distribution_plotly(self, data, plot_type: str, bins: int, title: str, **kwargs):
        """Plotly implementation."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly is not installed. Install with: pip install plotly")
        
        if plot_type == 'all':
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram', 'Box Plot', 'Violin Plot', 'Cumulative distribution')
            )
            
            fig.add_trace(go.Histogram(x=data, nbinsx=bins, name='Histogram'), row=1, col=1)
            
            fig.add_trace(go.Box(y=data, name='Box Plot'), row=1, col=2)
            
            fig.add_trace(go.Violin(y=data, name='Violin Plot'), row=2, col=1)
            
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            cdf = np.cumsum(hist * np.diff(bin_edges))
            fig.add_trace(go.Scatter(x=bin_edges[1:], y=cdf, name='CDF'), row=2, col=2)
            
        else:
            if plot_type == 'hist':
                fig = px.histogram(data, nbins=bins, title=title)
            elif plot_type == 'box':
                fig = px.box(y=data, title=title)
            elif plot_type == 'violin':
                fig = px.violin(y=data, title=title, box=True)
            else:
                fig = px.histogram(data, nbins=bins, title=title)
        
        return fig

    def plot_distribution(self, 
                            data: Union[pd.DataFrame, pd.Series, np.ndarray, str, Path],
                            column: Optional[str] = None,
                            plot_type: Literal['hist', 'kde', 'box', 'violin', 'all'] = 'hist',
                            backend: Optional[Literal['matplotlib', 'seaborn', 'plotly']] = "seaborn",
                            bins: int = 30,
                            figsize: Optional[Tuple[int, int]] = None,
                            save_fig: Optional[bool] = False,
                            filename: Optional[str] = None,
                            **kwargs):
        """
        Plot the distribution of a variable.

        Parameters
        ----------
        data : DataFrame, Series, ndarray, str or Path
            Data to plot, or a path to a data file.
        column : str, optional
            Column to plot when data is a DataFrame or a file.
        plot_type : {'hist', 'kde', 'box', 'violin', 'all'}, default 'hist'
            Plot type ('all' shows a 2x2 diagnostic panel).
        backend : {'matplotlib', 'seaborn', 'plotly'}, default 'seaborn'
            Visualization backend.
        bins : int, default 30
            Histogram bins.
        figsize : tuple, optional
            Figure size.
        save_fig : bool, optional
            Save the figure to disk.
        filename : str, optional
            Output file name.

        Examples
        --------
        >>> utils.plot_distribution("data.csv", column="age")
        >>> utils.plot_distribution(df, column="salary", plot_type="all")
        """
        logger.info(f"Plotting distribution (type={plot_type}, backend={backend})")
        backend = backend or self._plot_backend
        figsize = figsize or self._default_figsize
        self._save_fig = save_fig
        
        data, source = self._resolve_data(data, column)
        
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("You must specify 'column' when data is a DataFrame")
            plot_data = data[column].dropna()
            title = f"Distribution of {column}"
            default_filename = f"distribution_{column}"
        elif isinstance(data, pd.Series):
            plot_data = data.dropna()
            title = f"Distribution of {data.name if data.name else 'Variable'}"
            default_filename = f"distribution_{data.name if data.name else 'variable'}"
        else:
            plot_data = pd.Series(data).dropna()
            title = "Distribution"
            default_filename = "distribution"
        
        filename = filename or default_filename
        
        try:
            if backend == 'seaborn':
                fig = self._plot_distribution_seaborn(plot_data, plot_type, bins, figsize, title, **kwargs)
            elif backend == 'matplotlib':
                fig = self._plot_distribution_matplotlib(plot_data, plot_type, bins, figsize, title, **kwargs)
            elif backend == 'plotly':
                fig = self._plot_distribution_plotly(plot_data, plot_type, bins, title, **kwargs)
            else:
                raise ValueError(f"Backend '{backend}' not supported")
            
            if save_fig and backend != 'plotly':
                self._save_figure(fig, filename)
            
            if backend == 'plotly':
                return fig
            
        except Exception as e:
            print(f"Error en plot_distribution: {e}")
            raise

    def plot_correlation_matrix(self, 
                                data: Union[pd.DataFrame, str, Path],
                                method: Literal['pearson', 'kendall', 'spearman'] = 'pearson',
                                backend: Optional[Literal['seaborn', 'plotly']] = "seaborn",
                                triangular: Optional[bool] = False,
                                figsize: Optional[Tuple[int, int]] = None,
                                save_fig: Optional[bool] = False,
                                filename: Optional[str] = None,
                                **kwargs):
        """
        Plot a correlation matrix heatmap.

        Parameters
        ----------
        data : DataFrame, str or Path
            Data, or a path to a data file.
        method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
            Correlation coefficient.
        backend : {'seaborn', 'plotly'}, default 'seaborn'
            Visualization backend.
        triangular : bool, default False
            Show only the lower triangle (seaborn backend).
        """
        logger.info(f"Plotting correlation matrix (method={method}, backend={backend})")
        backend = backend or self._plot_backend
        figsize = figsize or self._default_figsize
        self.save_fig = save_fig 
        filename = filename or "correlation_matrix"

        data, source = self._resolve_data(data)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("A DataFrame is required to compute the correlation matrix")
        else:
            data = data.select_dtypes(include=['float64', 'int64'])
        
        corr_matrix = data.corr(method=method)
        
        if backend == 'seaborn':
            fig, ax = plt.subplots(figsize=figsize)
            if triangular:
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                            cmap='coolwarm', center=0, ax=ax,
                            square=True, linewidths=0.5, **kwargs)
            else:
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                            cmap='coolwarm', center=0, ax=ax,
                            square=True, linewidths=0.5, **kwargs)
            ax.set_title(f'Correlation matrix ({method})', fontsize=14, pad=20)
            plt.tight_layout()
            
        elif backend == 'plotly':
            import plotly.graph_objects as go
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                **kwargs
            ))
            
            fig.update_layout(
                title=f'Correlation matrix ({method})',
                xaxis_title='Variables',
                yaxis_title='Variables',
                width=figsize[0]*100,
                height=figsize[1]*100
            )
        
        if save_fig:
            if backend == 'seaborn':
                self._save_figure(fig, filename)
            elif backend == 'plotly':
                try:
                    os.makedirs(self._figures_dir, exist_ok=True)
                    filepath = os.path.join(self._figures_dir, f"{filename}.{self._fig_format}")
                    fig.write_image(filepath)
                    logger.info("Plotly figure saved: %s", filepath)
                except Exception as e:
                    logger.warning("Error saving Plotly figure: %s", e)
        if backend == 'plotly':
            return fig

    def plot_scatter_matrix(self, 
                            data: Union[pd.DataFrame, str, Path],
                            columns: Optional[List[str]] = None,
                            backend: Optional[Literal['seaborn', 'plotly', 'pandas']] = None,
                            figsize: Optional[Tuple[int, int]] = None,
                            save_fig: Optional[bool] = False,
                            filename: Optional[str] = None,
                            **kwargs):
        """
        Scatter matrix (pairplot) of the numeric columns.

        Parameters
        ----------
        data : DataFrame, str or Path
            Data, or a path to a data file.
        columns : list of str, optional
            Subset of columns to include.
        backend : {'seaborn', 'plotly', 'pandas'}, optional
            Visualization backend.
        """
        logger.info(f"Plotting scatter matrix (backend={backend})")
        backend = backend or self._plot_backend
        figsize = figsize or self._default_figsize
        self.save_fig = save_fig 
        filename = filename or "scatter_matrix"
        
        data, source = self._resolve_data(data)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("A DataFrame is required for the scatter matrix")
        
        if columns:
            data = data[columns]
        
        if backend == 'seaborn':
            fig = sns.pairplot(data, **kwargs)
            fig.fig.suptitle('Scatter matrix', y=1.02)
            
        elif backend == 'plotly':
            import plotly.express as px
            fig = px.scatter_matrix(data, **kwargs)
            fig.update_layout(title='Scatter matrix')
            
        elif backend == 'pandas':
            from pandas.plotting import scatter_matrix
            fig, ax = plt.subplots(figsize=figsize)
            scatter_matrix(data, ax=ax, **kwargs)
        
        if save_fig:
            if backend in ['seaborn', 'pandas']:
                self._save_figure(fig.figure if hasattr(fig, 'figure') else fig, filename)
            elif backend == 'plotly':
                try:
                    os.makedirs(self._figures_dir, exist_ok=True)
                    filepath = os.path.join(self._figures_dir, f"{filename}.{self._fig_format}")
                    fig.write_image(filepath)
                    logger.info("Plotly figure saved: %s", filepath)
                except Exception as e:
                    logger.warning("Error saving Plotly figure: %s", e)
        
        if backend == 'plotly':
            return fig

    # ============= PLOTS WITH CONFIDENCE INTERVALS =============

    def plot_distribution_with_ci(self,
                                data: Union[pd.DataFrame, pd.Series, np.ndarray, str, Path],
                                column: Optional[str] = None,
                                confidence_level: float = 0.95,
                                ci_method: str = 'parametric',
                                bins: int = 30,
                                figsize: Optional[Tuple[int, int]] = None,
                                save_fig: Optional[bool] = False,
                                filename: Optional[str] = None,
                                **kwargs) -> plt.Figure:
        """
        Plot a distribution with the confidence interval of the mean
        highlighted. Accepts DataFrames, arrays or file paths.
        """
        logger.info(f"Plotting distribution with CI (method={ci_method}, level={confidence_level})")
        data, source = self._resolve_data(data, column)
        
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("You must specify 'column' when data is a DataFrame")
            plot_data = data[column].dropna()
            data_name = column
        elif isinstance(data, pd.Series):
            plot_data = data.dropna()
            data_name = data.name if data.name else 'Variable'
        else:
            plot_data = pd.Series(data).dropna()
            data_name = 'Variable'

        data_array = plot_data.values
        filename = filename or f"distribution_ci_{data_name.lower().replace(' ', '_')}"

        ci_result = self.calculate_confidence_intervals(data_array, confidence_level=confidence_level, method=ci_method)
        normality_result = self.check_normality(data_array)

        kde = stats.gaussian_kde(data_array)
        x_range = np.linspace(data_array.min(), data_array.max(), 300)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize or (14, 6))

        ax1.hist(data_array, bins=bins, density=True,
                color='skyblue', edgecolor='black', alpha=0.7)

        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        ax1.axvline(ci_result['mean'], color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {ci_result['mean']:.2f}")

        ax1.set_title(f"Distribution of {data_name}")
        ax1.set_xlabel("Values")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        ax2.axvspan(ci_result["ci_lower"], ci_result["ci_upper"],
                    color='orange', alpha=0.3,
                    label=f"CI {confidence_level*100:.0f}%")

        ax2.axvline(ci_result["mean"], color='red', linewidth=2)

        if normality_result["is_normal"]:
            normal_y = stats.norm.pdf(x_range, ci_result['mean'], ci_result['std'])
            ax2.plot(x_range, normal_y, 'g--', linewidth=2, alpha=0.7,
                    label="Theoretical normal")

        ax2.set_title(f"CI with method '{ci_method}'")
        ax2.set_xlabel("Values")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(alpha=0.3)

        info = (
            f"Statistics for {data_name}:\n"
            f"- n = {ci_result['n']}\n"
            f"- Mean = {ci_result['mean']:.3f}\n"
            f"- Std dev = {ci_result['std']:.3f}\n"
            f"- CI {confidence_level*100:.0f}% = [{ci_result['ci_lower']:.3f}, {ci_result['ci_upper']:.3f}]\n"
            f"- Margin of error = +/-{ci_result['margin_error']:.3f}\n"
            f"- Normality = {normality_result['interpretation']}\n"
            f"- Shapiro p-value = {normality_result['shapiro_pvalue']:.4f}"
        )

        fig.text(0.01, 0.01, info, fontsize=9,
                bbox=dict(facecolor='lightgray', alpha=0.6),
                va='bottom')

        plt.tight_layout()

        self.save_fig = save_fig
        if save_fig:
            self._save_figure(fig, filename)


    # ============= ADDITIONAL UTILITIES =============

    def get_descriptive_stats(self, data: Union[pd.DataFrame, pd.Series, np.ndarray, list], column: Optional[str] = None, columns: Optional[List[str]] = None) -> dict:
        """
        Descriptive statistics for arbitrary array-like data.

        Prefer :meth:`statslibx.DescriptiveStats.summary` when working
        with DataFrames; this helper exists for quick checks on raw
        arrays.

        Parameters
        ----------
        data : DataFrame, Series, ndarray or list
            Data to analyze.
        column : str, optional
            Column name (when data is a DataFrame).
        columns : list of str, optional
            Multiple columns; returns a dict keyed by column.

        Returns
        -------
        dict
            Descriptive statistics (count, mean, median, mode, std,
            variance, min, max, q1, q3, iqr, skewness, kurtosis, range).
        """
        if columns is not None:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("The 'columns' parameter requires a DataFrame")
            result: Dict[str, dict] = {}
            for col in columns:
                result[col] = self.get_descriptive_stats(data, column=col)
            return result

        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("A column must be specified")
            data_series = data[column]
        else:
            data_series = pd.Series(data)

        data_clean = data_series.dropna()

        if len(data_clean) == 0:
            return {k: np.nan for k in [
                'count','mean','median','mode','std','variance',
                'min','max','q1','q3','iqr','skewness','kurtosis','range'
            ]}

        mode_result = stats.mode(data_clean, keepdims=False)

        return {
            'count': len(data_clean),
            'mean': float(np.mean(data_clean)),
            'median': float(np.median(data_clean)),
            'mode': mode_result.mode,
            'std': float(np.std(data_clean, ddof=1)),
            'variance': float(np.var(data_clean, ddof=1)),
            'min': float(np.min(data_clean)),
            'max': float(np.max(data_clean)),
            'q1': float(np.percentile(data_clean, 25)),
            'q3': float(np.percentile(data_clean, 75)),
            'iqr': float(np.percentile(data_clean, 75) - np.percentile(data_clean, 25)),
            'skewness': float(stats.skew(data_clean)),
            'kurtosis': float(stats.kurtosis(data_clean)),
            'range': float(np.max(data_clean) - np.min(data_clean))
        }
    def help(self) -> None:
        """Print a quick reference of the UtilsStats API."""
        help_text = """
================================================================================
UtilsStats - quick reference
================================================================================
General-purpose utilities: file loading, quick statistical checks and
standalone visualization. All methods accept DataFrames, Series, arrays
or file paths.

Statistical checks:
  .check_normality(data, column=None, alpha=0.05)      Shapiro-Wilk
  .calculate_confidence_intervals(data, confidence_level=0.95,
                                  method='parametric'|'bootstrap')
  .detect_outliers(data, method='iqr'|'zscore'|'isolation_forest')
  .calculate_effect_size(data, group1, group2, method='cohen'|'hedges')
  .get_descriptive_stats(data, column=None)   (prefer DescriptiveStats.summary())

Visualization:
  .plot_distribution(data, column=None, plot_type='hist'|'kde'|'box'|'violin'|'all',
                     backend='matplotlib'|'seaborn'|'plotly')
  .plot_distribution_with_ci(data, column=None, confidence_level=0.95)
  .plot_correlation_matrix(data, method='pearson'|'spearman'|'kendall')
  .plot_scatter_matrix(data, columns=None)

Configuration:
  .set_plot_backend(backend) / .set_default_figsize(figsize)
  .set_save_fig_options(save_fig, fig_format='png', fig_dpi=300, figures_dir='figures')

I/O:
  .load_data(path)             CSV / Excel / JSON / Parquet / Feather / TSV
  .validate_dataframe(data)    Coerce arrays / lists / paths to DataFrame

For full statistical analysis prefer the dedicated classes:
DescriptiveStats, InferentialStats, ComputationalStats.
================================================================================
For details: help(UtilsStats.<method>)
"""
        print(help_text)
