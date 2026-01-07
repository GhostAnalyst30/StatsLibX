import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Literal, Tuple
import warnings
import os
from scipy import stats
import seaborn as sns
from pathlib import Path


class UtilsStats:
    """
    Clase utilitaria para operaciones estadísticas comunes y visualización
    
    Esta clase proporciona métodos para validación de datos, análisis estadísticos
    básicos y visualización de resultados. Ahora con soporte para leer archivos directamente.
    
    Examples:
    ---------
    >>> utils = UtilsStats()
    >>> # Desde archivo
    >>> data = utils.load_data("datos.csv")
    >>> utils.check_normality(data, column='edad')
    >>> # Desde array
    >>> data = np.random.normal(0, 1, 100)
    >>> utils.check_normality(data)
    >>> utils.plot_distribution(data)
    """
    
    def __init__(self):
        """Inicializar la clase utilitaria"""
        self._plot_backend = 'seaborn'
        self._default_figsize = (12, 5)
        self._save_fig = False
        self._fig_format = 'png'
        self._fig_dpi = 300
        self._figures_dir = 'figures'
        
        # Configuración de estilo para matplotlib
        plt.style.use('default')
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Configurar estilos de plotting por defecto"""
        plt.rcParams['figure.figsize'] = [self._default_figsize[0], self._default_figsize[1]]
        plt.rcParams['figure.dpi'] = self._fig_dpi
        plt.rcParams['savefig.dpi'] = self._fig_dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['lines.linewidth'] = 2
    
    def set_plot_backend(self, backend: Literal['matplotlib', 'seaborn', 'plotly']):
        """Establecer el backend de visualización por defecto"""
        self._plot_backend = backend
    
    def set_default_figsize(self, figsize: Tuple[int, int]):
        """Establecer el tamaño de figura por defecto"""
        self._default_figsize = figsize
        plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
    
    def set_save_fig_options(self, save_fig: Optional[bool] = False, 
                            fig_format: str = 'png', 
                            fig_dpi: int = 300,
                            figures_dir: str = 'figures'):
        """Configurar opciones para guardar figuras"""
        self._save_fig = save_fig
        self._fig_format = fig_format
        self._fig_dpi = fig_dpi
        self._figures_dir = figures_dir
    
    def _save_figure(self, fig, filename: str, **kwargs):
        """Guardar figura si save_fig está activado"""
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
                print(f"✓ Figura guardada: {filepath}")
                
            except Exception as e:
                print(f"✗ Error guardando figura: {e}")

    # ============= NUEVO: MÉTODOS DE CARGA DE DATOS =============

    def load_data(self, path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Carga datos desde archivo en múltiples formatos
        
        Parameters:
        -----------
        path : str o Path
            Ruta al archivo de datos
        **kwargs : dict
            Argumentos adicionales para la función de lectura de pandas
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con los datos cargados
            
        Supported formats:
        ------------------
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - Text/TSV (.txt, .tsv)
        - JSON (.json)
        - Parquet (.parquet)
        - Feather (.feather)
        
        Examples:
        ---------
        >>> utils = UtilsStats()
        >>> df = utils.load_data("datos.csv")
        >>> df = utils.load_data("datos.xlsx", sheet_name="Hoja1")
        >>> df = utils.load_data("datos.json")
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe: {path}")
        
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
                raise ValueError(f"Formato de archivo no soportado: {ext}")
            
            print(f"✓ Datos cargados exitosamente desde: {path}")
            print(f"  Shape: {df.shape}")
            print(f"  Columnas: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error al cargar el archivo {path}: {str(e)}")

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
        # Si es string o Path, intentar cargar como archivo
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.exists():
                df = self.load_data(path)
                if column is not None and column in df.columns:
                    return df[column], 'file'
                return df, 'file'
            else:
                raise FileNotFoundError(f"El archivo no existe: {path}")
        
        # Si ya son datos en memoria, devolverlos tal cual
        return data, 'memory'

    # ============= MÉTODOS DE ANÁLISIS ESTADÍSTICO (ACTUALIZADOS) =============

    def validate_dataframe(self, data: Union[pd.DataFrame, np.ndarray, list, str, Path]) -> pd.DataFrame:
        """
        Valida y convierte datos a DataFrame
        
        Ahora acepta también rutas de archivos
        """
        # Intentar resolver si es un archivo
        data, source = self._resolve_data(data)
        
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame({'var': data})
            elif data.ndim == 2:
                return pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
            else:
                raise ValueError("Solo se soportan arrays 1D y 2D")
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise TypeError(f"Tipo de dato no soportado: {type(data)}")

    def format_number(self, num: float, decimals: int = 6, scientific: bool = False) -> str:
        """Formatea un número con decimales especificados"""
        if scientific and abs(num) < 0.001:
            return f"{num:.{decimals}e}"
        return f"{num:.{decimals}f}"

    def check_normality(self, 
                        data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path], 
                        column: Optional[str] = None,
                        alpha: float = 0.05) -> dict:
        """
        Verifica si los datos siguen distribución normal usando Shapiro-Wilk
        
        Parameters:
        -----------
        data : Series, ndarray, DataFrame, str o Path
            Datos a analizar o ruta al archivo
        column : str, optional
            Columna a analizar (si data es DataFrame o archivo)
        alpha : float
            Nivel de significancia
            
        Examples:
        ---------
        >>> utils.check_normality("datos.csv", column="edad")
        >>> utils.check_normality(np.random.normal(0, 1, 100))
        """
        # Resolver datos
        data, source = self._resolve_data(data, column)
        
        # Extraer array
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Debe especificar 'column' cuando data es DataFrame")
            data = data[column]
        
        if isinstance(data, pd.Series):
            data = data.dropna().values
        else:
            data = np.array(data)
            data = data[~np.isnan(data)]
        
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        return {
            'is_normal': shapiro_p > alpha,
            'shapiro_statistic': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'alpha': alpha,
            'interpretation': 'Normal' if shapiro_p > alpha else 'No Normal'
        }

    def calculate_confidence_intervals(self, 
                                        data: Union[pd.Series, np.ndarray, pd.DataFrame, str, Path],
                                        column: Optional[str] = None,
                                        confidence_level: float = 0.95,
                                        method: str = 'parametric') -> dict:
        """
        Calcula intervalos de confianza para la media
        
        Parameters:
        -----------
        data : Series, ndarray, DataFrame, str o Path
            Datos a analizar o ruta al archivo
        column : str, optional
            Columna a analizar
        confidence_level : float
            Nivel de confianza (default: 0.95)
        method : str
            'parametric' o 'bootstrap'
        """
        # Resolver datos
        data, source = self._resolve_data(data, column)
        
        # Extraer array
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Debe especificar 'column' cuando data es DataFrame")
            data = data[column]
        
        if isinstance(data, pd.Series):
            data_clean = data.dropna().values
        else:
            data_clean = np.array(data)
            data_clean = data_clean[~np.isnan(data_clean)]
        
        n = len(data_clean)
        mean = np.mean(data_clean)
        std = np.std(data_clean, ddof=1)
        
        if method == 'parametric':
            se = std / np.sqrt(n)
            z_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin_error = z_value * se
            
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
            
        elif method == 'bootstrap':
            n_bootstraps = 1000
            bootstrap_means = []
            
            for _ in range(n_bootstraps):
                bootstrap_sample = np.random.choice(data_clean, size=n, replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            alpha = 1 - confidence_level
            ci_lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
            ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
            margin_error = (ci_upper - ci_lower) / 2
            
        else:
            raise ValueError("Método debe ser 'parametric' o 'bootstrap'")
        
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
        Detecta outliers usando diferentes métodos
        
        Parameters:
        -----------
        data : Series, ndarray, DataFrame, str o Path
            Datos a analizar o ruta al archivo
        column : str, optional
            Columna a analizar
        method : str
            'iqr', 'zscore', o 'isolation_forest'
        
        Returns:
        --------
        np.ndarray
            Array booleano indicando outliers
        """
        # Resolver datos
        data, source = self._resolve_data(data, column)
        
        # Extraer array
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Debe especificar 'column' cuando data es DataFrame")
            data = data[column]
        
        if isinstance(data, pd.Series):
            data = data.values
        
        data_clean = data[~np.isnan(data)]
        
        if method == 'iqr':
            q1 = np.percentile(data_clean, 25)
            q3 = np.percentile(data_clean, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (data_clean < lower_bound) | (data_clean > upper_bound)
            
        elif method == 'zscore':
            threshold = kwargs.get('threshold', 3)
            z_scores = np.abs((data_clean - np.mean(data_clean)) / np.std(data_clean))
            outliers = z_scores > threshold
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            contamination = kwargs.get('contamination', 0.1)
            X = data_clean.reshape(-1, 1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            outliers = clf.fit_predict(X) == -1
            
        else:
            raise ValueError("Método debe ser 'iqr', 'zscore', o 'isolation_forest'")
        
        return outliers

    def calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray, 
                                method: Literal['cohen', 'hedges'] = 'cohen') -> dict:
        """
        Calcula el tamaño del efecto entre dos grupos
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std
        
        if method == 'hedges':
            correction = 1 - (3 / (4 * (n1 + n2) - 9))
            effect_size = cohens_d * correction
        else:
            effect_size = cohens_d
        
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            interpretation = "Muy pequeño"
        elif abs_effect < 0.5:
            interpretation = "Pequeño"
        elif abs_effect < 0.8:
            interpretation = "Mediano"
        else:
            interpretation = "Grande"
        
        return {
            'effect_size': effect_size,
            'method': method,
            'interpretation': interpretation,
            'mean_diff': mean1 - mean2,
            'pooled_std': pooled_std
        }

    # ============= MÉTODOS DE VISUALIZACIÓN COMPLETOS =============

    def _plot_distribution_seaborn(self, data, plot_type, bins, figsize, title, **kwargs):
        """Implementación con seaborn"""
        if plot_type == 'all':
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Histograma
            sns.histplot(data, bins=bins, kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Histograma con KDE')
            
            # Box plot
            sns.boxplot(y=data, ax=axes[0, 1])
            axes[0, 1].set_title('Box Plot')
            
            # Violin plot
            sns.violinplot(y=data, ax=axes[1, 0])
            axes[1, 0].set_title('Violin Plot')
            
            # Q-Q plot
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

    def _plot_distribution_matplotlib(self, data, plot_type, bins, figsize, title, **kwargs):
        """Implementación con matplotlib puro"""
        if plot_type == 'all':
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Histograma
            axes[0, 0].hist(data, bins=bins, alpha=0.7, edgecolor='black', density=True)
            axes[0, 0].set_title('Histograma')
            axes[0, 0].set_ylabel('Densidad')
            
            # Box plot
            axes[0, 1].boxplot(data)
            axes[0, 1].set_title('Box Plot')
            
            # KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            axes[1, 0].plot(x_range, kde(x_range))
            axes[1, 0].fill_between(x_range, kde(x_range), alpha=0.3)
            axes[1, 0].set_title('KDE')
            axes[1, 0].set_ylabel('Densidad')
            
            # Q-Q plot
            stats.probplot(data, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot')
            
            fig.suptitle(title, fontsize=16)
            plt.tight_layout()
            
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'hist':
                ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, **kwargs)
                ax.set_ylabel('Frecuencia')
            elif plot_type == 'box':
                ax.boxplot(data, vert=True)
            elif plot_type == 'kde':
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                ax.plot(x_range, kde(x_range), **kwargs)
                ax.fill_between(x_range, kde(x_range), alpha=0.3)
                ax.set_ylabel('Densidad')
        
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
        
        return fig

    def plot_distribution(self, 
                            data: Union[pd.DataFrame, pd.Series, np.ndarray, str, Path],
                            column: Optional[str] = None,
                            plot_type: Literal['hist', 'kde', 'box', 'violin', 'all'] = 'hist',
                            backend: Optional[Literal['matplotlib', 'seaborn', 'plotly']] = "seaborn",
                            bins: int = 30,
                            figsize: Optional[Tuple[int, int]] = None,
                            save_fig: Optional[bool] = None,
                            filename: Optional[str] = None,
                            **kwargs):
        """
        Graficar distribución de una variable
        
        Parameters:
        -----------
        data : DataFrame, Series, ndarray, str o Path
            Datos a graficar o ruta al archivo
        column : str, optional
            Columna a graficar (si data es DataFrame o archivo)
        plot_type : str
            Tipo de gráfico
        backend : str, optional
            Backend de visualización
        bins : int
            Número de bins para histograma
        figsize : tuple, optional
            Tamaño de la figura
        save_fig : bool, optional
            Si guardar la figura
        filename : str, optional
            Nombre del archivo
            
        Examples:
        ---------
        >>> utils.plot_distribution("datos.csv", column="edad")
        >>> utils.plot_distribution(df, column="salario", plot_type="all")
        """
        backend = backend or self._plot_backend
        figsize = figsize or self._default_figsize
        save_fig = save_fig if save_fig is not None else self._save_fig
        
        # Resolver datos
        data, source = self._resolve_data(data, column)
        
        # Extraer datos
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Debe especificar 'column' cuando data es DataFrame")
            plot_data = data[column].dropna()
            title = f"Distribución de {column}"
            default_filename = f"distribucion_{column}"
        elif isinstance(data, pd.Series):
            plot_data = data.dropna()
            title = f"Distribución de {data.name if data.name else 'Variable'}"
            default_filename = f"distribucion_{data.name if data.name else 'variable'}"
        else:
            plot_data = pd.Series(data).dropna()
            title = "Distribución"
            default_filename = "distribucion"
        
        filename = filename or default_filename
        
        try:
            if backend == 'seaborn':
                fig = self._plot_distribution_seaborn(plot_data, plot_type, bins, figsize, title, **kwargs)
            elif backend == 'matplotlib':
                fig = self._plot_distribution_matplotlib(plot_data, plot_type, bins, figsize, title, **kwargs)
            elif backend == 'plotly':
                fig = self._plot_distribution_plotly(plot_data, plot_type, bins, title, **kwargs)
            else:
                raise ValueError(f"Backend '{backend}' no soportado")
            
            # Guardar figura si está activado
            if save_fig and backend != 'plotly':
                self._save_figure(fig, filename)
            
            return fig
            
        except Exception as e:
            print(f"Error en plot_distribution: {e}")
            raise

    def _plot_distribution_plotly(self, data, plot_type, bins, title, **kwargs):
        """Implementación con plotly"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly no está instalado. Instale con: pip install plotly")
        
        if plot_type == 'all':
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histograma', 'Box Plot', 'Violin Plot', 'Distribución Acumulada')
            )
            
            # Histograma
            fig.add_trace(go.Histogram(x=data, nbinsx=bins, name='Histograma'), row=1, col=1)
            
            # Box plot
            fig.add_trace(go.Box(y=data, name='Box Plot'), row=1, col=2)
            
            # Violin plot
            fig.add_trace(go.Violin(y=data, name='Violin Plot'), row=2, col=1)
            
            # Distribución acumulada
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

    def plot_correlation_matrix(self, 
                                data: Union[pd.DataFrame, str, Path],
                                method: str = 'pearson',
                                backend: Optional[Literal['seaborn', 'plotly']] = None,
                                figsize: Optional[Tuple[int, int]] = None,
                                save_fig: Optional[bool] = None,
                                filename: Optional[str] = None,
                                **kwargs):
        """
        Visualizar matriz de correlación
        
        Parameters:
        -----------
        data : DataFrame, str o Path
            Datos para calcular correlación o ruta al archivo
        method : str
            'pearson', 'spearman' o 'kendall'
        backend : str, optional
            Backend de visualización
        """
        backend = backend or self._plot_backend
        figsize = figsize or self._default_figsize
        save_fig = save_fig if save_fig is not None else self._save_fig
        filename = filename or "matriz_correlacion"
        
        # Resolver datos
        data, source = self._resolve_data(data)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Se requiere un DataFrame para calcular matriz de correlación")
        
        # Calcular matriz de correlación
        corr_matrix = data.corr(method=method)
        
        if backend == 'seaborn':
            fig, ax = plt.subplots(figsize=figsize)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                        cmap='coolwarm', center=0, ax=ax,
                        square=True, linewidths=0.5, **kwargs)
            ax.set_title(f'Matriz de Correlación ({method})', fontsize=14, pad=20)
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
                title=f'Matriz de Correlación ({method})',
                xaxis_title='Variables',
                yaxis_title='Variables',
                width=figsize[0]*100,
                height=figsize[1]*100
            )
        
        # Guardar figura
        if save_fig:
            if backend == 'seaborn':
                self._save_figure(fig, filename)
            elif backend == 'plotly':
                try:
                    os.makedirs(self._figures_dir, exist_ok=True)
                    filepath = os.path.join(self._figures_dir, f"{filename}.{self._fig_format}")
                    fig.write_image(filepath)
                    print(f"✓ Figura Plotly guardada: {filepath}")
                except Exception as e:
                    print(f"✗ Error guardando figura Plotly: {e}")
        
        return fig

    def plot_scatter_matrix(self, 
                            data: Union[pd.DataFrame, str, Path],
                            columns: Optional[List[str]] = None,
                            backend: Optional[Literal['seaborn', 'plotly', 'pandas']] = None,
                            figsize: Optional[Tuple[int, int]] = None,
                            save_fig: Optional[bool] = None,
                            filename: Optional[str] = None,
                            **kwargs):
        """
        Matriz de gráficos de dispersión (pairplot)
        
        Parameters:
        -----------
        data : DataFrame, str o Path
            Datos o ruta al archivo
        """
        backend = backend or self._plot_backend
        figsize = figsize or self._default_figsize
        save_fig = save_fig if save_fig is not None else self._save_fig
        filename = filename or "scatter_matrix"
        
        # Resolver datos
        data, source = self._resolve_data(data)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Se requiere un DataFrame para matriz de dispersión")
        
        if columns:
            data = data[columns]
        
        if backend == 'seaborn':
            fig = sns.pairplot(data, **kwargs)
            fig.fig.suptitle('Matriz de Dispersión', y=1.02)
            
        elif backend == 'plotly':
            import plotly.express as px
            fig = px.scatter_matrix(data, **kwargs)
            fig.update_layout(title='Matriz de Dispersión')
            
        elif backend == 'pandas':
            from pandas.plotting import scatter_matrix
            fig, ax = plt.subplots(figsize=figsize)
            scatter_matrix(data, ax=ax, **kwargs)
        
        # Guardar figura
        if save_fig:
            if backend in ['seaborn', 'pandas']:
                self._save_figure(fig.figure if hasattr(fig, 'figure') else fig, filename)
            elif backend == 'plotly':
                try:
                    os.makedirs(self._figures_dir, exist_ok=True)
                    filepath = os.path.join(self._figures_dir, f"{filename}.{self._fig_format}")
                    fig.write_image(filepath)
                    print(f"✓ Figura Plotly guardada: {filepath}")
                except Exception as e:
                    print(f"✗ Error guardando figura Plotly: {e}")
        
        return fig

    # ============= GRÁFICOS CON INTERVALOS DE CONFIANZA =============

    def plot_distribution_with_ci(self,
                                data: Union[pd.DataFrame, pd.Series, np.ndarray, str, Path],
                                column: Optional[str] = None,
                                confidence_level: float = 0.95,
                                ci_method: str = 'parametric',
                                bins: int = 30,
                                figsize: Optional[Tuple[int, int]] = None,
                                save_fig: Optional[bool] = None,
                                filename: Optional[str] = None,
                                **kwargs) -> plt.Figure:
        """
        Distribución con intervalos de confianza
        
        Ahora acepta rutas de archivos
        """
        # Resolver datos
        data, source = self._resolve_data(data, column)
        
        # ======= PREPARACIÓN =======
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Debe especificar 'column' cuando data es DataFrame")
            plot_data = data[column].dropna()
            data_name = column
        elif isinstance(data, pd.Series):
            plot_data = data.dropna()
            data_name = data.name if data.name else 'Variable'
        else:
            plot_data = pd.Series(data).dropna()
            data_name = 'Variable'

        data_array = plot_data.values
        filename = filename or f"distribucion_ci_{data_name.lower().replace(' ', '_')}"

        # Estadísticas
        ci_result = self.calculate_confidence_intervals(data_array, confidence_level=confidence_level, method=ci_method)
        normality_result = self.check_normality(data_array)

        # KDE
        kde = stats.gaussian_kde(data_array)
        x_range = np.linspace(data_array.min(), data_array.max(), 300)

        # ======= FIGURA =======
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (14, 6))

        # ============================================================
        # PANEL 1: HISTOGRAMA + KDE
        # ============================================================
        ax1.hist(data_array, bins=bins, density=True,
                color='skyblue', edgecolor='black', alpha=0.7)

        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        ax1.axvline(ci_result['mean'], color='red', linestyle='--', linewidth=2,
                    label=f"Media: {ci_result['mean']:.2f}")

        ax1.set_title(f"Distribución de {data_name}")
        ax1.set_xlabel("Valores")
        ax1.set_ylabel("Densidad")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ============================================================
        # PANEL 2: KDE + INTERVALO DE CONFIANZA
        # ============================================================

        # KDE pura
        ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        # Intervalo de Confianza
        ax2.axvspan(ci_result["ci_lower"], ci_result["ci_upper"],
                    color='orange', alpha=0.3,
                    label=f"IC {confidence_level*100:.0f}%")

        # Media
        ax2.axvline(ci_result["mean"], color='red', linewidth=2)

        # Distribución normal teórica (si aplica)
        if normality_result["is_normal"]:
            normal_y = stats.norm.pdf(x_range, ci_result['mean'], ci_result['std'])
            ax2.plot(x_range, normal_y, 'g--', linewidth=2, alpha=0.7,
                    label="Normal Teórica")

        ax2.set_title(f"IC con método '{ci_method}'")
        ax2.set_xlabel("Valores")
        ax2.set_ylabel("Densidad")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # ======= CUADRO DE INFO =======
        info = (
            f"Estadísticas de {data_name}:\n"
            f"• n = {ci_result['n']}\n"
            f"• Media = {ci_result['mean']:.3f}\n"
            f"• Desv. Est. = {ci_result['std']:.3f}\n"
            f"• IC {confidence_level*100:.0f}% = [{ci_result['ci_lower']:.3f}, {ci_result['ci_upper']:.3f}]\n"
            f"• Margen Error = ±{ci_result['margin_error']:.3f}\n"
            f"• Normalidad = {normality_result['interpretation']}\n"
            f"• p-value Shapiro = {normality_result['shapiro_pvalue']:.4f}"
        )

        fig.text(0.01, 0.01, info, fontsize=9,
                bbox=dict(facecolor='lightgray', alpha=0.6),
                va='bottom')

        plt.tight_layout()

        # Guardado opcional
        save_fig = save_fig if save_fig is not None else self._save_fig
        if save_fig:
            self._save_figure(fig, filename)

        return fig


    def plot_multiple_distributions_with_ci(self, 
                                            data_dict: dict,
                                            confidence_level: float = 0.95,
                                            figsize: Optional[Tuple[int, int]] = None,
                                            save_fig: Optional[bool] = None,
                                            filename: Optional[str] = None,
                                            **kwargs) -> plt.Figure:
        """
        Grafica múltiples distribuciones con sus intervalos de confianza
        """
        n_distributions = len(data_dict)
        fig, axes = plt.subplots(n_distributions, 2, 
                               figsize=figsize or (14, 5 * n_distributions))
        
        if n_distributions == 1:
            axes = axes.reshape(1, -1)
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_distributions))
        
        for idx, (name, data) in enumerate(data_dict.items()):
            ax1, ax2 = axes[idx]
            
            if isinstance(data, pd.Series):
                data_array = data.dropna().values
            else:
                data_array = np.array(data)
                data_array = data_array[~np.isnan(data_array)]
            
            # Calcular estadísticas
            ci_result = self.calculate_confidence_intervals(data_array, confidence_level=confidence_level)
            
            # Gráfica izquierda: Distribución básica
            ax1.hist(data_array, bins=30, alpha=0.7, color=colors[idx], 
                    edgecolor='black', density=True)
            
            kde = stats.gaussian_kde(data_array)
            x_range = np.linspace(data_array.min(), data_array.max(), 200)
            ax1.plot(x_range, kde(x_range), 'k-', linewidth=2)
            ax1.axvline(ci_result['mean'], color='red', linestyle='--', linewidth=2)
            
            ax1.set_title(f'{name}\nMedia: {ci_result["mean"]:.2f}')
            ax1.grid(True, alpha=0.3)
            
            # Gráfica derecha: Con intervalos de confianza
            ax2.hist(data_array, bins=30, alpha=0.7, color=colors[idx], 
                    edgecolor='black', density=True)
            ax2.plot(x_range, kde(x_range), 'k-', linewidth=2)
            
            ax2.axvline(ci_result['mean'], color='red', linestyle='-', linewidth=3)
            ax2.axvspan(ci_result['ci_lower'], ci_result['ci_upper'], 
                        alpha=0.3, color='orange')
            ax2.axvline(ci_result['ci_lower'], color='orange', linestyle='--', linewidth=2)
            ax2.axvline(ci_result['ci_upper'], color='orange', linestyle='--', linewidth=2)
            
            ax2.set_title(f'{name} con IC {confidence_level*100}%')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar figura si está activado
        save_fig = save_fig if save_fig is not None else self._save_fig
        if save_fig:
            filename = filename or "multiples_distribuciones_ci"
            self._save_figure(fig, filename)
        
        return fig

    # ============= MÉTODOS UTILITARIOS ADICIONALES =============

    def get_descriptive_stats(self, 
                                data: Union[pd.DataFrame, pd.Series, np.ndarray, str, Path],
                                column: Optional[str] = None) -> dict:
        """
        Obtiene estadísticas descriptivas completas
        
        Ahora acepta rutas de archivos
        """
        # Resolver datos
        data, source = self._resolve_data(data, column)
        
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Debe especificar 'column' cuando data es DataFrame")
            data_series = data[column]
        elif isinstance(data, pd.Series):
            data_series = data
        else:
            data_series = pd.Series(data)
        
        data_clean = data_series.dropna()
        
        return {
            'count': len(data_clean),
            'mean': np.mean(data_clean),
            'median': np.median(data_clean),
            'mode': stats.mode(data_clean)[0][0] if len(data_clean) > 0 else np.nan,
            'std': np.std(data_clean, ddof=1),
            'variance': np.var(data_clean, ddof=1),
            'min': np.min(data_clean),
            'max': np.max(data_clean),
            'q1': np.percentile(data_clean, 25),
            'q3': np.percentile(data_clean, 75),
            'iqr': np.percentile(data_clean, 75) - np.percentile(data_clean, 25),
            'skewness': stats.skew(data_clean),
            'kurtosis': stats.kurtosis(data_clean),
            'range': np.max(data_clean) - np.min(data_clean)
        }

    def help(self):
        """
        Muestra ayuda completa de la clase DescriptiveStats
        """
        help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    📊 CLASE UtilsStats - AYUDA COMPLETA                    ║
╚════════════════════════════════════════════════════════════════════════════╝

📝 DESCRIPCIÓN:
   Clase para análisis estadístico descriptivo univariado y multivariado.
   Proporciona herramientas para análisis exploratorio de datos, medidas de
   tendencia central, dispersión, forma de distribución y regresión lineal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 MÉTODOS PRINCIPALES:

┌────────────────────────────────────────────────────────────────────────────┐
│ 1. 📊 ANÁLISIS ESTADÍSTICO                                                 │
└────────────────────────────────────────────────────────────────────────────┘

  • .check_normality(data, alpha=0.05)
    Verifica normalidad usando test Shapiro-Wilk
    Retorna: dict con estadístico, p-value e interpretación

  • .calculate_confidence_intervals(data, confidence_level=0.95, 
                                   method='parametric')
    Calcula intervalos de confianza para la media
    Métodos: 'parametric' o 'bootstrap'

  • .detect_outliers(data, method='iqr', **kwargs)
    Detecta valores atípicos
    Métodos: 'iqr', 'zscore', 'isolation_forest'

  • .calculate_effect_size(group1, group2, method='cohen')
    Calcula tamaño del efecto entre grupos
    Métodos: 'cohen' (Cohen's d) o 'hedges' (Hedges' g)

  • .get_descriptive_stats(data, column=None)
    Estadísticas descriptivas completas en un dict

┌────────────────────────────────────────────────────────────────────────────┐
│ 2. 🎨 VISUALIZACIÓN DE DISTRIBUCIONES                                      │
└────────────────────────────────────────────────────────────────────────────┘

  • .plot_distribution(data, column=None, plot_type='hist', 
                      backend='seaborn', bins=30, figsize=None, 
                      save_fig=None, filename=None)
    
    Grafica distribución de una variable
    
    plot_type: 'hist', 'kde', 'box', 'violin', 'all'
    backend: 'matplotlib', 'seaborn', 'plotly'

  • .plot_distribution_with_ci(data, column=None, confidence_level=0.95,
                               ci_method='parametric', bins=30, figsize=None,
                               save_fig=None, filename=None)
    
    Distribución con intervalos de confianza visualizados

  • .plot_multiple_distributions_with_ci(data_dict, confidence_level=0.95)
    
    Compara múltiples distribuciones con sus IC

┌────────────────────────────────────────────────────────────────────────────┐
│ 3. 🎨 VISUALIZACIÓN MULTIVARIADA                                           │
└────────────────────────────────────────────────────────────────────────────┘

  • .plot_correlation_matrix(data, method='pearson', backend='seaborn',
                            figsize=None, save_fig=None)
    
    Matriz de correlación con heatmap
    Métodos: 'pearson', 'spearman', 'kendall'

  • .plot_scatter_matrix(data, columns=None, backend='seaborn',
                        figsize=None, save_fig=None)
    
    Matriz de gráficos de dispersión (pairplot)
    Backends: 'seaborn', 'plotly', 'pandas'

┌────────────────────────────────────────────────────────────────────────────┐
│ 4. ⚙️  CONFIGURACIÓN                                                       │
└────────────────────────────────────────────────────────────────────────────┘

  • .set_plot_backend(backend)
    Establece backend por defecto: 'matplotlib', 'seaborn', 'plotly'

  • .set_default_figsize(figsize)
    Establece tamaño de figura por defecto: (ancho, alto)

  • .set_save_fig_options(save_fig=False, fig_format='png', 
                         fig_dpi=300, figures_dir='figures')
    
    Configura guardado automático de figuras

┌────────────────────────────────────────────────────────────────────────────┐
│ 5. 🛠️  UTILIDADES                                                          │
└────────────────────────────────────────────────────────────────────────────┘

  • .validate_dataframe(data)
    Valida y convierte datos a DataFrame

  • .format_number(num, decimals=6, scientific=False)
    Formatea números con precisión específica

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 EJEMPLOS DE USO:

  ┌─ Ejemplo 1: Configuración Inicial ──────────────────────────────────────┐
  │ from utils import UtilsStats                                            │
  │ import pandas as pd                                                      │
  │ import numpy as np                                                       │
  │                                                                          │
  │ # Inicializar                                                            │
  │ utils = UtilsStats()                                                    │
  │                                                                          │
  │ # Configurar visualización                                               │
  │ utils.set_plot_backend('seaborn')                                       │
  │ utils.set_default_figsize((12, 6))                                      │
  │                                                                          │
  │ # Configurar guardado automático                                         │
  │ utils.set_save_fig_options(                                             │
  │     save_fig=True,                                                      │
  │     fig_format='png',                                                   │
  │     fig_dpi=300,                                                        │
  │     figures_dir='mis_graficos'                                          │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 2: Análisis de Normalidad ─────────────────────────────────────┐
  │ # Generar datos                                                          │
  │ datos_normales = np.random.normal(0, 1, 1000)                           │
  │ datos_no_normales = np.random.exponential(2, 1000)                      │
  │                                                                          │
  │ # Test de normalidad                                                     │
  │ resultado1 = utils.check_normality(datos_normales)                      │
  │ print(f"Normales: {resultado1['interpretation']}")                      │
  │ print(f"p-value: {resultado1['shapiro_pvalue']:.4f}")                   │
  │                                                                          │
  │ resultado2 = utils.check_normality(datos_no_normales)                   │
  │ print(f"No normales: {resultado2['interpretation']}")                   │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 3: Intervalos de Confianza ────────────────────────────────────┐
  │ # Método paramétrico                                                     │
  │ ci_param = utils.calculate_confidence_intervals(                        │
  │     datos_normales,                                                     │
  │     confidence_level=0.95,                                              │
  │     method='parametric'                                                 │
  │ )                                                                        │
  │                                                                          │
  │ print(f"Media: {ci_param['mean']:.3f}")                                 │
  │ print(f"IC 95%: [{ci_param['ci_lower']:.3f}, "                          │
  │       f"{ci_param['ci_upper']:.3f}]")                                   │
  │                                                                          │
  │ # Método bootstrap (para datos no normales)                              │
  │ ci_boot = utils.calculate_confidence_intervals(                         │
  │     datos_no_normales,                                                  │
  │     confidence_level=0.95,                                              │
  │     method='bootstrap'                                                  │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 4: Detección de Outliers ──────────────────────────────────────┐
  │ # Método IQR (rango intercuartílico)                                     │
  │ datos = np.random.normal(100, 15, 1000)                                 │
  │ datos = np.append(datos, [200, 210, -50])  # Agregar outliers           │
  │                                                                          │
  │ outliers_iqr = utils.detect_outliers(datos, method='iqr')               │
  │ print(f"Outliers IQR: {outliers_iqr.sum()}")                            │
  │                                                                          │
  │ # Método Z-score                                                         │
  │ outliers_z = utils.detect_outliers(                                     │
  │     datos,                                                              │
  │     method='zscore',                                                    │
  │     threshold=3                                                         │
  │ )                                                                        │
  │ print(f"Outliers Z-score: {outliers_z.sum()}")                          │
  │                                                                          │
  │ # Isolation Forest (machine learning)                                    │
  │ outliers_if = utils.detect_outliers(                                    │
  │     datos,                                                              │
  │     method='isolation_forest',                                          │
  │     contamination=0.05                                                  │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 5: Tamaño del Efecto ──────────────────────────────────────────┐
  │ # Comparar dos grupos                                                    │
  │ grupo_control = np.random.normal(100, 15, 100)                          │
  │ grupo_tratamiento = np.random.normal(110, 15, 100)                      │
  │                                                                          │
  │ efecto = utils.calculate_effect_size(                                   │
  │     grupo_control,                                                      │
  │     grupo_tratamiento,                                                  │
  │     method='cohen'                                                      │
  │ )                                                                        │
  │                                                                          │
  │ print(f"Cohen's d: {efecto['effect_size']:.3f}")                        │
  │ print(f"Interpretación: {efecto['interpretation']}")                    │
  │ print(f"Diferencia de medias: {efecto['mean_diff']:.2f}")               │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 6: Gráficos de Distribución ───────────────────────────────────┐
  │ df = pd.DataFrame({                                                      │
  │     'edad': np.random.normal(35, 10, 500),                              │
  │     'salario': np.random.lognormal(10.5, 0.5, 500)                      │
  │ })                                                                       │
  │                                                                          │
  │ # Histograma simple                                                      │
  │ fig1 = utils.plot_distribution(                                         │
  │     df,                                                                 │
  │     column='edad',                                                      │
  │     plot_type='hist',                                                   │
  │     bins=30                                                             │
  │ )                                                                        │
  │                                                                          │
  │ # Panel completo (histograma, box, violin, Q-Q)                          │
  │ fig2 = utils.plot_distribution(                                         │
  │     df,                                                                 │
  │     column='salario',                                                   │
  │     plot_type='all',                                                    │
  │     backend='seaborn'                                                   │
  │ )                                                                        │
  │                                                                          │
  │ # Con Plotly (interactivo)                                               │
  │ fig3 = utils.plot_distribution(                                         │
  │     df,                                                                 │
  │     column='edad',                                                      │
  │     plot_type='violin',                                                 │
  │     backend='plotly'                                                    │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 7: Distribución con Intervalos de Confianza ───────────────────┐
  │ # Visualizar distribución con IC                                         │
  │ fig = utils.plot_distribution_with_ci(                                  │
  │     df,                                                                 │
  │     column='edad',                                                      │
  │     confidence_level=0.95,                                              │
  │     ci_method='parametric',                                             │
  │     bins=30,                                                            │
  │     save_fig=True,                                                      │
  │     filename='edad_con_ic'                                              │
  │ )                                                                        │
  │                                                                          │
  │ # Comparar múltiples distribuciones                                      │
  │ data_dict = {                                                            │
  │     'Grupo A': df['edad'][:200],                                        │
  │     'Grupo B': df['edad'][200:400],                                     │
  │     'Grupo C': df['edad'][400:]                                         │
  │ }                                                                        │
  │                                                                          │
  │ fig = utils.plot_multiple_distributions_with_ci(                        │
  │     data_dict,                                                          │
  │     confidence_level=0.95                                               │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 8: Matriz de Correlación ──────────────────────────────────────┐
  │ # Crear datos correlacionados                                            │
  │ df = pd.DataFrame({                                                      │
  │     'A': np.random.normal(0, 1, 100),                                   │
  │     'B': np.random.normal(0, 1, 100),                                   │
  │     'C': np.random.normal(0, 1, 100)                                    │
  │ })                                                                       │
  │ df['D'] = df['A'] * 0.8 + np.random.normal(0, 0.2, 100)                │
  │                                                                          │
  │ # Matriz de correlación con seaborn                                      │
  │ fig = utils.plot_correlation_matrix(                                    │
  │     df,                                                                 │
  │     method='pearson',                                                   │
  │     backend='seaborn',                                                  │
  │     figsize=(10, 8)                                                     │
  │ )                                                                        │
  │                                                                          │
  │ # Con Plotly (interactiva)                                               │
  │ fig = utils.plot_correlation_matrix(                                    │
  │     df,                                                                 │
  │     method='spearman',                                                  │
  │     backend='plotly'                                                    │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 9: Matriz de Dispersión ───────────────────────────────────────┐
  │ # Pairplot completo                                                      │
  │ fig = utils.plot_scatter_matrix(                                        │
  │     df,                                                                 │
  │     columns=['A', 'B', 'C', 'D'],                                       │
  │     backend='seaborn'                                                   │
  │ )                                                                        │
  │                                                                          │
  │ # Con Plotly                                                             │
  │ fig = utils.plot_scatter_matrix(                                        │
  │     df,                                                                 │
  │     backend='plotly'                                                    │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 10: Estadísticas Descriptivas Completas ───────────────────────┐
  │ # Obtener todas las estadísticas                                         │
  │ stats = utils.get_descriptive_stats(df, column='edad')                  │
  │                                                                          │
  │ print(f"Media: {stats['mean']:.2f}")                                    │
  │ print(f"Mediana: {stats['median']:.2f}")                                │
  │ print(f"Desv. Est.: {stats['std']:.2f}")                                │
  │ print(f"IQR: {stats['iqr']:.2f}")                                       │
  │ print(f"Asimetría: {stats['skewness']:.3f}")                            │
  │ print(f"Curtosis: {stats['kurtosis']:.3f}")                             │
  └──────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 CARACTERÍSTICAS CLAVE:

  ✓ Múltiples backends de visualización (matplotlib, seaborn, plotly)
  ✓ Guardado automático de figuras en alta resolución
  ✓ Análisis estadísticos robustos
  ✓ Detección de outliers con 3 métodos
  ✓ Intervalos de confianza paramétricos y bootstrap
  ✓ Visualizaciones profesionales listas para publicación
  ✓ Manejo automático de valores faltantes
  ✓ Integración perfecta con pandas y numpy
  ✓ Gráficos interactivos con Plotly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 BACKENDS DE VISUALIZACIÓN:

  🔹 Matplotlib:
     • Rápido y ligero
     • Ideal para gráficos simples
     • Mejor para exportar a archivos

  🔹 Seaborn:
     • Gráficos estadísticos elegantes
     • Temas predefinidos atractivos
     • Mejor para análisis exploratorio

  🔹 Plotly:
     • Gráficos interactivos
     • Zoom, pan, hover tooltips
     • Ideal para presentaciones y dashboards

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 CONSEJOS Y MEJORES PRÁCTICAS:

  1. Siempre verificar normalidad antes de usar métodos paramétricos
  2. Usar bootstrap para IC cuando los datos no son normales
  3. Detectar outliers antes de calcular estadísticas
  4. Guardar figuras en alta resolución (300 DPI) para publicaciones
  5. Usar Plotly para presentaciones interactivas
  6. Usar seaborn para análisis exploratorio rápido

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENTACIÓN ADICIONAL:
   Para más información sobre métodos específicos, use:
   help(UtilsStats.nombre_metodo)

╚════════════════════════════════════════════════════════════════════════════╝
    """
        print(help_text)