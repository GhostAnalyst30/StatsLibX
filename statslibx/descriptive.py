import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, Union, Literal, List
from datetime import datetime
import flet as ft
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import plotly.express as px

class DescriptiveStats:
    """
    Class for univariate and multivariate descriptive statistics
    """
    
    def __init__(self, data: Union[pd.DataFrame, np.ndarray],
                sep: str = None,
                decimal: str = None,
                thousand: str = None,
                backend: Literal['pandas', 'polars'] = 'pandas'):
        """
        # Initialize DataFrame
        
        ## **Parameters:**

        - **data** : Data to analyze
        - **sep** : Column separator
        - **decimal** : Decimal separator
        - **thousand** : Thousand separator
        - **backend** : 'pandas' or 'polars' for processing
        (Proximamente estara habilitado polars para big data)

        **Examples:**

        ``Example 1:
        stats = DescriptiveStats(data)
        ``
        """

        if isinstance(data, str) and os.path.exists(data):
                data = DescriptiveStats.from_file(data).data

        if isinstance(data, pl.DataFrame):
            raise TypeError(
                "Polars aún no soportado. Use pandas.DataFrame."
            )


        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame({'var': data})
            else:
                data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])]) \
                    if isinstance(data, pd.DataFrame) else pl.DataFrame(data, )
        
        self.data = data
        self.backend = backend
        self._numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.sep = sep
        self.decimal = decimal
        self.thousand = thousand
    
    @classmethod
    def from_file(self, path: str):
        """
        Carga automática de archivos y devuelve instancia de Intelligence.
        Soporta CSV, Excel, TXT, JSON, Parquet, Feather, TSV.
        Automatic file upload and returns Intelligence instance. 
        Supports CSV, Excel, TXT, JSON, Parquet, Feather, TSV.

        Parametros / Parameters:
        ------------------------
        path : str
            Ruta del archivo
            File path
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado / File not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(path, sep=self.sep, decimal=self.decimal, thousand=self.thousand)

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path, decimal=self.decimal, thousand=self.thousand)

        elif ext in [".txt", ".tsv"]:
            df = pd.read_table(path, sep=self.sep, decimal=self.decimal, thousand=self.thousand)

        elif ext == ".json":
            df = pd.read_json(path)

        elif ext == ".parquet":
            df = pd.read_parquet(path)

        elif ext == ".feather":
            df = pd.read_feather(path)

        else:
            raise ValueError(f"Formato no soportado / Unsupported format: {ext}")

        return DescriptiveStats(df)
        
    # ============= MÉTODOS UNIVARIADOS =============
    
    def mean(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Media aritmética / Arithmetic mean
        
        Parametros / Parameters:
        ------------------------
        **column** : str
            Nombre de la columna
            Name of the column
        """
        if column:
            return self.data[column].mean()
        return self.data[self._numeric_cols].mean()
    
    def median(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Mediana / Median
        
        Parametros / Parameters:
        ------------------------
        **column** : str
            Nombre de la columna
            Name of the column
        """
        if column:
            return self.data[column].median()
        return self.data[self._numeric_cols].median()
    
    def mode(self, column: Optional[str] = None):
        """
        Moda / Mode
        
        Parametros / Parameters:
        ------------------------
        column : str
            Nombre de la columna
            Name of the column
        """
        if column:
            return self.data[column].mode()[0]
        return self.data[self._numeric_cols].mode().iloc[0]
    
    def variance(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Varianza / Variance
        
        Parametros / Parameters:
        ------------------------
        column : str
            Nombre de la columna
            Name of the column
        """
        if column:
            return self.data[column].var()
        return self.data[self._numeric_cols].var()
    
    def std(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Desviación estándar / Standard deviation

        Parametros / Parameters:
        ------------------------
        column : str
            Nombre de la columna
            Name of the column
        
        """
        if column:
            return self.data[column].std()
        return self.data[self._numeric_cols].std()
    
    def skewness(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Asimetría / Asymmetry
        
        Parametros / Parameters:
        ------------------------
        column : str
            Nombre de la columna
            Name of the column        
        """
        if column:
            return self.data[column].skew()
        return self.data[self._numeric_cols].skew()
    
    def kurtosis(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Curtosis / Kurtosis
        
        Parametros / Parameters:
        ------------------------
        column : str
            Nombre de la columna
            Name of the column
        """
        if column:
            return self.data[column].kurtosis()
        return self.data[self._numeric_cols].kurtosis()
    
    def quantile(self, q: Union[float, List[float]], column: Optional[str] = None):
        """
        Cuantiles - Percentiles / Quantiles - Percentiles
        
        Parametros / Parameters:
        ------------------------
        q : float / List[float]
            Cuantiles a calcular
            Quantiles to calculate
        column : str
            Nombre de la columna
            Name of the column
        """
        if column:
            return self.data[column].quantile(q)
        return self.data[self._numeric_cols].quantile(q)
    
    def outliers(self, column: str, method: Literal['iqr', 'zscore'] = 'iqr', 
                 threshold: float = 1.5) -> pd.Series:
        """
        Detectar outliers en una columna / Detecting outliers in a column

        
        Parametros / Parameters:
        ------------------------
        column : str
            Nombre de la columna
            Name of the column
        method : str
            'iqr' o 'zscore'
        threshold : float
            1.5 para IQR, 3 para zscore típicamente
            1.5 for IQR, 3 for zscore typically
        """
        col_data = self.data[column]
        
        if method == 'iqr':
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (col_data < lower_bound) | (col_data > upper_bound)
        else:  # zscore
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = z_scores > threshold
        
        return outliers
    
    # ============= MÉTODOS MULTIVARIADOS =============
    
    def correlation(self, method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Matriz de correlación / Correlation matrix
        
        Parametros / Parameters:
        ------------------------
        method : str
            'pearson', 'spearman' o 'kendall'
        columns : list, optional
            Lista de columnas a incluir
            List of columns to include
        """
        data_subset = self.data[columns] if columns else self.data[self._numeric_cols]
        return data_subset.corr(method=method)
    
    def covariance(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Matriz de covarianza
        
        Parametros / Parameters:
        ------------------------
        columns: list, optional
            Lista de columnas a incluir
            List of columns to include
        """
        data_subset = self.data[columns] if columns else self.data[self._numeric_cols]
        return data_subset.cov()
    
    # ============= MÉTODOS DE RESUMEN =============
    
    def summary(self, columns: Optional[List[str]] = None, 
                show_plot: bool = False, 
                plot_backend: str = 'seaborn') -> 'DescriptiveSummary':
        """
        Resumen completo de estadísticas descriptivas / Complete descriptive statistics summary
        
        Parametros / Parameters:
        ------------------------
        columns : list, optional
            Columnas específicas a resumir
            Specific columns to summarize
        show_plot : bool
            Si mostrar gráficos
            If to show graphics
        plot_backend : str
            'seaborn', 'plotly' o 'matplotlib'
        """
        cols = columns if columns else self._numeric_cols
        
        results = {}
        for col in cols:
            col_data = self.data[col]
            results[col] = {
                'count': col_data.count(),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'mode': col_data.mode()[0] if len(col_data.mode()) > 0 else np.nan,
                'std': col_data.std(),
                'variance': col_data.var(),
                'min': col_data.min(),
                'q1': col_data.quantile(0.25),
                'q3': col_data.quantile(0.75),
                'max': col_data.max(),
                'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis(),
            }
        
        return DescriptiveSummary(results, show_plot=show_plot, plot_backend=plot_backend)
    
    # ============= REGRESIÓN LINEAL =============
    
    def linear_regression(self, 
                        X: Union[str, List[str]], 
                        y: str,
                        engine: Literal['statsmodels', 'scikit-learn'] = 'statsmodels',
                        fit_intercept: bool = True,
                        show_plot: bool = False,
                        plot_backend: str = 'seaborn',
                        handle_missing: Literal['drop', 'error', 'warn'] = 'drop') -> tuple:
        """
        Regresión lineal simple o múltiple con opción de mostrar gráfico / Simple or multiple \
            linear regression with option to show graph

        Parametros / Parameters:
        ------------------------
        X: str, list, optional
            Nombre de la variable independiente

        y: str
            Nombre de la variable dependiente

        engine: str
            Motor de la regresion

        fit_intercept: bool
            Intercepto de la regresion

        show_plot: bool
            Visualizar la regresion (recomendable, solo [X,y])

        handle_missing:
            'drop', 'error' o 'warn'
        """
        if isinstance(X, str):
            X = [X]

        # Verificar columnas
        missing_columns = [col for col in [y] + X if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Columnas no encontradas: {missing_columns}")

        # Preparar datos
        regression_data = self.data[[y] + X].copy()
        numeric_cols = regression_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            regression_data[col] = regression_data[col].replace([np.inf, -np.inf], np.nan)

        # Manejo de valores faltantes
        if regression_data.isnull().any().any():
            if handle_missing == 'error':
                raise ValueError("Datos contienen valores faltantes")
            regression_data = regression_data.dropna()

        X_data = regression_data[X].values
        y_data = regression_data[y].values

        # Ajustar modelo
        result = LinearRegressionResult(X_data, y_data, X, y, engine=engine, fit_intercept=fit_intercept)
        result.fit()
        result.show_plot = show_plot
        result.plot_backend = plot_backend
        return result


    
    def help(self, lang="es-Es"):
        """
        Muestra ayuda completa de la clase DescriptiveStats

        Parametros / Parameters:
        ------------------------
        lang: str
            Idioma Usuario: Codigo de Idioma (es-Es) o "Español"
            User Language: Languaje Code (en-Us) or "English"
        """
        if lang in ["en-US", "English", "english"]:
            lang = "en-US"
        else:
            lang = ""

        match lang:
            case "es-ES":
                help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    📊 CLASE DescriptiveStats - AYUDA COMPLETA              ║
╚════════════════════════════════════════════════════════════════════════════╝

📝 DESCRIPCIÓN:
   Clase para análisis estadístico descriptivo univariado y multivariado.
   Proporciona herramientas para análisis exploratorio de datos, medidas de
   tendencia central, dispersión, forma de distribución y regresión lineal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 MÉTODOS PRINCIPALES:

┌────────────────────────────────────────────────────────────────────────────┐
│ 1. 📊 ESTADÍSTICAS UNIVARIADAS                                             │
└────────────────────────────────────────────────────────────────────────────┘

  🔹 Medidas de Tendencia Central:
     • .mean(column=None)              → Media aritmética
     • .median(column=None)            → Mediana (valor central)
     • .mode(column=None)              → Moda (valor más frecuente)

  🔹 Medidas de Dispersión:
     • .std(column=None)               → Desviación estándar
     • .variance(column=None)          → Varianza
     • .quantile(q, column=None)       → Cuantiles/Percentiles

  🔹 Medidas de Forma:
     • .skewness(column=None)          → Asimetría (sesgo)
     • .kurtosis(column=None)          → Curtosis (apuntamiento)

  🔹 Detección de Valores Atípicos:
     • .outliers(column, method='iqr', threshold=1.5)
       Métodos: 'iqr' (rango intercuartílico) o 'zscore' (puntuación z)

┌────────────────────────────────────────────────────────────────────────────┐
│ 2. 🔗 ESTADÍSTICAS MULTIVARIADAS                                           │
└────────────────────────────────────────────────────────────────────────────┘

  • .correlation(method='pearson', columns=None)
    Matriz de correlación entre variables
    Métodos: 'pearson', 'spearman', 'kendall'

  • .covariance(columns=None)
    Matriz de covarianza entre variables

┌────────────────────────────────────────────────────────────────────────────┐
│ 3. 📋 RESUMEN COMPLETO                                                     │
└────────────────────────────────────────────────────────────────────────────┘

  • .summary(columns=None, show_plot=False, plot_backend='seaborn')
    Resumen descriptivo completo con todas las estadísticas
    
    Incluye: conteo, media, mediana, moda, desv. est., varianza,
            mínimo, Q1, Q3, máximo, IQR, asimetría, curtosis
  • .summary().to_dataframe(format)
    Format:
        - Wide
        - Long
        - Compact

  • .summary().to_categorical_summary() 
  • .summary().to_styled_df() 


┌────────────────────────────────────────────────────────────────────────────┐
│ 4. 📈 REGRESIÓN LINEAL                                                     │
└────────────────────────────────────────────────────────────────────────────┘

  • .linear_regression(y, X, engine='statsmodels', 
                      fit_intercept=True, show_plot=False,
                      plot_backend='seaborn', handle_missing='drop')
    
    Regresión lineal simple o múltiple con análisis completo
    
    Parámetros:
      y               : Variable dependiente (str)
      X               : Variable(s) independiente(s) (str o list)
      engine          : 'statsmodels' o 'scikit-learn'
      fit_intercept   : Incluir intercepto (bool)
      show_plot       : Mostrar gráficos diagnósticos (bool)
      handle_missing  : 'drop', 'error', 'warn'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 EJEMPLOS DE USO:

    ┌─ Ejemplo 1: Inicialización ─────────────────────────────────────────────┐
    │ import pandas as pd                                                     │
    │ from descriptive import DescriptiveStats                                │
    │                                                                         │
    │ # Con DataFrame                                                         │
    │ df = pd.read_csv('datos.csv')                                           │
    │ stats = DescriptiveStats(df)                                            │
    │                                                                         │
    │ # Con array numpy                                                       │
    │ import numpy as np                                                      │
    │ datos = np.random.normal(0, 1, 1000)                                    │
    │ stats = DescriptiveStats(datos)                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 2: Análisis Univariado ────────────────────────────────────────┐
    │ # Estadísticas de una columna                                           │
    │ media = stats.mean('edad')                                              │
    │ mediana = stats.median('edad')                                          │
    │ desv_est = stats.std('edad')                                            │
    │                                                                         │
    │ # Cuartiles                                                             │
    │ q25 = stats.quantile(0.25, 'edad')                                      │
    │ q75 = stats.quantile(0.75, 'edad')                                      │
    │                                                                         │
    │ # Detectar outliers                                                     │
    │ outliers_mask = stats.outliers('edad', method='iqr', threshold=1.5)     │
    │ print(f"Outliers detectados: {outliers_mask.sum()}")                    │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 3: Resumen Completo ───────────────────────────────────────────┐
    │ # Resumen de todas las variables numéricas                              │
    │ resumen = stats.summary()                                               │
    │ print(resumen)                                                          │
    │                                                                         │
    │ # Resumen de columnas específicas con visualización                     │
    │ resumen = stats.summary(                                                │
    │     columns=['edad', 'salario', 'experiencia'],                         │
    │     show_plot=True,                                                     │
    │     plot_backend='seaborn'                                              │
    │ )                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 4: Análisis Multivariado ──────────────────────────────────────┐
    │ # Matriz de correlación                                                 │
    │ corr_pearson = stats.correlation(method='pearson')                      │
    │ corr_spearman = stats.correlation(method='spearman')                    │
    │                                                                         │
    │ # Matriz de covarianza                                                  │
    │ cov_matrix = stats.covariance()                                         │
    │                                                                         │
    │ # Correlación entre variables específicas                               │
    │ corr_subset = stats.correlation(                                        │
    │     method='pearson',                                                   │
    │     columns=['edad', 'salario', 'experiencia']                          │
    │ )                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 5: Regresión Lineal Simple ────────────────────────────────────┐
    │ # Regresión simple: salario ~ experiencia                               │
    │ modelo = stats.linear_regression(                                       │
    │     y='salario',                                                        │
    │     X='experiencia',                                                    │
    │     engine='statsmodels',                                               │
    │     show_plot=True                                                      │
    │ )                                                                       │
    │                                                                         │
    │ # Ver resultados                                                        │
    │ print(modelo.summary())                                                 │
    │                                                                         │
    │ # Acceder a coeficientes                                                │
    │ print(f"Intercepto: {modelo.intercept_}")                               │
    │ print(f"Pendiente: {modelo.coef_[0]}")                                  │
    │ print(f"R²: {modelo.r_squared}")                                        │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 6: Regresión Lineal Múltiple ──────────────────────────────────┐
    │ # Regresión múltiple: salario ~ experiencia + edad + educacion          │
    │ modelo = stats.linear_regression(                                       │
    │     y='salario',                                                        │
    │     X=['experiencia', 'edad', 'educacion'],                             │
    │     engine='statsmodels',                                               │
    │     fit_intercept=True,                                                 │
    │     handle_missing='drop'                                               │
    │ )                                                                       │
    │                                                                         │
    │ print(modelo.summary())                                                 │
    │                                                                         │
    │ # Hacer predicciones                                                    │
    │ import numpy as np                                                      │
    │ X_nuevo = np.array([[5, 30, 16], [10, 35, 18]])  # experiencia, edad    │
    │ predicciones = modelo.predict(X_nuevo)                                  │
    └─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 CARACTERÍSTICAS CLAVE:

    ✓ Análisis univariado completo
    ✓ Análisis multivariado (correlación, covarianza)
    ✓ Detección de outliers con múltiples métodos
    ✓ Regresión lineal con statsmodels o scikit-learn
    ✓ Manejo automático de valores faltantes
    ✓ Soporte para pandas DataFrame y numpy arrays
    ✓ Salidas formateadas profesionales
    ✓ Visualizaciones opcionales

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENTACIÓN ADICIONAL:
    Para más información sobre métodos específicos, use:
    help(DescriptiveStats.nombre_metodo)

╚════════════════════════════════════════════════════════════════════════════╝
    """
            case "en-US":
                # --- Falta por traducir
                help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    📊 DescriptiveStats CLASS - COMPLETE HELP               ║
╚════════════════════════════════════════════════════════════════════════════╝

📝 DESCRIPTION:
    Class for univariate and multivariate descriptive statistical analysis. 
    Provides tools for exploratory data analysis, measures of 
    central tendency, dispersion, shape of distribution and linear regression.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 MAIN METHODS:

┌────────────────────────────────────────────────────────────────────────────┐
│ 1. 📊 UNIVARIATE STATISTICS                                                │
└────────────────────────────────────────────────────────────────────────────┘

    🔹 Measures of Central Tendency:
        • .mean(column=None)              → Arithmetic mean
        • .median(column=None)            → Median (center value)
        • .mode(column=None)              → Mode (most frequent value)

    🔹 Dispersion Measurements:
        • .std(column=None)               → Standard deviation
        • .variance(column=None)          → Variance
        • .quantile(q, column=None)       → Quantiles/Percentiles

    🔹 Shape Measurements:
        • .skewness(column=None)          → Asymmetry (bias)
        • .kurtosis(column=None)          → Kurtosis (pointing)

    🔹 Outlier Detection:
        • .outliers(column, method='iqr', threshold=1.5)
        Methods: 'iqr' (interquartile range) or 'zscore' (z-score)

┌────────────────────────────────────────────────────────────────────────────┐
│ 2. 🔗 MULTIVARIATE STATISTICS                                              │
└────────────────────────────────────────────────────────────────────────────┘

    🔹 .correlation(method='pearson', columns=None)
        Correlation matrix between variables
        Methods: 'pearson', 'spearman', 'kendall'

    🔹 .covariance(columns=None)
        Covariance matrix between variables

┌────────────────────────────────────────────────────────────────────────────┐
│ 3. 📋 COMPLETE SUMMARY                                                     │
└────────────────────────────────────────────────────────────────────────────┘

    🔹 .summary(columns=None, show_plot=False, plot_backend='seaborn')
        Complete descriptive summary with all statistics
        
        Includes: count, mean, median, mode, dev. est., variance, 
            minimum, Q1, Q3, maximum, IQR, skewness, kurtosis

    🔹 .summary().to_dataframe(format)
        Format:
            - Wide
            - Long
            - Compact
    🔹 .summary().to_categorical_summary() 
    🔹 .summary().to_styled_df() 


┌────────────────────────────────────────────────────────────────────────────┐
│ 4. 📈 LINEAR REGRESSION                                                    │
└────────────────────────────────────────────────────────────────────────────┘

    🔹 .linear_regression(y, X, engine='statsmodels', 
                        fit_intercept=True, show_plot=False,
                        plot_backend='seaborn', handle_missing='drop')
    
        Simple or multiple linear regression with full analysis
    
        Parameters: 
            X : Independent variable(s) (str or list) 
            y: Dependent variable (str) 
            engine: 'statsmodels' or 'scikit-learn' 
            fit_intercept : Include intercept (bool) 
            show_plot : Show diagnostic plots (bool) 
            handle_missing : 'drop', 'error', 'warn'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 EXAMPLES OF USE:

    ┌─ Example 1: Initialization ─────────────────────────────────────────────┐
    │ import pandas as pd                                                     │
    │ from statslibx.descriptive import DescriptiveStats                      │
    │ from statslibx.datasets import load_dataset                             │
    │                                                                         │
    │ # With DataFrame                                                        │
    │ df = load_dataset('datos.csv')                                          │
    │ stats = DescriptiveStats(df)                                            │
    │                                                                         │
    │ # With array numpy                                                      │
    │ import numpy as np                                                      │
    │ datos = np.random.normal(0, 1, 1000)                                    │
    │ stats = DescriptiveStats(datos)                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Example 2: Univariate Analysis ────────────────────────────────────────┐
    │ # Statistics of a column                                                │
    │ mean = stats.mean('edad')                                               │
    │ median = stats.median('edad')                                           │
    │ desv_est = stats.std('edad')                                            │
    │                                                                         │
    │ # Quartiles                                                             │
    │ q25 = stats.quantile(0.25, 'edad')                                      │
    │ q75 = stats.quantile(0.75, 'edad')                                      │
    │                                                                         │
    │ # To detect outsolves                                                   │
    │ outliers_mask = stats.outliers('edad', method='iqr', threshold=1.5)     │
    │ print(f"Outliers detected: {outliers_mask.sum()}")                      │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Example 3: Complete Summary ───────────────────────────────────────────┐
    │ # Summary of all numerical variables                                    │
    │ summary = stats.summary()                                               │
    │ print(summary)                                                          │
    │                                                                         │
    │ # Resumen de columnas específicas con visualización                     │
    │ resumen = stats.summary(                                                │
    │     columns=['edad', 'salario', 'experiencia'],                         │
    │     show_plot=True,                                                     │
    │     plot_backend='seaborn'                                              │
    │ )                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 4: Análisis Multivariado ──────────────────────────────────────┐
    │ # Matriz de correlación                                                  │
    │ corr_pearson = stats.correlation(method='pearson')                      │
    │ corr_spearman = stats.correlation(method='spearman')                    │
    │                                                                          │
    │ # Matriz de covarianza                                                   │
    │ cov_matrix = stats.covariance()                                         │
    │                                                                          │
    │ # Correlación entre variables específicas                               │
    │ corr_subset = stats.correlation(                                        │
    │     method='pearson',                                                   │
    │     columns=['edad', 'salario', 'experiencia']                          │
    │ )                                                                        │
    └──────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 5: Regresión Lineal Simple ────────────────────────────────────┐
    │ # Regresión simple: salario ~ experiencia                               │
    │ modelo = stats.linear_regression(                                       │
    │     y='salario',                                                        │
    │     X='experiencia',                                                    │
    │     engine='statsmodels',                                               │
    │     show_plot=True                                                      │
    │ )                                                                        │
    │                                                                          │
    │ # Ver resultados                                                         │
    │ print(modelo.summary())                                                  │
    │                                                                          │
    │ # Acceder a coeficientes                                                 │
    │ print(f"Intercepto: {modelo.intercept_}")                               │
    │ print(f"Pendiente: {modelo.coef_[0]}")                                  │
    │ print(f"R²: {modelo.r_squared}")                                        │
    └──────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 6: Regresión Lineal Múltiple ──────────────────────────────────┐
    │ # Regresión múltiple: salario ~ experiencia + edad + educacion          │
    │ modelo = stats.linear_regression(                                       │
    │     y='salario',                                                        │
    │     X=['experiencia', 'edad', 'educacion'],                             │
    │     engine='statsmodels',                                               │
    │     fit_intercept=True,                                                 │
    │     handle_missing='drop'                                               │
    │ )                                                                        │
    │                                                                          │
    │ print(modelo.summary())                                                  │
    │                                                                          │
    │ # Hacer predicciones                                                     │
    │ import numpy as np                                                       │
    │ X_nuevo = np.array([[5, 30, 16], [10, 35, 18]])  # experiencia, edad   │
    │ predicciones = modelo.predict(X_nuevo)                                  │
    └──────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 CARACTERÍSTICAS CLAVE:

    ✓ Análisis univariado completo
    ✓ Análisis multivariado (correlación, covarianza)
    ✓ Detección de outliers con múltiples métodos
    ✓ Regresión lineal con statsmodels o scikit-learn
    ✓ Manejo automático de valores faltantes
    ✓ Soporte para pandas DataFrame y numpy arrays
    ✓ Salidas formateadas profesionales
    ✓ Visualizaciones opcionales

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENTACIÓN ADICIONAL:
    Para más información sobre métodos específicos, use:
    help(DescriptiveStats.nombre_metodo)

╚════════════════════════════════════════════════════════════════════════════╝
    """
        
        print(help_text)

    
    
class DescriptiveSummary:
    """Clase para formatear salida de estadística descriptiva"""
    
    def __init__(self, results: dict, show_plot: bool = False, plot_backend: str = 'seaborn'):
        self.results = results
        self.show_plot = show_plot
        self.plot_backend = plot_backend
        
    def __repr__(self):
        return self._format_output()
    
    def _format_output(self):
        """Formato de tabla organizada para múltiples variables"""
        output = []
        output.append("=" * 100)
        output.append("RESUMEN DE ESTADÍSTICA DESCRIPTIVA".center(100))
        output.append("=" * 100)
        output.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Variables analizadas: {len(self.results)}")
        output.append("-" * 100)
        
        for var_name, stats in self.results.items():
            output.append(f"\n{'VARIABLE: ' + var_name:^100}")
            output.append("-" * 100)
            
            # Tendencia central
            output.append("\nMedidas de Tendencia Central:")
            output.append(f"{'  Conteo':<40} {stats['count']:>20.0f}")
            output.append(f"{'  Media':<40} {stats['mean']:>20.6f}")
            output.append(f"{'  Mediana':<40} {stats['median']:>20.6f}")
            output.append(f"{'  Moda':<40} {stats['mode']:>20.6f}")
            
            # Dispersión
            output.append("\nMedidas de Dispersión:")
            output.append(f"{'  Desviación Estándar':<40} {stats['std']:>20.6f}")
            output.append(f"{'  Varianza':<40} {stats['variance']:>20.6f}")
            output.append(f"{'  Rango Intercuartílico (IQR)':<40} {stats['iqr']:>20.6f}")
            
            # Cuartiles
            output.append("\nCuartiles y Rango:")
            output.append(f"{'  Mínimo':<40} {stats['min']:>20.6f}")
            output.append(f"{'  Primer Cuartil (Q1)':<40} {stats['q1']:>20.6f}")
            output.append(f"{'  Tercer Cuartil (Q3)':<40} {stats['q3']:>20.6f}")
            output.append(f"{'  Máximo':<40} {stats['max']:>20.6f}")
            
            # Forma
            output.append("\nForma de la Distribución:")
            output.append(f"{'  Asimetría (Skewness)':<40} {stats['skewness']:>20.6f}")
            output.append(f"{'  Curtosis (Kurtosis)':<40} {stats['kurtosis']:>20.6f}")
            
            output.append("-" * 100)
        
        output.append("=" * 100)
        return "\n".join(output)
    
    def to_dataframe(self, format='wide'):
        """
        Convierte los resultados a DataFrame.
        
        Parameters:
        -----------
        format : str, default 'wide'
            - 'wide': Variables en columnas, estadísticas en filas
            - 'long': Formato largo (variable, estadística, valor)
            - 'compact': Variables en filas, estadísticas en columnas
        """
        if format == 'wide':
            return self._to_wide_df()
        elif format == 'long':
            return self._to_long_df()
        elif format == 'compact':
            return self._to_compact_df()
        else:
            raise ValueError("format debe ser 'wide', 'long' o 'compact'")
    
    def _to_wide_df(self):
        """
        Formato ancho: Variables en columnas, estadísticas en filas.
        
        Ejemplo:
                        Variable1  Variable2  Variable3
        count              150.0      150.0      150.0
        mean                 5.8        3.1        3.8
        median               5.8        3.0        4.0
        ...
        """
        df = pd.DataFrame(self.results)
        
        # Ordenar índice por categorías
        order = [
            'count', 'mean', 'median', 'mode',  # Tendencia central
            'std', 'variance', 'iqr',            # Dispersión
            'min', 'q1', 'q3', 'max',            # Cuartiles
            'skewness', 'kurtosis'               # Forma
        ]
        
        # Reordenar filas según el orden definido
        df = df.reindex([stat for stat in order if stat in df.index])
        
        return df
    
    def _to_compact_df(self):
        """
        Formato compacto: Variables en filas, estadísticas en columnas.
        
        Ejemplo:
                count   mean  median   mode   std  variance  ...
        Var1    150.0   5.8     5.8    5.0   0.8      0.68  ...
        Var2    150.0   3.1     3.0    3.0   0.4      0.19  ...
        Var3    150.0   3.8     4.0    1.0   1.8      3.11  ...
        """
        df_data = []
        
        for var_name, stats in self.results.items():
            row = {'Variable': var_name}
            row.update(stats)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df = df.set_index('Variable')
        
        # Ordenar columnas por categorías
        order = [
            'count', 'mean', 'median', 'mode',
            'std', 'variance', 'iqr',
            'min', 'q1', 'q3', 'max',
            'skewness', 'kurtosis'
        ]
        
        df = df[[col for col in order if col in df.columns]]
        
        return df
    
    def _to_long_df(self):
        """
        Formato largo: Una fila por cada combinación variable-estadística.
        
        Ejemplo:
            Variable  Estadistica    Valor
        0       Var1        count   150.00
        1       Var1         mean     5.84
        2       Var1       median     5.80
        ...
        """
        data = []
        
        for var_name, stats in self.results.items():
            for stat_name, value in stats.items():
                data.append({
                    'Variable': var_name,
                    'Estadistica': stat_name,
                    'Valor': value
                })
        
        return pd.DataFrame(data)
    
    def to_styled_df(self):
        """
        Devuelve un DataFrame con formato wide y estilo aplicado.
        Útil para notebooks de Jupyter.
        """
        df = self._to_wide_df()
        
        styled = df.style.format("{:.4f}") \
                    .background_gradient(cmap='YlOrRd', axis=1) \
                    .set_caption(f"Estadística Descriptiva - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return styled
    
    def to_categorical_summary(self):
        """
        Crea un resumen organizado por categorías de estadísticas.
        
        Returns:
        --------
        dict of DataFrames
        """
        df_wide = self._to_wide_df()
        
        return {
            'Tendencia Central': df_wide.loc[['count', 'mean', 'median', 'mode']],
            'Dispersión': df_wide.loc[['std', 'variance', 'iqr']],
            'Cuartiles': df_wide.loc[['min', 'q1', 'q3', 'max']],
            'Forma': df_wide.loc[['skewness', 'kurtosis']]
        }
    

import numpy as np
from datetime import datetime


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegressionResult:
    """Clase para resultados de regresión lineal"""

    def __init__(self, X, y, X_names, y_name, engine='statsmodels', fit_intercept=True):
        self.X = X
        self.y = y
        self.X_names = X_names
        self.y_name = y_name
        self.engine = engine
        self.fit_intercept = fit_intercept
        self.model = None
        self.results = None
        self.show_plot = False
        self.plot_backend = 'seaborn'

        # Atributos que se llenarán después del fit
        self.coef_ = None
        self.intercept_ = None
        self.r_squared = None
        self.adj_r_squared = None
        self.f_statistic = None
        self.f_pvalue = None
        self.aic = None
        self.bic = None
        self.residuals = None
        self.predictions = None
        self.std_errors = None
        self.t_values = None
        self.p_values = None

    def fit(self):
        """Ajustar el modelo"""
        if self.engine == 'statsmodels':
            import statsmodels.api as sm
            X = self.X.copy()
            if self.fit_intercept:
                X = sm.add_constant(X)
            self.model = sm.OLS(self.y, X)
            self.results = self.model.fit()

            # Extraer atributos
            if self.fit_intercept:
                self.intercept_ = self.results.params[0]
                self.coef_ = self.results.params[1:]
                self.std_errors = self.results.bse[1:]
                self.t_values = self.results.tvalues[1:]
                self.p_values = self.results.pvalues[1:]
            else:
                self.intercept_ = 0
                self.coef_ = self.results.params
                self.std_errors = self.results.bse
                self.t_values = self.results.tvalues
                self.p_values = self.results.pvalues

            self.r_squared = self.results.rsquared
            self.adj_r_squared = self.results.rsquared_adj
            self.f_statistic = self.results.fvalue
            self.f_pvalue = self.results.f_pvalue
            self.aic = self.results.aic
            self.bic = self.results.bic
            self.residuals = self.results.resid
            self.predictions = self.results.fittedvalues

        else:  # scikit-learn
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(fit_intercept=self.fit_intercept)
            self.model.fit(self.X, self.y)

            self.coef_ = self.model.coef_
            self.intercept_ = self.model.intercept_
            self.predictions = self.model.predict(self.X)
            self.residuals = self.y - self.predictions
            self.r_squared = self.model.score(self.X, self.y)

            # Calcular R^2 ajustado
            n, k = self.X.shape
            self.adj_r_squared = 1 - (1 - self.r_squared) * (n - 1) / (n - k - 1)

        return self

    def predict(self, X_new):
        """Hacer predicciones con nuevos datos"""
        if self.engine == 'statsmodels':
            import statsmodels.api as sm
            if self.fit_intercept:
                X_new = sm.add_constant(X_new)
            return self.results.predict(X_new)
        else:
            return self.model.predict(X_new)

    def summary(self):
        """Mostrar resumen estilo OLS"""
        return self.__repr__()

    def __repr__(self):
        output = []
        output.append("=" * 100)
        output.append("RESULTADOS DE REGRESIÓN LINEAL".center(100))
        output.append("=" * 100)
        output.append(f"Variable Dependiente: {self.y_name}")
        output.append(f"Variables Independientes: {', '.join(self.X_names)}")
        output.append(f"Motor: {self.engine}")
        output.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("-" * 100)

        # Información del modelo
        output.append("\nINFORMACIÓN DEL MODELO:")
        output.append("-" * 100)
        output.append(f"{'Estadístico':<50} {'Valor':>20}")
        output.append("-" * 100)
        output.append(f"{'R-cuadrado':<50} {self.r_squared:>20.6f}")
        output.append(f"{'R-cuadrado Ajustado':<50} {self.adj_r_squared:>20.6f}")

        if self.f_statistic is not None:
            output.append(f"{'Estadístico F':<50} {self.f_statistic:>20.6f}")
            output.append(f"{'Prob (F-estadístico)':<50} {self.f_pvalue:>20.6e}")

        if self.aic is not None:
            output.append(f"{'AIC':<50} {self.aic:>20.6f}")
            output.append(f"{'BIC':<50} {self.bic:>20.6f}")

        # Coeficientes
        output.append("\nCOEFICIENTES:")
        output.append("-" * 100)
        if self.std_errors is not None:
            output.append(f"{'Variable':<20} {'Coef.':>15} {'Std Err':>15} {'t':>15} {'P>|t|':>15}")
            output.append("-" * 100)
            output.append(f"{'const':<20} {self.intercept_:>15.6f} {'-':>15} {'-':>15} {'-':>15}")
            for i, name in enumerate(self.X_names):
                output.append(
                    f"{name:<20} {self.coef_[i]:>15.6f} {self.std_errors[i]:>15.6f} "
                    f"{self.t_values[i]:>15.3f} {self.p_values[i]:>15.6f}"
                )
        else:
            output.append(f"{'Variable':<20} {'Coeficiente':>20}")
            output.append("-" * 100)
            output.append(f"{'const':<20} {self.intercept_:>20.6f}")
            for i, name in enumerate(self.X_names):
                output.append(f"{name:<20} {self.coef_[i]:>20.6f}")

        # Análisis de residuos
        output.append("\nANÁLISIS DE RESIDUOS:")
        output.append("-" * 100)
        output.append(f"{'Estadístico':<50} {'Valor':>20}")
        output.append("-" * 100)
        output.append(f"{'Media de Residuos':<50} {np.mean(self.residuals):>20.6f}")
        output.append(f"{'Desv. Std. de Residuos':<50} {np.std(self.residuals):>20.6f}")
        output.append(f"{'Mínimo Residuo':<50} {np.min(self.residuals):>20.6f}")
        output.append(f"{'Máximo Residuo':<50} {np.max(self.residuals):>20.6f}")
        output.append("=" * 100)

        if self.show_plot:
            self.plot()
            output.append("\n[Gráficos diagnósticos generados]")

        return "\n".join(output)

    def plot(self):
        """Generar gráficos de regresión y residuales"""
        if len(self.X_names) == 1:
            # Scatter + línea de regresión
            df_plot = pd.DataFrame({
                self.X_names[0]: self.X.flatten(),
                self.y_name: self.y,
                'Predicciones': self.predictions
            })
            sns.lmplot(x=self.X_names[0], y=self.y_name, data=df_plot, ci=None)
            plt.title(f"Regresión lineal: {self.y_name} ~ {self.X_names[0]}")
            plt.show()
        else:
            # Para regresión múltiple, solo gráfico residuos vs predicciones
            plt.scatter(self.predictions, self.residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Predicciones")
            plt.ylabel("Residuos")
            plt.title("Residuos vs Predicciones")
            plt.show()
