from dataclasses import dataclass
import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, Union, Literal, List, Dict, Any
from datetime import datetime
from scipy import stats
import os

class InferentialStats:
    """
    Class for inferential statistics (hypothesis tests, confidence intervals, etc.)
    """
    
    def __init__(self, data: Union[pd.DataFrame, np.ndarray],
                backend: Literal['pandas', 'polars'] = 'pandas'):
        """
        Initialize DataFrame

        Parameters:
        -----------
        data : DataFrame o ndarray
            Data to analyze
        backend : str
            'pandas' or 'polars' for processing
        """

        if isinstance(data, str) and os.path.exists(data):
                data = InferentialStats.from_file(data).data

        if isinstance(data, pl.DataFrame):
            raise TypeError(
                "Polars aún no soportado. Use pandas.DataFrame."
            )

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame({'var': data})
            else:
                data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        
        self.data = data
        self.backend = backend
        self._numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    @classmethod
    def from_file(path: str):
        """
        Carga automática de archivos y devuelve instancia de Intelligence.
        Soporta CSV, Excel, TXT, JSON, Parquet, Feather, TSV.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(path)

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)

        elif ext in [".txt", ".tsv"]:
            df = pd.read_table(path)

        elif ext == ".json":
            df = pd.read_json(path)

        elif ext == ".parquet":
            df = pd.read_parquet(path)

        elif ext == ".feather":
            df = pd.read_feather(path)

        else:
            raise ValueError(f"Formato no soportado: {ext}")

        return InferentialStats(df)
    
    # ============= INTERVALOS DE CONFIANZA =============
    
    def confidence_interval(self, column: str, confidence: float = 0.95,
                            statistic: Literal['mean', 'median', 'proportion'] = 'mean') -> tuple:
        """
        Confidence interval for different statistics
        
        Parameters:
        -----------
        column : str
            Column to analyze
        confidence : float
            Confidence level (default 0.95 = 95%)
        statistic : str
            'mean', 'median' o 'proportion'
        
        Returns:
        --------
        tuple : (lower_bound, upper_bound, point_estimate)
        """
        from scipy import stats
        
        data = self.data[column].dropna()
        n = len(data)
        alpha = 1 - confidence
        
        if statistic == 'mean':
            point_est = data.mean()
            se = stats.sem(data)
            margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
            return (point_est - margin, point_est + margin, point_est)
        
        elif statistic == 'median':
            # Bootstrap para mediana
            point_est = data.median()
            n_bootstrap = 10000
            bootstrap_medians = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=n, replace=True)
                bootstrap_medians.append(np.median(sample))
            
            lower = np.percentile(bootstrap_medians, (alpha/2) * 100)
            upper = np.percentile(bootstrap_medians, (1 - alpha/2) * 100)
            return (lower, upper, point_est)
        
        elif statistic == 'proportion':
            # Asume datos binarios (0/1)
            point_est = data.mean()
            se = np.sqrt(point_est * (1 - point_est) / n)
            z_critical = stats.norm.ppf((1 + confidence) / 2)
            margin = z_critical * se
            return (point_est - margin, point_est + margin, point_est)
    
    # ============= PRUEBAS DE HIPÓTESIS =============
    
    def t_test_1sample(self, column: str, popmean: float = None, 
                        popmedian: float = None,
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided') -> 'TestResult':
        """
        One sample t test (for mean or median)
        
        Parameters:
        -----------
        column : str
            Columna a analizar
        popmean : float, optional
            Media poblacional hipotética
        popmedian : float, optional
            Mediana poblacional hipotética (usa signed-rank test)
        alternative : str
            Hipótesis alternativa
        """
        from scipy import stats
        
        data = self.data[column].dropna()
        
        if popmean is not None:
            statistic, pvalue = stats.ttest_1samp(data, popmean, alternative=alternative)
            
            return TestResult(
                test_name='T-Test de Una Muestra (Media)',
                statistic=statistic,
                pvalue=pvalue,
                alternative=alternative,
                params={
                    'popmean': popmean, 
                    'sample_mean': data.mean(), 
                    'n': len(data),
                    'df': len(data) - 1
                }
            )
        
        elif popmedian is not None:
            # Wilcoxon signed-rank test para mediana
            statistic, pvalue = stats.wilcoxon(data - popmedian, alternative=alternative)
            
            return TestResult(
                test_name='Wilcoxon Signed-Rank Test (Mediana)',
                statistic=statistic,
                pvalue=pvalue,
                alternative=alternative,
                params={
                    'popmedian': popmedian,
                    'sample_median': data.median(),
                    'n': len(data)
                }
            )
        
        else:
            raise ValueError("Debe especificar popmean o popmedian")
    
    def t_test_2sample(self, column1: str, column2: str,
                        equal_var: bool = True,
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided') -> 'TestResult':
        """
        Prueba t de dos muestras independientes
        
        Parameters:
        -----------
        column1, column2 : str
            Columnas a comparar
        equal_var : bool
            Asumir varianzas iguales
        alternative : str
            Hipótesis alternativa
        """
        from scipy import stats
        
        data1 = self.data[column1].dropna()
        data2 = self.data[column2].dropna()
        
        statistic, pvalue = stats.ttest_ind(data1, data2, equal_var=equal_var, alternative=alternative)
        
        return TestResult(
            test_name='T-Test de Dos Muestras',
            statistic=statistic,
            pvalue=pvalue,
            alternative=alternative,
            params={
                'mean1': data1.mean(), 'mean2': data2.mean(),
                'std1': data1.std(), 'std2': data2.std(),
                'n1': len(data1), 'n2': len(data2),
                'equal_var': equal_var
            }
        )
    
    def t_test_paired(self, column1: str, column2: str,
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided') -> 'TestResult':
        """
        Prueba t pareada

        Parameters:
        -----------
        column1, column2: 
            Datos a analizar
        alternative:
            "two-sided", "less" o "greater"
        """
        from scipy import stats
        
        data1 = self.data[column1].dropna()
        data2 = self.data[column2].dropna()
        
        statistic, pvalue = stats.ttest_rel(data1, data2, alternative=alternative)
        
        return TestResult(
            test_name='T-Test Pareado',
            statistic=statistic,
            pvalue=pvalue,
            alternative=alternative,
            params={'mean_diff': (data1 - data2).mean(), 'n': len(data1)}
        )
    
    def mann_whitney_test(self, column1: str, column2: str,
                            alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided') -> 'TestResult':
        """
        Prueba de Mann-Whitney U (alternativa no paramétrica al t-test)
        
        Parameters:
        -----------
        column1, column2 : str
            Columnas a comparar
        alternative : str
            Hipótesis alternativa
        """
        from scipy import stats
        
        data1 = self.data[column1].dropna()
        data2 = self.data[column2].dropna()
        
        statistic, pvalue = stats.mannwhitneyu(data1, data2, alternative=alternative)
        
        return TestResult(
            test_name='Mann-Whitney U Test',
            statistic=statistic,
            pvalue=pvalue,
            alternative=alternative,
            params={
                'median1': data1.median(),
                'median2': data2.median(),
                'n1': len(data1),
                'n2': len(data2)
            }
        )
    
    def chi_square_test(self, column1: str, column2: str) -> 'TestResult':
        """
        Prueba Chi-cuadrado de independencia
        
        Parameters:
        -----------
        column1, column2 : str
            Variables categóricas a probar
        """
        from scipy import stats
        
        contingency_table = pd.crosstab(self.data[column1], self.data[column2])
        chi2, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
        
        return TestResult(
            test_name='Prueba Chi-Cuadrado de Independencia',
            statistic=chi2,
            pvalue=pvalue,
            alternative='two-sided',
            params={'dof': dof, 'contingency_table': contingency_table}
        )
    
    def anova_oneway(self, column: str, groups: str) -> 'TestResult':
        """
        ANOVA de un factor
        
        Parameters:
        -----------
        column : str
            Variable dependiente (numérica)
        groups : str
            Variable de agrupación (categórica)
        """
        from scipy import stats
        
        groups_data = [group[column].values for name, group in self.data.groupby(groups)]
        statistic, pvalue = stats.f_oneway(*groups_data)
        
        return TestResult(
            test_name='ANOVA de Un Factor',
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params={
                'groups': len(groups_data),
                'n_total': sum(len(g) for g in groups_data)
            }
        )
    
    def kruskal_wallis_test(self, column: str, groups: str) -> 'TestResult':
        """
        Prueba de Kruskal-Wallis (ANOVA no paramétrico)
        
        Parameters:
        -----------
        column : str
            Variable dependiente (numérica)
        groups : str
            Variable de agrupación (categórica)
        """
        from scipy import stats
        
        groups_data = [group[column].values for name, group in self.data.groupby(groups)]
        statistic, pvalue = stats.kruskal(*groups_data)
        
        return TestResult(
            test_name='Kruskal-Wallis Test',
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params={
                'groups': len(groups_data),
                'n_total': sum(len(g) for g in groups_data)
            }
        )
    
    def normality_test(self, column: str, 
                        method: Literal['shapiro', 'ks', 'anderson', 'jarque_bera', 'all'] = 'shapiro',
                        test_statistic: Literal['mean', 'median', 'mode'] = 'mean') -> Union['TestResult', dict]:
        """
        Prueba de normalidad con múltiples métodos y estadísticos
        
        Parameters:
        -----------
        column : str
            Columna a analizar
        method : str
            'shapiro' (Shapiro-Wilk)
            'ks' (Kolmogorov-Smirnov)
            'anderson' (Anderson-Darling)
            'jarque_bera' (Jarque-Bera)
            'all' (ejecutar todos los tests)
        test_statistic : str
            'mean', 'median' o 'mode' - estadístico para centrar la distribución
        
        Returns:
        --------
        TestResult o dict
            Si method='all', retorna dict con todos los resultados
        """
        from scipy import stats
        
        data = self.data[column].dropna().values
        n = len(data)
        
        # Centrar los datos según el estadístico elegido
        if test_statistic == 'mean':
            loc = np.mean(data)
            scale = np.std(data, ddof=1)
        elif test_statistic == 'median':
            loc = np.median(data)
            # MAD (Median Absolute Deviation) como escala
            scale = np.median(np.abs(data - loc)) * 1.4826
        elif test_statistic == 'mode':
            from scipy.stats import mode as scipy_mode
            mode_result = scipy_mode(data, keepdims=True)
            loc = mode_result.mode[0]
            scale = np.std(data, ddof=1)
        else:
            raise ValueError(f"test_statistic '{test_statistic}' no reconocido")
        
        if method == 'all':
            results = {}
            
            # Shapiro-Wilk
            if n <= 5000:  # Shapiro tiene límite de muestra
                stat_sw, p_sw = stats.shapiro(data)
                results['shapiro'] = TestResult(
                    test_name=f'Shapiro-Wilk ({test_statistic})',
                    statistic=stat_sw,
                    pvalue=p_sw,
                    alternative='two-sided',
                    params={'n': n, 'test_statistic': test_statistic, 'loc': loc, 'scale': scale}
                )
            
            # Kolmogorov-Smirnov
            stat_ks, p_ks = stats.kstest(data, 'norm', args=(loc, scale))
            results['kolmogorov_smirnov'] = TestResult(
                test_name=f'Kolmogorov-Smirnov ({test_statistic})',
                statistic=stat_ks,
                pvalue=p_ks,
                alternative='two-sided',
                params={'n': n, 'test_statistic': test_statistic, 'loc': loc, 'scale': scale}
            )
            
            # Anderson-Darling
            anderson_result = stats.anderson(data, dist='norm')
            results['anderson_darling'] = {
                'test_name': f'Anderson-Darling ({test_statistic})',
                'statistic': anderson_result.statistic,
                'critical_values': anderson_result.critical_values,
                'significance_levels': anderson_result.significance_level,
                'params': {'n': n, 'test_statistic': test_statistic, 'loc': loc, 'scale': scale}
            }
            
            # Jarque-Bera
            stat_jb, p_jb = stats.jarque_bera(data)
            results['jarque_bera'] = TestResult(
                test_name=f'Jarque-Bera ({test_statistic})',
                statistic=stat_jb,
                pvalue=p_jb,
                alternative='two-sided',
                params={
                    'n': n,
                    'test_statistic': test_statistic,
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data)
                }
            )
            
            return results
        
        elif method == 'shapiro':
            if n > 5000:
                raise ValueError("Shapiro-Wilk requiere n <= 5000. Use otro método o 'all'")
            statistic, pvalue = stats.shapiro(data)
            test_name = f'Shapiro-Wilk ({test_statistic})'
            params = {'n': n, 'test_statistic': test_statistic, 'loc': loc, 'scale': scale}
        
        elif method == 'ks':
            statistic, pvalue = stats.kstest(data, 'norm', args=(loc, scale))
            test_name = f'Kolmogorov-Smirnov ({test_statistic})'
            params = {'n': n, 'test_statistic': test_statistic, 'loc': loc, 'scale': scale}
        
        elif method == 'anderson':
            anderson_result = stats.anderson(data, dist='norm')
            return {
                'test_name': f'Anderson-Darling ({test_statistic})',
                'statistic': anderson_result.statistic,
                'critical_values': anderson_result.critical_values,
                'significance_levels': anderson_result.significance_level,
                'params': {'n': n, 'test_statistic': test_statistic, 'loc': loc, 'scale': scale},
                'interpretation': self._interpret_anderson(anderson_result)
            }
        
        elif method == 'jarque_bera':
            statistic, pvalue = stats.jarque_bera(data)
            test_name = f'Jarque-Bera ({test_statistic})'
            params = {
                'n': n,
                'test_statistic': test_statistic,
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        
        else:
            raise ValueError(f"Método '{method}' no reconocido")
        
        return TestResult(
            test_name=test_name,
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params=params
        )
    
    def _interpret_anderson(self, anderson_result):
        """Interpreta resultados de Anderson-Darling"""
        interpretations = []
        for i, (crit_val, sig_level) in enumerate(zip(anderson_result.critical_values, 
                                                    anderson_result.significance_level)):
            if anderson_result.statistic < crit_val:
                interpretations.append(f"No se rechaza normalidad al {sig_level}% de significancia")
            else:
                interpretations.append(f"Se RECHAZA normalidad al {sig_level}% de significancia")
        return interpretations

    def hypothesis_test(
            self,
            method: Literal["mean", "difference_mean", "proportion", "variance"] = "mean",
            column1: str = None,
            column2: str = None,
            alpha: float = 0.05,
            homoscedasticity: Literal["levene", "bartlett", "var_test"] = "levene") -> Dict[str, Any]:
            
        """
        Test de Hipotesis   

        Parameters:
        -----------
        method : str
            'mean', 'difference_mean', 'proportion' o 'variance'
        column1, column2 : str
            Columnas numéricas a comparar
        alpha : float
            Nivel de significancia (default 0.05)
        homoscedasticity : str
            Método de homocedasticidad
            'levene', 'bartlett' o 'var_test' 
        """

        data = self.data

        if column1 is None:
            raise ValueError("Debes especificar 'column1'.")

        x = data[column1].dropna()

        if method in ["difference_mean", "variance"] and column2 is None:
            raise ValueError("Para este método debes pasar 'column2'.")

        y = data[column2].dropna() if column2 else None

        # --- homoscedasticity test ---
        homo_result = None
        if method in ["difference_mean", "variance"]:
            homo_result = self._homoscedasticity_test(x, y, homoscedasticity)

        # --- MAIN HYPOTHESIS TESTS ---
        if method == "mean":
            # One-sample t-test
            t_stat, p_value = stats.ttest_1samp(x, popmean=np.mean(x))
            test_name = "One-sample t-test"

        elif method == "difference_mean":
            # Two-sample t-test
            equal_var = homo_result["equal_var"]
            t_stat, p_value = stats.ttest_ind(x, y, equal_var=equal_var)
            test_name = "Two-sample t-test"

        elif method == "proportion":
            # Proportion test (z-test)
            p_hat = np.mean(x)
            n = len(x)
            z_stat = (p_hat - 0.5) / np.sqrt(0.5 * 0.5 / n)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            t_stat = z_stat
            test_name = "Proportion Z-test"

        elif method == "variance":
            # Classic F-test
            var_x = np.var(x, ddof=1)
            var_y = np.var(y, ddof=1)
            F = var_x / var_y
            dfn = len(x) - 1
            dfd = len(y) - 1

            p_value = 2 * min(stats.f.cdf(F, dfn, dfd), 1 - stats.f.cdf(F, dfn, dfd))
            t_stat = F
            test_name = "Variance F-test"

        return {
            "test": test_name,
            "statistic": t_stat,
            "p_value": p_value,
            "alpha": alpha,
            "reject_H0": p_value < alpha,
            "homoscedasticity_test": homo_result
        }

    def _homoscedasticity_test(
        self,
        x,
        y,
        method: Literal["levene", "bartlett", "var_test"] = "levene") -> Dict[str, Any]:

        if method == "levene":
            stat, p = stats.levene(x, y)
        elif method == "bartlett":
            stat, p = stats.bartlett(x, y)
        elif method == "var_test":
            # R's var.test equivalent: F-test
            var_x = np.var(x, ddof=1)
            var_y = np.var(y, ddof=1)
            F = var_x / var_y
            dfn = len(x) - 1
            dfd = len(y) - 1
            p = 2 * min(stats.f.cdf(F, dfn, dfd), 1 - stats.f.cdf(F, dfn, dfd))
            stat = F
        else:
            raise ValueError("Método de homocedasticidad no válido.")

        return {
            "method": method,
            "statistic": stat,
            "p_value": p,
            "equal_var": p > 0.05   # estándar
        }
    
    def variance_test(self, column1: str, column2: str,
                    method: Literal['levene', 'bartlett', 'var_test'] = 'levene',
                    center: Literal['mean', 'median', 'trimmed'] = 'median'
                    ) -> 'TestResult':
        """
        Prueba de igualdad de varianzas entre dos columnas.

        Parameters:
        -----------
        column1, column2 : str
            Columnas numéricas a comparar
        method : str
            'levene'   -> robusto, recomendado cuando no se asume normalidad
            'bartlett' -> muy sensible a normalidad
            'var_test' -> equivalente a var.test de R (F-test)
        center : str
            Método de centrado para Levene ('mean', 'median', 'trimmed')

        Returns:
        --------
        TestResult
        """
        from scipy import stats

        data1 = self.data[column1].dropna().values
        data2 = self.data[column2].dropna().values

        if method == 'levene':
            statistic, pvalue = stats.levene(data1, data2, center=center)
            test_name = f'Test de Levene (center={center})'
            params = {
                'var1': data1.var(ddof=1),
                'var2': data2.var(ddof=1),
                'n1': len(data1), 'n2': len(data2)
            }

        elif method == 'bartlett':
            statistic, pvalue = stats.bartlett(data1, data2)
            test_name = 'Test de Bartlett'
            params = {
                'var1': data1.var(ddof=1),
                'var2': data2.var(ddof=1),
                'n1': len(data1), 'n2': len(data2)
            }

        elif method == 'var_test':
            # F-test clásico de comparación de varianzas
            var1 = data1.var(ddof=1)
            var2 = data2.var(ddof=1)
            f_stat = var1 / var2
            df1 = len(data1) - 1
            df2 = len(data2) - 1

            # p-valor bilateral
            pvalue = 2 * min(
                stats.f.cdf(f_stat, df1, df2),
                1 - stats.f.cdf(f_stat, df1, df2)
            )

            statistic = f_stat
            test_name = 'F-test de Varianzas (var.test estilo R)'
            params = {
                'var1': var1, 'var2': var2,
                'ratio': f_stat,
                'df1': df1, 'df2': df2
            }

        else:
            raise ValueError(f"Método '{method}' no válido. Usa levene, bartlett o var_test.")

        return TestResult(
            test_name=test_name,
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params=params
        )

    
    def help(self):
        """
        Muestra ayuda completa de la clase InferentialStats
        """
        help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   🔬 CLASE InferentialStats - AYUDA COMPLETA               ║
╚════════════════════════════════════════════════════════════════════════════╝

📝 DESCRIPCIÓN:
   Clase para estadística inferencial: pruebas de hipótesis, intervalos de
   confianza y pruebas de normalidad. Permite realizar inferencias sobre
   poblaciones a partir de muestras de datos.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 MÉTODOS PRINCIPALES:

┌────────────────────────────────────────────────────────────────────────────┐
│ 1. 📊 INTERVALOS DE CONFIANZA                                              │
└────────────────────────────────────────────────────────────────────────────┘

  • .confidence_interval(column, confidence=0.95, statistic='mean')
    
    Calcula intervalos de confianza para diferentes estadísticos
    
    Parámetros:
      column      : Columna a analizar (str)
      confidence  : Nivel de confianza (float, default 0.95 = 95%)
      statistic   : 'mean', 'median' o 'proportion'
    
    Retorna: (lower_bound, upper_bound, point_estimate)

┌────────────────────────────────────────────────────────────────────────────┐
│ 2. 🧪 PRUEBAS DE HIPÓTESIS - UNA MUESTRA                                   │
└────────────────────────────────────────────────────────────────────────────┘

  • .t_test_1sample(column, popmean=None, popmedian=None, 
                   alternative='two-sided')
    
    Prueba t de una muestra (o Wilcoxon para mediana)
    
    Parámetros:
      column      : Columna a analizar
      popmean     : Media poblacional hipotética (para t-test)
      popmedian   : Mediana poblacional hipotética (para Wilcoxon)
      alternative : 'two-sided', 'less', 'greater'

┌────────────────────────────────────────────────────────────────────────────┐
│ 3. 🧪 PRUEBAS DE HIPÓTESIS - DOS MUESTRAS                                  │
└────────────────────────────────────────────────────────────────────────────┘

  🔹 Pruebas Paramétricas:
  
  • .t_test_2sample(column1, column2, equal_var=True, 
                   alternative='two-sided')
    Prueba t de dos muestras independientes
  
  • .t_test_paired(column1, column2, alternative='two-sided')
    Prueba t pareada (muestras dependientes)

  🔹 Pruebas No Paramétricas:
  
  • .mann_whitney_test(column1, column2, alternative='two-sided')
    Alternativa no paramétrica al t-test de dos muestras

  🔹 Pruebas Extras:
  • .hypothesis_test(method='mean', column1=None, column2=None, 
                   alpha=0.05, homoscedasticity='levene')
  • .variance_test(column1, column2, method='levene', center='median')
    

┌────────────────────────────────────────────────────────────────────────────┐
│ 4. 🧪 PRUEBAS PARA MÚLTIPLES GRUPOS                                        │
└────────────────────────────────────────────────────────────────────────────┘

  🔹 Pruebas Paramétricas:
  
  • .anova_oneway(column, groups)
    ANOVA de un factor para comparar múltiples grupos
  
  🔹 Pruebas No Paramétricas:
  
  • .kruskal_wallis_test(column, groups)
    Alternativa no paramétrica a ANOVA

┌────────────────────────────────────────────────────────────────────────────┐
│ 5. 🧪 PRUEBAS PARA VARIABLES CATEGÓRICAS                                   │
└────────────────────────────────────────────────────────────────────────────┘

  • .chi_square_test(column1, column2)
    Prueba Chi-cuadrado de independencia entre variables categóricas

┌────────────────────────────────────────────────────────────────────────────┐
│ 6. 📈 PRUEBAS DE NORMALIDAD                                                │
└────────────────────────────────────────────────────────────────────────────┘

  • .normality_test(column, method='shapiro', test_statistic='mean')
    
    Prueba si los datos siguen una distribución normal
    
    Métodos disponibles:
      'shapiro'      : Shapiro-Wilk (mejor para n ≤ 5000)
      'ks'           : Kolmogorov-Smirnov
      'anderson'     : Anderson-Darling
      'jarque_bera'  : Jarque-Bera (basado en asimetría y curtosis)
      'all'          : Ejecuta todos los tests
    
    test_statistic: 'mean', 'median' o 'mode' para centrar la distribución

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 EJEMPLOS DE USO:

  ┌─ Ejemplo 1: Intervalos de Confianza ────────────────────────────────────┐
  │ from inferential import InferentialStats                                │
  │ import pandas as pd                                                      │
  │                                                                          │
  │ df = pd.read_csv('datos.csv')                                           │
  │ inf_stats = InferentialStats(df)                                        │
  │                                                                          │
  │ # IC para la media (95%)                                                 │
  │ lower, upper, mean = inf_stats.confidence_interval(                     │
  │     'salario',                                                           │
  │     confidence=0.95,                                                    │
  │     statistic='mean'                                                    │
  │ )                                                                        │
  │ print(f"IC 95%: [{lower:.2f}, {upper:.2f}]")                            │
  │                                                                          │
  │ # IC para la mediana (bootstrap)                                         │
  │ lower, upper, median = inf_stats.confidence_interval(                   │
  │     'edad',                                                              │
  │     confidence=0.99,                                                    │
  │     statistic='median'                                                  │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 2: Prueba t de Una Muestra ────────────────────────────────────┐
  │ # H0: μ = 50000 (la media salarial es 50000)                            │
  │ # H1: μ ≠ 50000 (prueba bilateral)                                      │
  │                                                                          │
  │ resultado = inf_stats.t_test_1sample(                                   │
  │     column='salario',                                                   │
  │     popmean=50000,                                                      │
  │     alternative='two-sided'                                             │
  │ )                                                                        │
  │                                                                          │
  │ print(resultado)                                                         │
  │ # Muestra: estadístico t, valor p, interpretación                       │
  │                                                                          │
  │ # Prueba unilateral                                                      │
  │ resultado = inf_stats.t_test_1sample(                                   │
  │     column='salario',                                                   │
  │     popmean=50000,                                                      │
  │     alternative='greater'  # H1: μ > 50000                              │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 3: Comparación de Dos Grupos ──────────────────────────────────┐
  │ # Prueba t independiente                                                 │
  │ resultado = inf_stats.t_test_2sample(                                   │
  │     column1='salario_hombres',                                          │
  │     column2='salario_mujeres',                                          │
  │     equal_var=True,                                                     │
  │     alternative='two-sided'                                             │
  │ )                                                                        │
  │ print(resultado)                                                         │
  │                                                                          │
  │ # Prueba Mann-Whitney (no paramétrica)                                   │
  │ resultado = inf_stats.mann_whitney_test(                                │
  │     column1='salario_grupo_a',                                          │
  │     column2='salario_grupo_b',                                          │
  │     alternative='two-sided'                                             │
  │ )                                                                        │
  │                                                                          │
  │ # Prueba t pareada (mediciones antes/después)                            │
  │ resultado = inf_stats.t_test_paired(                                    │
  │     column1='peso_antes',                                               │
  │     column2='peso_despues',                                             │
  │     alternative='two-sided'                                             │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 4: ANOVA y Kruskal-Wallis ─────────────────────────────────────┐
  │ # ANOVA para comparar múltiples grupos                                   │
  │ resultado = inf_stats.anova_oneway(                                     │
  │     column='rendimiento',                                               │
  │     groups='departamento'                                               │
  │ )                                                                        │
  │ print(resultado)                                                         │
  │                                                                          │
  │ # Kruskal-Wallis (alternativa no paramétrica)                            │
  │ resultado = inf_stats.kruskal_wallis_test(                              │
  │     column='satisfaccion',                                              │
  │     groups='categoria'                                                  │
  │ )                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 5: Chi-Cuadrado ───────────────────────────────────────────────┐
  │ # Probar independencia entre variables categóricas                       │
  │ resultado = inf_stats.chi_square_test(                                  │
  │     column1='genero',                                                   │
  │     column2='preferencia_producto'                                      │
  │ )                                                                        │
  │ print(resultado)                                                         │
  │                                                                          │
  │ # El resultado incluye la tabla de contingencia                          │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Ejemplo 6: Pruebas de Normalidad ──────────────────────────────────────┐
  │ # Shapiro-Wilk (recomendado para n ≤ 5000)                              │
  │ resultado = inf_stats.normality_test(                                   │
  │     column='edad',                                                      │
  │     method='shapiro',                                                   │
  │     test_statistic='mean'                                               │
  │ )                                                                        │
  │ print(resultado)                                                         │
  │                                                                          │
  │ # Kolmogorov-Smirnov                                                     │
  │ resultado = inf_stats.normality_test(                                   │
  │     column='salario',                                                   │
  │     method='ks'                                                         │
  │ )                                                                        │
  │                                                                          │
  │ # Ejecutar todos los tests                                               │
  │ resultados = inf_stats.normality_test(                                  │
  │     column='ingresos',                                                  │
  │     method='all',                                                       │
  │     test_statistic='median'                                             │
  │ )                                                                        │
  │                                                                          │
  │ # Acceder a cada test                                                    │
  │ print(resultados['shapiro'])                                            │
  │ print(resultados['kolmogorov_smirnov'])                                 │
  │ print(resultados['anderson_darling'])                                   │
  │ print(resultados['jarque_bera'])                                        │
  └──────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 GUÍA DE SELECCIÓN DE PRUEBAS:

  ┌─ Comparar Una Muestra vs Valor de Referencia ───────────────────────────┐
  │ Datos normales        → t_test_1sample (con popmean)                    │
  │ Datos no normales     → t_test_1sample (con popmedian, usa Wilcoxon)   │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Comparar Dos Grupos Independientes ────────────────────────────────────┐
  │ Datos normales        → t_test_2sample                                  │
  │ Datos no normales     → mann_whitney_test                               │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Comparar Dos Grupos Pareados ──────────────────────────────────────────┐
  │ Datos normales        → t_test_paired                                   │
  │ Datos no normales     → (use scipy.stats.wilcoxon directamente)        │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Comparar Múltiples Grupos ─────────────────────────────────────────────┐
  │ Datos normales        → anova_oneway                                    │
  │ Datos no normales     → kruskal_wallis_test                             │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ Probar Independencia entre Categóricas ────────────────────────────────┐
  │ Variables categóricas → chi_square_test                                 │
  └──────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 CARACTERÍSTICAS CLAVE:

  ✓ Pruebas paramétricas y no paramétricas
  ✓ Intervalos de confianza con múltiples métodos
  ✓ Pruebas de normalidad completas
  ✓ Interpretación automática de resultados
  ✓ Manejo automático de valores faltantes
  ✓ Salidas formateadas profesionales
  ✓ Soporte para análisis bilateral y unilateral

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  INTERPRETACIÓN DE RESULTADOS:

  • Valor p < 0.05: Se rechaza H0 (evidencia significativa)
  • Valor p ≥ 0.05: No se rechaza H0 (evidencia insuficiente)
  • IC que no incluye el valor nulo: Evidencia contra H0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENTACIÓN ADICIONAL:
   Para más información sobre métodos específicos, use:
   help(InferentialStats.nombre_metodo)

╚════════════════════════════════════════════════════════════════════════════╝
    """
        print(help_text)

@dataclass
class TestResult:
    """Clase para resultados de pruebas de hipótesis"""
    
    def __init__(self, test_name: str, statistic: float, pvalue: float, 
                 alternative: str, params: dict):
        self.test_name = test_name
        self.statistic = statistic
        self.pvalue = pvalue
        self.alternative = alternative
        self.params = params
        
    def __repr__(self):
        return self._format_output()
    
    def _format_output(self):
        """Formato de salida para pruebas de hipótesis"""
        output = []
        output.append("=" * 80)
        output.append(self.test_name.center(80))
        output.append("=" * 80)
        output.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Hipótesis Alternativa: {self.alternative}")
        output.append("-" * 80)
        
        output.append("\nRESULTADOS:")
        output.append("-" * 80)
        output.append(f"{'Estadístico':<40} {self.statistic:>20.6f}")
        output.append(f"{'Valor p':<40} {self.pvalue:>20.6e}")
        
        # Interpretación
        alpha = 0.05
        if self.pvalue < alpha:
            interpretation = "❌ Se RECHAZA la hipótesis nula"
        else:
            interpretation = "✔️ No hay evidencia suficiente para rechazar la hipótesis nula"
        
        output.append("\nINTERPRETACIÓN:")
        output.append("-" * 80)
        output.append(f"Alpha = {alpha}")
        output.append(interpretation)
        
        output.append("\nPARÁMETROS:")
        output.append("-" * 80)
        for k, v in self.params.items():
            output.append(f"{k:<40} {str(v):>20}")
        
        output.append("=" * 80)
        return "\n".join(output)
