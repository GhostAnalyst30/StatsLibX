from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Union, Literal, Dict, Any, Tuple
from datetime import datetime
from scipy import stats

class InferentialStats:
    """    
    InferentialStats
    A class for performing inferential statistical analysis, including hypothesis tests, confidence intervals, 
    normality tests, and more. This class supports operations on pandas DataFrame or numpy arrays.
    Attributes:
    -----------
    data : pd.DataFrame
        The dataset to analyze.
        The backend used for processing ('pandas' or 'polars').
    sep : str
        Separator for reading files.
    decimal : str
        Decimal separator for reading files.
    thousand : str
        Thousand separator for reading files.
    lang : str
        Language for help and error messages ('es-ES' or 'en-US').
    
    Methods:
    --------
    from_file(path: str):
        Load data from a file and return an instance of InferentialStats.
    
    confidence_interval(column: str, confidence: float = 0.95, statistic: Literal['mean', 'median', 'proportion'] = 'mean') -> tuple:
        Calculate confidence intervals for mean, median, or proportion.
    
    t_test_1sample(column: str, popmean: float = None, popmedian: float = None, alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> 'TestResult':
        Perform a one-sample t-test or Wilcoxon signed-rank test for median.
    
    t_test_2sample(column1: str, column2: str, equal_var: bool = True, alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> 'TestResult':
        Perform a two-sample independent t-test.
    
    t_test_paired(column1: str, column2: str, alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> 'TestResult':
        Perform a paired t-test for dependent samples.
    
    mann_whitney_test(column1: str, column2: str, alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> 'TestResult':
        Perform the Mann-Whitney U test, a non-parametric alternative to the two-sample t-test.
    
    chi_square_test(column1: str, column2: str, alpha: float = 0.05) -> 'TestResult':
        Perform a Chi-square test of independence between two categorical variables.
    
    anova_oneway(column: str, groups: str, alpha: float = 0.05) -> 'TestResult':
        Perform a one-way ANOVA test to compare means across multiple groups.
    
    kruskal_wallis_test(column: str, groups: str, alpha: float = 0.05) -> 'TestResult':
        Perform the Kruskal-Wallis test, a non-parametric alternative to one-way ANOVA.
    
    normality_test(column: str, method: Literal['shapiro', 'ks', 'anderson', 'jarque_bera', 'all'] = 'shapiro', test_statistic: Literal['mean', 'median', 'mode'] = 'mean', alpha: float = 0.05) -> Union['TestResult', dict]:
        Perform normality tests using various methods.
    
    hypothesis_test(method: Literal["mean", "difference_mean", "proportion", "variance"] = "mean", column1: str = None, column2: str = None, pop_mean: float = None, pop_proportion: Union[float, Tuple[float, float]] = 0.5, alpha: float = 0.05, homoscedasticity: Literal["levene", "bartlett", "var_test"] = "levene") -> Dict[str, Any]:
        Perform hypothesis testing for mean, difference of means, proportion, or variance.
    
    variance_test(column1: str, column2: str, method: Literal['levene', 'bartlett', 'var_test'] = 'levene', center: Literal['mean', 'median', 'trimmed'] = 'median', alpha: float = 0.05) -> 'TestResult':
        Perform a test for equality of variances between two columns.
    
    help():
        Display a detailed help guide for the InferentialStats class and its methods.
    """
    
    def __init__(self, data: Union[pd.DataFrame, np.ndarray],
                lang: Literal['es-ES', 'en-US'] = 'es-ES'):
        """
        Initialize DataFrame

        Parameters:
        -----------
        data : DataFrame o ndarray
            Data to analyze
        """

        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        else:
            raise TypeError(
                "Data must be a pandas.DataFrame or numpy.ndarray."
            )   

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame({'var': data})
            else:
                data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        
        self._numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.data.select_dtypes(include=[np.object]).columns.tolist()
        self.lang = lang

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
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
                        alpha: float = 0.05) -> 'TestResult':
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
                },
                alpha=alpha
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
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> 'TestResult':
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
            },
            alpha=alpha
        )
    
    def t_test_paired(self, column1: str, column2: str,
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> 'TestResult':
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
            params={'mean_diff': (data1 - data2).mean(), 'n': len(data1)},
            alpha=alpha
        )
    
    def mann_whitney_test(self, column1: str, column2: str,
                            alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> 'TestResult':
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
            },
            alpha=alpha
        )
    
    def chi_square_test(self, column1: str, column2: str,
                        alpha: float = 0.05) -> 'TestResult':
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
            params={'dof': dof, 'contingency_table': contingency_table},
            alpha=alpha
        )
    
    def anova_oneway(self, column: str, groups: str,
                        alpha: float = 0.05) -> 'TestResult':
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
        clean_data = self.data[[column, groups]].dropna()
        
        groups_data = [group[column].values
                            for _, group in clean_data.groupby(groups)
                            if len(group) > 1 and group[column].var() > 0
                    ]

        statistic, pvalue = stats.f_oneway(*groups_data)
        
        return TestResult(  
            test_name='ANOVA de Un Factor',
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params={
                'groups': len(groups_data),
                'n_total': sum(len(g) for g in groups_data)
            },
            alpha=alpha
        )
    
    def kruskal_wallis_test(self, column: str, groups: str,
                            alpha: float = 0.05) -> 'TestResult':
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

        clean_data = self.data[[column, groups]].dropna()
        
        groups_data = [group[column].values
                            for _, group in clean_data.groupby(groups)
                            if len(group) > 1 and group[column].var() > 0
                    ]
        statistic, pvalue = stats.kruskal(*groups_data)
        
        return TestResult(
            test_name='Kruskal-Wallis Test',
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params={
                'groups': len(groups_data),
                'n_total': sum(len(g) for g in groups_data)
            },
            alpha=alpha
        )
    
    def normality_test(self, column: str, 
                        method: Literal['shapiro', 'ks', 'anderson', 'jarque_bera', 'all'] = 'shapiro',
                        test_statistic: Literal['mean', 'median', 'mode'] = 'mean',
                        alpha: float = 0.05) -> Union['TestResult', dict]:
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

        critical_values = None
        significance_levels = None
        
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
            results['anderson_darling'] = TestResult(
                test_name=f'Anderson-Darling ({test_statistic})',
                statistic=anderson_result.statistic,
                critical_values=anderson_result.critical_values,
                significance_levels=anderson_result.significance_level,
                params={'n': n, 'test_statistic': test_statistic, 'loc': loc, 'scale': scale}
            )
            
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
            test_name = f'Anderson-Darling ({test_statistic})'
            pvalue = None
            statistic = anderson_result.statistic
            critical_values = anderson_result.critical_values
            significance_levels = anderson_result.significance_level
            params = {'n': n, 'test_statistic': test_statistic, 'loc': loc, 'scale': scale}
        
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
            params=params,
            critical_values=critical_values,
            significance_levels=significance_levels,
            alpha=alpha
        )

    def hypothesis_test(
            self,
            method: Literal["mean", "difference_mean", "proportion", "variance"] = "mean",
            column1: str = None,
            column2: str = None,
            pop_mean: float = None,
            pop_proportion: Union[float, Tuple[float, float]] = 0.5,
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
        pop_mean : float
            Media poblacional
        pop_proportion : float
            Proporción poblacional (default 0.5)
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
            t_stat, p_value = stats.ttest_1samp(x, popmean=pop_mean)
            test_name = "One-sample t-test"

        elif method == "difference_mean":
            # Two-sample t-test
            equal_var = homo_result["equal_var"]
            t_stat, p_value = stats.ttest_ind(x, y, equal_var=equal_var)
            test_name = "Two-sample t-test"

        elif method == "proportion":
            # Proportion test (z-test)

            x = np.asarray(x)

            # Caso 1: datos ya binarios
            unique_vals = np.unique(x)
            if set(unique_vals).issubset({0, 1}):

                if pop_proportion is None:
                    raise ValueError("Debe especificarse pop_proportion")

                pop_p = pop_proportion

            # Caso 2: datos continuos → binarizar
            else:
                if not isinstance(pop_proportion, tuple):
                    raise ValueError(
                        "Para datos continuos, pop_proportion debe ser (p0, binizar_value)"
                    )

                pop_p, binizar_value = pop_proportion
                x = (x > binizar_value).astype(int)

            if not (0 < pop_p < 1):
                raise ValueError("pop_proportion debe estar entre 0 y 1")

            n = len(x)
            p_hat = np.mean(x)

            if n * pop_p < 5 or n * (1 - pop_p) < 5:
                raise ValueError(
                    "Condiciones del Z-test no cumplidas: np0 y n(1-p0) deben ser ≥ 5"
                )

            z_stat = (p_hat - pop_p) / np.sqrt(pop_p * (1 - pop_p) / n)
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

        if p_value < alpha:
            self.interpretation = "Se RECHAZA la hipótesis nula"
        else:
            self.interpretation = ("Se RECHAZA la hipotesis alternativa")
        return TestResult(
            test_name=test_name,
            statistic=t_stat,
            pvalue=p_value,
            alternative='two-sided',
            alpha=alpha,
            homo_result=homo_result
        )
    
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
                    center: Literal['mean', 'median', 'trimmed'] = 'median',
                    alpha: float = 0.05) -> 'TestResult':
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
            params=params,
            alpha=alpha
        )

    
    def help(self):
        """
        Muestra ayuda completa de la clase DescriptiveStats

        Parametros / Parameters:
        ------------------------
        lang: str
            Idioma Usuario: Codigo de Idioma (es-Es) o "Español"
            User Language: Languaje Code (en-Us) or "English"
        """

        if self.lang in ["en-US", "English", "english"]:
            self.lang = "en-US"
        else:
            self.lang = "es-ES"
        help_text = " "
        match self.lang:
            case "es-ES":
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
    │ import pandas as pd                                                     │
    │                                                                         │
    │ df = pd.read_csv('datos.csv')                                           │
    │ inf_stats = InferentialStats(df)                                        │
    │                                                                         │
    │ # IC para la media (95%)                                                │
    │ lower, upper, mean = inf_stats.confidence_interval(                     │
    │     'salario',                                                          │
    │     confidence=0.95,                                                    │
    │     statistic='mean'                                                    │
    │ )                                                                       │
    │ print(f"IC 95%: [{lower:.2f}, {upper:.2f}]")                            │
    │                                                                         │
    │ # IC para la mediana (bootstrap)                                        │
    │ lower, upper, median = inf_stats.confidence_interval(                   │
    │     'edad',                                                             │
    │     confidence=0.99,                                                    │
    │     statistic='median'                                                  │
    │ )                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 2: Prueba t de Una Muestra ────────────────────────────────────┐
    │ # H0: μ = 50000 (la media salarial es 50000)                            │
    │ # H1: μ ≠ 50000 (prueba bilateral)                                      │
    │                                                                         │
    │ resultado = inf_stats.t_test_1sample(                                   │
    │     column='salario',                                                   │
    │     popmean=50000,                                                      │
    │     alternative='two-sided'                                             │
    │ )                                                                       │
    │                                                                         │
    │ print(resultado)                                                        │
    │ # Muestra: estadístico t, valor p, interpretación                       │
    │                                                                         │
    │ # Prueba unilateral                                                     │
    │ resultado = inf_stats.t_test_1sample(                                   │
    │     column='salario',                                                   │
    │     popmean=50000,                                                      │
    │     alternative='greater'  # H1: μ > 50000                              │
    │ )                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 3: Comparación de Dos Grupos ──────────────────────────────────┐
    │ # Prueba t independiente                                                │
    │ resultado = inf_stats.t_test_2sample(                                   │
    │     column1='salario_hombres',                                          │
    │     column2='salario_mujeres',                                          │
    │     equal_var=True,                                                     │
    │     alternative='two-sided'                                             │
    │ )                                                                       │
    │ print(resultado)                                                        │
    │                                                                         │
    │ # Prueba Mann-Whitney (no paramétrica)                                  │
    │ resultado = inf_stats.mann_whitney_test(                                │
    │     column1='salario_grupo_a',                                          │
    │     column2='salario_grupo_b',                                          │
    │     alternative='two-sided'                                             │
    │ )                                                                       │
    │                                                                         │
    │ # Prueba t pareada (mediciones antes/después)                           │
    │ resultado = inf_stats.t_test_paired(                                    │
    │     column1='peso_antes',                                               │
    │     column2='peso_despues',                                             │
    │     alternative='two-sided'                                             │
    │ )                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 4: ANOVA y Kruskal-Wallis ─────────────────────────────────────┐
    │ # ANOVA para comparar múltiples grupos                                  │
    │ resultado = inf_stats.anova_oneway(                                     │
    │     column='rendimiento',                                               │
    │     groups='departamento'                                               │
    │ )                                                                       │
    │ print(resultado)                                                        │
    │                                                                         │
    │ # Kruskal-Wallis (alternativa no paramétrica)                           │
    │ resultado = inf_stats.kruskal_wallis_test(                              │
    │     column='satisfaccion',                                              │
    │     groups='categoria'                                                  │
    │ )                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 5: Chi-Cuadrado ───────────────────────────────────────────────┐
    │ # Probar independencia entre variables categóricas                      │
    │ resultado = inf_stats.chi_square_test(                                  │
    │     column1='genero',                                                   │
    │     column2='preferencia_producto'                                      │
    │ )                                                                       │
    │ print(resultado)                                                        │
    │                                                                         │
    │ # El resultado incluye la tabla de contingencia                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Ejemplo 6: Pruebas de Normalidad ──────────────────────────────────────┐
    │ # Shapiro-Wilk (recomendado para n ≤ 5000)                              │
    │ resultado = inf_stats.normality_test(                                   │
    │     column='edad',                                                      │
    │     method='shapiro',                                                   │
    │     test_statistic='mean'                                               │
    │ )                                                                       │
    │ print(resultado)                                                        │
    │                                                                         │
    │ # Kolmogorov-Smirnov                                                    │
    │ resultado = inf_stats.normality_test(                                   │
    │     column='salario',                                                   │
    │     method='ks'                                                         │
    │ )                                                                       │
    │                                                                         │
    │ # Ejecutar todos los tests                                              │
    │ resultados = inf_stats.normality_test(                                  │
    │     column='ingresos',                                                  │
    │     method='all',                                                       │
    │     test_statistic='median'                                             │
    │ )                                                                       │
    │                                                                         │
    │ # Acceder a cada test                                                   │
    │ print(resultados['shapiro'])                                            │
    │ print(resultados['kolmogorov_smirnov'])                                 │
    │ print(resultados['anderson_darling'])                                   │
    │ print(resultados['jarque_bera'])                                        │
    └─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 GUÍA DE SELECCIÓN DE PRUEBAS:

    ┌─ Comparar Una Muestra vs Valor de Referencia ───────────────────────────┐
    │ Datos normales        → t_test_1sample (con popmean)                    │
    │ Datos no normales     → t_test_1sample (con popmedian, usa Wilcoxon)    │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Comparar Dos Grupos Independientes ────────────────────────────────────┐
    │ Datos normales        → t_test_2sample                                  │
    │ Datos no normales     → mann_whitney_test                               │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Comparar Dos Grupos Pareados ──────────────────────────────────────────┐
    │ Datos normales        → t_test_paired                                   │
    │ Datos no normales     → (use scipy.stats.wilcoxon directamente)         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Comparar Múltiples Grupos ─────────────────────────────────────────────┐
    │ Datos normales        → anova_oneway                                    │
    │ Datos no normales     → kruskal_wallis_test                             │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Probar Independencia entre Categóricas ────────────────────────────────┐
    │ Variables categóricas → chi_square_test                                 │
    └─────────────────────────────────────────────────────────────────────────┘

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
            case "en-US":
                help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   🔬 CLASS InferentialStats - COMPLETE HELP                ║
╚════════════════════════════════════════════════════════════════════════════╝

📝 DESCRIPTION:
    Class for inferential statistics: hypothesis tests, intervals 
    confidence and normality tests. Allows inferences to be made about 
    populations from data samples.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 MAIN METHODS:

┌────────────────────────────────────────────────────────────────────────────┐
│ 1. 📊 CONFIDENCE INTERVALS                                                 │
└────────────────────────────────────────────────────────────────────────────┘

    • .confidence_interval(column, confidence=0.95, statistic='mean')
    
        Calculate confidence intervals for different statistics
        
        Parameters:
            column      : Column to analyze (str)
            confidence  : Confidence level (float, default 0.95 = 95%)
            statistic   : 'mean', 'median' or 'proportion'
        
        Return: (lower_bound, upper_bound, point_estimate)

┌────────────────────────────────────────────────────────────────────────────┐
│ 2. 🧪 HYPOTHESIS TESTING - A SAMPLE                                        │
└────────────────────────────────────────────────────────────────────────────┘

    • .t_test_1sample(column, popmean=None, popmedian=None, 
                        alternative='two-sided')
    
        One sample t test (or Wilcoxon for median)
    
        Parameters:
            column      : Column to analyze
            popmean     : Hypothetical population mean (for t-test)
            popmedian   : Hypothetical population median (for Wilcoxon)
            alternative : 'two-sided', 'less', 'greater'

┌────────────────────────────────────────────────────────────────────────────┐
│ 3. 🧪 HYPOTHESIS TESTING - TWO SAMPLES                                     │
└────────────────────────────────────────────────────────────────────────────┘

    🔹 Parametric Tests:

    • .t_test_2sample(column1, column2, equal_var=True, 
                    alternative='two-sided')
        Two independent samples t test

    • .t_test_paired(column1, column2, alternative='two-sided')
        Paired t test (dependent samples)

    🔹 Non-Parametric Tests:

    • .mann_whitney_test(column1, column2, alternative='two-sided')
        Non-parametric alternative to the two-sample t-test

    🔹 Extra Tests:
    • .hypothesis_test(method='mean', column1=None, column2=None, 
                    alpha=0.05, homoscedasticity='levene')
    • .variance_test(column1, column2, method='levene', center='median')
    

┌────────────────────────────────────────────────────────────────────────────┐
│ 4. 🧪 TESTING FOR MULTIPLE GROUPS                                          │
└────────────────────────────────────────────────────────────────────────────┘

    🔹 Parametric Tests:

    • .anova_oneway(column, groups)
        One-way ANOVA to compare multiple groups

    🔹 Non-Parametric Tests:

    • .kruskal_wallis_test(column, groups)
        Non-parametric alternative to ANOVA

┌────────────────────────────────────────────────────────────────────────────┐
│ 5. 🧪 TESTS FOR CATEGORICAL VARIABLES                                      │
└────────────────────────────────────────────────────────────────────────────┘

    • .chi_square_test(column1, column2)
        Chi-square test of independence between categorical variables

┌────────────────────────────────────────────────────────────────────────────┐
│ 6. 📈 NORMALITY TESTS                                                      │
└────────────────────────────────────────────────────────────────────────────┘

    • .normality_test(column, method='shapiro', test_statistic='mean')
    
        Tests whether the data follows a normal distribution
        
        Available methods:
            'shapiro'      : Shapiro-Wilk (best for n ≤ 5000)
            'ks'           : Kolmogorov-Smirnov
            'anderson'     : Anderson-Darling
            'jarque_bera'  : Jarque-Bera (based on skewness and kurtosis)
            'all'          : Run all tests
        
        test_statistic: 'mean', 'median' o 'mode' to focus the distribution

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 EXAMPLES OF USE:

    ┌─ Example 1: Confidence Intervals ───────────────────────────────────────┐
    │ from inferential import InferentialStats                                │
    │ import pandas as pd                                                     │
    │                                                                         │
    │ df = pd.read_csv('data.csv')                                            │
    │ inf_stats = InferentialStats(df)                                        │
    │                                                                         │
    │ # CI for mean (95%)                                                     │
    │ lower, upper, mean = inf_stats.confidence_interval(                     │
    │     'salario',                                                          │
    │     confidence=0.95,                                                    │
    │     statistic='mean'                                                    │
    │ )                                                                       │
    │ print(f"IC 95%: [{lower:.2f}, {upper:.2f}]")                            │
    │                                                                         │
    │ # CI for the median (bootstrap)                                         │
    │ lower, upper, median = inf_stats.confidence_interval(                   │
    │     'edad',                                                             │
    │     confidence=0.99,                                                    │
    │     statistic='median'                                                  │
    │ )                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Example 2: One Sample t-test ──────────────────────────────────────────┐
    │ # H0: μ = 50000 (the average salary is 50,000)                          │
    │ # H1: μ ≠ 50000 (two-sided test)                                        │
    │                                                                         │
    │ result = inf_stats.t_test_1sample(                                      │
    │     column='salary',                                                    │
    │     popmean=50000,                                                      │
    │     alternative='two-sided'                                             │
    │ )                                                                       │
    │                                                                         │
    │ print(result)                                                           │
    │ # Sample: t-statistic, p-value, interpretation                          │
    │                                                                         │
    │ # One-sided test                                                        │
    │ result = inf_stats.t_test_1sample(                                      │
    │     column='salary',                                                    │
    │     popmean=50000,                                                      │
    │     alternative='greater'  # H1: μ > 50000                              │
    │ )                                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Example 3: Comparison of Two Groups ───────────────────────────────────┐
    │ # Independent t test                                                    │ 
    │ result = inf_stats.t_test_2sample(                                      │ 
    │     column1='men_salary',                                               │ 
    │     column2='women_salary',                                             │ 
    │     equal_var=True,                                                     │ 
    │     alternative='two-sided'                                             │ 
    │ )                                                                       │ 
    │ print(result)                                                           │ 
    │                                                                         │ 
    │ # Mann-Whitney test (non-parametric)                                    │ 
    │     result = inf_stats.mann_whitney_test(                               │ 
    │     column1='salary_group_a',                                           │ 
    │     column2='salary_group_b',                                           │ 
    │     alternative='two-sided'                                             │ 
    │ )                                                                       │ 
    │                                                                         │ 
    │ # Paired t-test (before/after measurements)                             │ 
    │ result = inf_stats.t_test_paired(                                       │ 
    │     column1='weight_before',                                            │ 
    │     column2='after_weight',                                             │ 
    │     alternative='two-sided'                                             │ 
    │)                                                                        │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Example 4: ANOVA and Kruskal-Wallis ───────────────────────────────────┐
    │ # ANOVA to compare multiple groups                                      │ 
    │ result = inf_stats.anova_oneway(                                        │ 
    │     column='performance',                                               │ 
    │     groups='department'                                                 │ 
    │ )                                                                       │ 
    │ print(result)                                                           │ 
    │                                                                         │ 
    │ # Kruskal-Wallis (non-parametric alternative)                           │ 
    │ result = inf_stats.kruskal_wallis_test(                                 │ 
    │     column='satisfaction',                                              │ 
    │     groups='category'                                                   │ 
    │)                                                                        │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Example 5: Chi-Square ─────────────────────────────────────────────────┐
    │ # Test independence between categorical variables                       │ 
    │ result = inf_stats.chi_square_test(                                     │ 
    │     column1='gender',                                                   │ 
    │     column2='product_preference'                                        │ 
    │ )                                                                       │ 
    │ print(result)                                                           │ 
    │                                                                         │ 
    │ # The result includes the contingency table                             │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Example 6: Normality Tests ────────────────────────────────────────────┐
    │ # Shapiro-Wilk (recommended for n ≤ 5000)                               │ 
    │ result = inf_stats.normality_test(                                      │ 
    │     column='age',                                                       │ 
    │     method='shapiro',                                                   │ 
    │     test_statistic='mean'                                               │ 
    │ )                                                                       │ 
    │ print(result)                                                           │ 
    │                                                                         │ 
    │ # Kolmogorov-Smirnov                                                    │ 
    │ result = inf_stats.normality_test(                                      │ 
    │     column='salary',                                                    │ 
    │     method='ks'                                                         │ 
    │ )                                                                       │ 
    │                                                                         │ 
    │ # Run all tests                                                         │ 
    │ results = inf_stats.normality_test(                                     │ 
    │     column='income',                                                    │ 
    │     method='all',                                                       │ 
    │     test_statistic='median'                                             │ 
    │ )                                                                       │ 
    │                                                                         │ 
    │ # Access each test                                                      │ 
    │ print(results['shapiro'])                                               │ 
    │ print(results['kolmogorov_smirnov'])                                    │ 
    │ print(results['anderson_darling'])                                      │ 
    │ print(results['jarque_bera'])                                           │
    └─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 GUÍA DE SELECCIÓN DE PRUEBAS:

    ┌─ Compare A Sample vs Reference Value ───────────────────────────────────┐
    │ Normal data           → t_test_1sample (with mean)                      │ 
    │ Non-normal data       → t_test_1sample (with popmedian, uses Wilcoxon)  │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Compare Two Independent Groups ────────────────────────────────────────┐
    │ Normal data           → t_test_2sample                                  │ 
    │ Non-normal data       → mann_whitney_test                               │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Compare Two Paired Groups ─────────────────────────────────────────────┐
    │ Normal data           → t_test_paired                                   │ 
    │ Non-normal data       → (use scipy.stats.wilcoxon directly)             │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Compare Multiple Groups ───────────────────────────────────────────────┐
    │ Normal data           → anova_oneway                                    │ 
    │ Non-normal data       → kruskal_wallis_test                             │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─ Testing Independence between Categories ───────────────────────────────┐
    │ Categorical variables → chi_square_test                                 │
    └─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 KEY FEATURES: 

    ✓ Parametric and non-parametric tests 
    ✓ Confidence intervals with multiple methods 
    ✓ Complete normality tests 
    ✓ Automatic interpretation of results 
    ✓ Automatic handling of missing values 
    ✓ Professional formatted outputs 
    ✓ Support for bilateral and unilateral analysis

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  INTERPRETATION OF RESULTS: 

    • P value < 0.05: H0 is rejected (significant evidence) 
    • P value ≥ 0.05: H0 is not rejected (insufficient evidence) 
    • CI that does not include the null value: Evidence against H0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 ADDITIONAL DOCUMENTATION: 
    For more information on specific methods, use: 
    help(InferentialStats.method_name)

╚════════════════════════════════════════════════════════════════════════════╝
"""
        print(help_text)

@dataclass
class TestResult:
    """Clase para resultados de pruebas de hipótesis"""
    
    def __init__(self, test_name: str, statistic: float, alpha: float = 0.05,
                    params: dict = None, pvalue: float = None, 
                    alternative: str = None, critical_values=None, 
                    significance_levels=None, homo_result=None):
        self.test_name = test_name
        self.statistic = statistic
        self.pvalue = pvalue
        self.alternative = alternative
        self.params = params
        self.critical_values = critical_values
        self.significance_levels = significance_levels
        self.interpretation = "Aun no hay interpretacion"
        self.homo_result = homo_result
        self.alpha = alpha

        if self.pvalue is not None:
            if self.pvalue < self.alpha:
                self.interpretation = "Se RECHAZA la hipótesis nula"
            else:
                self.interpretation = "Se RECHAZA la hipótesis alternativa"
        
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

        # Mostrar valores críticos o p-value
        if self.critical_values is not None and self.significance_levels is not None:
            output.append("Valores Críticos:")
            for sl, cv in zip(self.significance_levels, self.critical_values):
                output.append(f"  α = {sl:>6.3f} → {cv:.6f}")
        elif self.pvalue is not None:
            output.append(f"{'Valor p':<40} {self.pvalue:>20.6e}")

        # -------------------------
        # INTERPRETACIÓN
        # -------------------------
        output.append("\nINTERPRETACIÓN:")
        output.append("-" * 80)

        alpha = 0.05

        # Caso tests con p-value
        if self.pvalue is not None:
            output.append(f"Alpha = {alpha}")

            if self.pvalue < alpha:
                output.append("❌ Se RECHAZA la hipótesis nula")
            else:
                output.append("✔️ No hay evidencia suficiente para rechazar la hipótesis nula")

        # Caso tests con valores críticos (ej. Anderson-Darling)
        else:
            # Protección mínima
            if self.significance_levels is None or self.critical_values is None:
                output.append("Resultado no disponible")
            else:
                idx = min(
                    range(len(self.significance_levels)),
                    key=lambda i: abs(self.significance_levels[i] - alpha)
                )

                critical_value = self.critical_values[idx]

                output.append(f"Nivel de significancia (α) = {alpha}")
                output.append(f"Estadístico A² = {self.statistic:.4f}")
                output.append(f"Valor crítico = {critical_value:.4f}")

                if self.statistic > critical_value:
                    output.append("❌ Se RECHAZA la hipótesis nula")
                else:
                    output.append("✔️ No hay evidencia suficiente para rechazar la hipótesis nula")

        # -------------------------
        # HOMOCEDASTICIDAD
        # -------------------------
        if isinstance(self.homo_result, dict):
            homo = self.homo_result

            if isinstance(homo, dict):
                output.append("\nTEST DE HOMOCEDASTICIDAD:")
                output.append(f"Método: {homo['method']}")
                output.append(f"Estadístico: {homo['statistic']:.6f}")
                output.append(f"Valor p: {homo['p_value']:.6e}")

                if homo.get("equal_var") is True:
                    output.append("✔️ Se asume igualdad de varianzas")
                elif homo.get("equal_var") is False:
                    output.append("❌ No se asume igualdad de varianzas")

        # -------------------------
        # PARÁMETROS
        # -------------------------
        if isinstance(self.params, dict):
            output.append("\nPARÁMETROS:")
            output.append("-" * 80)
            for k, v in self.params.items():
                output.append(f"{k:<40} {str(v):>20}")
                
        output.append("=" * 80)
        return "\n".join(output)
