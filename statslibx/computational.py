from itertools import combinations
from typing import Union, Optional, Literal, List, Tuple, Any, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sympy as sp
from scipy import stats as scipy_stats
from scipy.optimize import minimize
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import logging

from .backend import Backend

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BaseResult(ABC):
    """Base class for all statistical results"""
    
    def __init__(self, name: str):
        self.name = name
        self._fitted = False
        
    @abstractmethod
    def summary(self) -> Dict:
        """Return summary of the results"""
        pass
    
    @abstractmethod
    def plot(self, **kwargs):
        """Plot the results"""
        pass
    
    def __repr__(self):
        return f"<{self.name} Result>"


@dataclass
class RegressionResult(BaseResult):
    """Enhanced regression result class"""
    X: Union[pd.Series, pd.DataFrame, np.ndarray]
    y: pd.Series
    degree: int = 1
    interaction_terms: bool = False
    
    def __post_init__(self):
        super().__init__("Regression")
        
        # Process input data
        self._process_data()
        
        # Fit the model
        self._fit_model()
        
        # Compute metrics
        self._compute_metrics()
        
        # Create symbolic expression
        self._create_symbolic_expression()
        
        self._fitted = True
    
    def _process_data(self):
        """Process input data"""
        if isinstance(self.X, pd.DataFrame):
            self.X_values = self.X.values
            self.feature_names = self.X.columns.tolist()
            self.n_features = self.X.shape[1]
        else:
            if isinstance(self.X, pd.Series):
                self.X_values = self.X.values.reshape(-1, 1)
                self.feature_names = [self.X.name or "X"]
            else:
                self.X_values = np.array(self.X).reshape(-1, 1)
                self.feature_names = ["X"]
            self.n_features = 1
        
        self.y_values = np.array(self.y).flatten()
        self.n_samples = len(self.y_values)
        
        if self.interaction_terms and self.n_features > 1:
            interaction_cols = []
            interaction_names = []
            for i, j in combinations(range(self.n_features), 2):
                interaction_cols.append(self.X_values[:, i] * self.X_values[:, j])
                interaction_names.append(f"{self.feature_names[i]}*{self.feature_names[j]}")
            if interaction_cols:
                self.X_values = np.column_stack([self.X_values, np.column_stack(interaction_cols)])
                self.feature_names = self.feature_names + interaction_names
                self.n_features = self.X_values.shape[1]

        # Create polynomial features if needed
        if self.degree > 1 and self.n_features == 1:
            self.X_poly = np.column_stack([self.X_values ** i for i in range(1, self.degree + 1)])
            self.X_design = np.column_stack([np.ones(self.n_samples), self.X_poly])
            
            # Generate polynomial feature names
            self.poly_feature_names = ['Intercept']
            for i in range(1, self.degree + 1):
                self.poly_feature_names.append(f'{self.feature_names[0]}^{i}')
            self.all_feature_names = self.poly_feature_names
        else:
            self.X_design = np.column_stack([np.ones(self.n_samples), self.X_values])
            self.all_feature_names = ['Intercept'] + self.feature_names
    
    def _fit_model(self):
        """Fit the regression model"""
        # Solve using normal equations
        try:
            self.coefficients = np.linalg.lstsq(self.X_design, self.y_values, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.coefficients = np.linalg.pinv(self.X_design.T @ self.X_design) @ self.X_design.T @ self.y_values
        
        self.intercept = self.coefficients[0]
        self.slopes = self.coefficients[1:]
        
        # Predictions
        self.y_pred = self.X_design @ self.coefficients
        self.residuals = self.y_values - self.y_pred
    
    def _compute_metrics(self):
        """Compute regression metrics"""
        n = self.n_samples
        p = len(self.coefficients) - 1
        
        # Basic metrics
        self.mse = np.mean(self.residuals ** 2)
        self.rmse = np.sqrt(self.mse)
        self.mae = np.mean(np.abs(self.residuals))
        self.mape = np.mean(np.abs(self.residuals / (self.y_values + 1e-8))) * 100
        
        # R-squared and adjusted R-squared
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.y_values - np.mean(self.y_values)) ** 2)
        self.r2 = 1 - (ss_res / (ss_tot + 1e-8))
        self.r2_adj = 1 - (1 - self.r2) * (n - 1) / (n - p - 1) if n > p + 1 else self.r2
        
        # Information criteria
        self.aic = n * np.log(self.mse) + 2 * (p + 1)
        self.bic = n * np.log(self.mse) + (p + 1) * np.log(n)
        
        # Standard errors and t-statistics
        try:
            XTX_inv = np.linalg.pinv(self.X_design.T @ self.X_design)
            self.std_errors = np.sqrt(np.diag(XTX_inv * self.mse))
            self.t_stats = self.coefficients / (self.std_errors + 1e-8)
            self.p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(self.t_stats), n - p - 1))
        except:
            self.std_errors = np.full_like(self.coefficients, np.nan)
            self.t_stats = np.full_like(self.coefficients, np.nan)
            self.p_values = np.full_like(self.coefficients, np.nan)
    
    def _create_symbolic_expression(self):
        """Create symbolic expression of the model"""
        x = sp.Symbol('x')
        if self.n_features == 1:
            expr = self.intercept
            for i, coef in enumerate(self.slopes):
                expr += coef * (x ** (i + 1))
            self.symbolic_expr = sp.simplify(expr)
            self.latex_expr = sp.latex(self.symbolic_expr)
        else:
            # For multiple features
            symbols = sp.symbols(' '.join([f'x{i+1}' for i in range(self.n_features)]))
            expr = self.intercept
            for i, coef in enumerate(self.slopes):
                expr += coef * symbols[i]
            self.symbolic_expr = sp.simplify(expr)
            self.latex_expr = sp.latex(self.symbolic_expr)
    
    def predict(self, X_new: Union[list, np.ndarray, float, pd.DataFrame]) -> np.ndarray:
        """Make predictions on new data"""
        X_new = np.array(X_new).reshape(-1, self.n_features)
        
        if self.degree > 1 and self.n_features == 1:
            X_new_poly = np.column_stack([X_new ** i for i in range(1, self.degree + 1)])
            X_new_design = np.column_stack([np.ones(len(X_new)), X_new_poly])
        else:
            X_new_design = np.column_stack([np.ones(len(X_new)), X_new])
        
        return X_new_design @ self.coefficients
    
    def summary(self) -> Dict:
        """Return detailed summary"""
        # Ensure all arrays have the same length
        coef_table = pd.DataFrame({
            'Coefficient': self.coefficients,
            'Std Error': self.std_errors,
            't-statistic': self.t_stats,
            'p-value': self.p_values
        }, index=self.all_feature_names)
        
        # Create a more readable formula string
        formula_parts = []
        if self.degree > 1 and self.n_features == 1:
            formula_parts.append(f"{self.coefficients[0]:.4f}")
            for i, coef in enumerate(self.coefficients[1:], 1):
                sign = "+" if coef >= 0 else "-"
                formula_parts.append(f"{sign} {abs(coef):.4f}·x^{i}")
            formula_str = " ".join(formula_parts)
        else:
            formula_str = f"y = {self.coefficients[0]:.4f}"
            for i, (coef, name) in enumerate(zip(self.coefficients[1:], self.feature_names)):
                sign = "+" if coef >= 0 else "-"
                formula_str += f" {sign} {abs(coef):.4f}·{name}"
        
        return {
            'model_info': {
                'degree': self.degree,
                'n_features': self.n_features,
                'n_samples': self.n_samples,
                'formula': formula_str
            },
            'coefficients': coef_table,
            'metrics': {
                'R2': self.r2,
                'Adjusted R2': self.r2_adj,
                'MSE': self.mse,
                'RMSE': self.rmse,
                'MAE': self.mae,
                'MAPE (%)': self.mape,
                'AIC': self.aic,
                'BIC': self.bic
            },
            'formula': {
                'symbolic': self.symbolic_expr,
                'latex': self.latex_expr
            }
        }
    
    def plot(self, plot_type: Literal['scatter', 'residuals', 'qq', 'all'] = 'all', 
             interactive: bool = False, **kwargs):
        """Plot regression results"""
        if plot_type == 'all' and not interactive:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Scatter plot with regression line
            if self.n_features == 1:
                axes[0, 0].scatter(self.X_values.flatten(), self.y_values, alpha=0.6, label='Actual')
                x_range = np.linspace(self.X_values.min(), self.X_values.max(), 100)
                y_range = self.predict(x_range)
                axes[0, 0].plot(x_range, y_range, 'r-', label='Predicted', linewidth=2)
                axes[0, 0].set_xlabel(self.feature_names[0])
            else:
                axes[0, 0].scatter(range(self.n_samples), self.y_values, alpha=0.6, label='Actual')
                axes[0, 0].plot(range(self.n_samples), self.y_pred, 'r-', label='Predicted', linewidth=2, alpha=0.7)
                axes[0, 0].set_xlabel('Sample Index')
            
            axes[0, 0].set_ylabel('y')
            axes[0, 0].set_title(f'Regression Fit (R² = {self.r2:.4f})')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residuals plot
            axes[0, 1].scatter(self.y_pred, self.residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals vs Predicted')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            scipy_stats.probplot(self.residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[1, 1].hist(self.residuals, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Residuals')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Residuals')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        elif interactive and self.n_features == 1:
            # Interactive plot with plotly
            fig = go.Figure()
            
            # Scatter points
            fig.add_trace(go.Scatter(
                x=self.X_values.flatten(),
                y=self.y_values,
                mode='markers',
                name='Actual',
                marker=dict(size=8, opacity=0.6)
            ))
            
            # Regression line
            x_range = np.linspace(self.X_values.min(), self.X_values.max(), 100)
            y_range = self.predict(x_range)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                name=f'Polynomial (degree={self.degree})',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f'Regression Analysis (R² = {self.r2:.4f})',
                xaxis_title=self.feature_names[0],
                yaxis_title='y',
                hovermode='closest',
                template='plotly_white'
            )
            
            fig.show()
        elif plot_type in ['scatter', 'residuals', 'qq']:
            # Single plot
            plt.figure(figsize=(8, 6))
            if plot_type == 'scatter' and self.n_features == 1:
                plt.scatter(self.X_values.flatten(), self.y_values, alpha=0.6)
                x_range = np.linspace(self.X_values.min(), self.X_values.max(), 100)
                y_range = self.predict(x_range)
                plt.plot(x_range, y_range, 'r-', linewidth=2)
                plt.xlabel(self.feature_names[0])
                plt.ylabel('y')
                plt.title(f'Regression Fit (R² = {self.r2:.4f})')
            elif plot_type == 'residuals':
                plt.scatter(self.y_pred, self.residuals, alpha=0.6)
                plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                plt.title('Residuals vs Predicted')
            elif plot_type == 'qq':
                scipy_stats.probplot(self.residuals, dist="norm", plot=plt)
                plt.title('Q-Q Plot')
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def get_formula(self, decimals: int = 4) -> str:
        """Get the regression formula as a string"""
        if self.degree > 1 and self.n_features == 1:
            parts = [f"{self.coefficients[0]:.{decimals}f}"]
            for i, coef in enumerate(self.coefficients[1:], 1):
                sign = "+" if coef >= 0 else "-"
                parts.append(f"{sign} {abs(coef):.{decimals}f}·x^{i}")
            return "y = " + " ".join(parts)
        else:
            formula = f"y = {self.coefficients[0]:.{decimals}f}"
            for coef, name in zip(self.coefficients[1:], self.feature_names):
                sign = "+" if coef >= 0 else "-"
                formula += f" {sign} {abs(coef):.{decimals}f}·{name}"
            return formula
    
    def __repr__(self):
        return f"<RegressionResult degree={self.degree}, R²={self.r2:.4f}, n={self.n_samples}>"
    
@dataclass
class InterpolationResult(BaseResult):
    """Enhanced interpolation result class"""
    points: List[Tuple[float, float]]
    method: Literal['lagrange', 'newton', 'spline'] = 'lagrange'
    spline_degree: int = 3
    
    def __post_init__(self):
        super().__init__("Interpolation")
        self._compute_interpolation()
        self._create_symbolic_expression()
        self._fitted = True
    
    def _compute_interpolation(self):
        """Compute interpolation based on chosen method"""
        self.x_points = np.array([p[0] for p in self.points])
        self.y_points = np.array([p[1] for p in self.points])
        
        if self.method == 'lagrange':
            self._lagrange_interpolation()
        elif self.method == 'newton':
            self._newton_interpolation()
        elif self.method == 'spline':
            self._spline_interpolation()
    
    def _lagrange_interpolation(self):
        """Compute Lagrange interpolation"""
        n = len(self.x_points)
        
        def lagrange_poly(x):
            total = 0
            for i in range(n):
                term = self.y_points[i]
                for j in range(n):
                    if j != i:
                        term *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
                total += term
            return total
        
        self._interpolator = lagrange_poly
        
        # For vectorized operations
        self._vectorized_interpolator = np.vectorize(lagrange_poly)
    
    def _newton_interpolation(self):
        """Compute Newton interpolation"""
        n = len(self.x_points)
        self.divided_diff = np.copy(self.y_points)
        
        for j in range(1, n):
            for i in range(n-1, j-1, -1):
                self.divided_diff[i] = (self.divided_diff[i] - self.divided_diff[i-1]) / (self.x_points[i] - self.x_points[i-j])
        
        def newton_poly(x):
            result = self.divided_diff[0]
            product = 1
            for i in range(1, n):
                product *= (x - self.x_points[i-1])
                result += self.divided_diff[i] * product
            return result
        
        self._interpolator = newton_poly
        self._vectorized_interpolator = np.vectorize(newton_poly)
    
    def _spline_interpolation(self):
        """Compute spline interpolation"""
        from scipy.interpolate import interp1d
        self._interpolator = interp1d(self.x_points, self.y_points, 
                                       kind=self.spline_degree, 
                                       bounds_error=False, 
                                       fill_value='extrapolate')
        self._vectorized_interpolator = self._interpolator
    
    def _create_symbolic_expression(self):
        """Create symbolic expression for the interpolation"""
        if self.method != 'spline':
            x = sp.Symbol('x')
            n = len(self.x_points)
            
            if self.method == 'lagrange':
                expr = 0
                for i in range(n):
                    term = self.y_points[i]
                    for j in range(n):
                        if j != i:
                            term *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
                    expr += term
                self.symbolic_expr = sp.simplify(expr)
            else:  # newton
                expr = self.divided_diff[0]
                product = 1
                for i in range(1, n):
                    product *= (x - self.x_points[i-1])
                    expr += self.divided_diff[i] * product
                self.symbolic_expr = sp.simplify(expr)
            
            self.latex_expr = sp.latex(self.symbolic_expr)
        else:
            self.symbolic_expr = "Spline interpolation (non-polynomial)"
            self.latex_expr = "\\text{Spline interpolation}"
    
    def predict(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate interpolation at given points"""
        return self._vectorized_interpolator(x)
    
    def summary(self) -> Dict:
        """Return summary of interpolation"""
        return {
            'method': self.method,
            'n_points': len(self.points),
            'x_range': (min(self.x_points), max(self.x_points)),
            'y_range': (min(self.y_points), max(self.y_points)),
            'formula': {
                'symbolic': self.symbolic_expr if hasattr(self, 'symbolic_expr') else None,
                'latex': self.latex_expr if hasattr(self, 'latex_expr') else None
            }
        }
    
    def plot(self, n_points: int = 1000, interactive: bool = False, **kwargs):
        """Plot interpolation result"""
        x_range = np.linspace(min(self.x_points), max(self.x_points), n_points)
        y_range = self.predict(x_range)
        
        if interactive:
            fig = go.Figure()
            
            # Original points
            fig.add_trace(go.Scatter(
                x=self.x_points,
                y=self.y_points,
                mode='markers',
                name='Data points',
                marker=dict(size=10, color='red')
            ))
            
            # Interpolation curve
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                name=f'{self.method.capitalize()} interpolation',
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f'Interpolation ({self.method.capitalize()})',
                xaxis_title='x',
                yaxis_title='y',
                hovermode='closest',
                template='plotly_white'
            )
            
            fig.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(x_range, y_range, 'b-', label=f'{self.method.capitalize()} interpolation', linewidth=2)
            plt.plot(self.x_points, self.y_points, 'ro', label='Data points', markersize=8)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Interpolation ({self.method.capitalize()})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def __repr__(self):
        return f"<InterpolationResult method={self.method}, n_points={len(self.points)}>"


@dataclass
class BootstrappingResult(BaseResult):
    """Enhanced bootstrapping result class"""
    data: np.ndarray
    n_samples: int = 1000
    statistic: Literal['mean', 'median', 'std', 'custom'] = 'mean'
    confidence_level: float = 0.95
    custom_func: Optional[callable] = None
    
    def __post_init__(self):
        super().__init__("Bootstrapping")
        self._compute_bootstrap()
        self._compute_confidence_intervals()
        self._fitted = True
    
    def _compute_bootstrap(self):
        """Perform bootstrap resampling"""
        self.bootstrap_stats = []
        n = len(self.data)
        
        if self.statistic == 'mean':
            stat_func = np.mean
        elif self.statistic == 'median':
            stat_func = np.median
        elif self.statistic == 'std':
            stat_func = np.std
        elif self.statistic == 'custom' and self.custom_func is not None:
            stat_func = self.custom_func
        else:
            raise ValueError(f"Unknown statistic: {self.statistic}")
        
        for _ in range(self.n_samples):
            sample = np.random.choice(self.data, size=n, replace=True)
            self.bootstrap_stats.append(stat_func(sample))
        
        self.bootstrap_stats = np.array(self.bootstrap_stats)
        self.original_stat = stat_func(self.data)
        self.bias = np.mean(self.bootstrap_stats) - self.original_stat
        self.std_error = np.std(self.bootstrap_stats)
    
    def _compute_confidence_intervals(self):
        """Compute confidence intervals"""
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        self.percentile_ci = (
            np.percentile(self.bootstrap_stats, lower_percentile),
            np.percentile(self.bootstrap_stats, upper_percentile)
        )
        
        self.basic_ci = (
            2 * self.original_stat - self.percentile_ci[1],
            2 * self.original_stat - self.percentile_ci[0]
        )
        
        self.normal_ci = (
            self.original_stat - scipy_stats.norm.ppf(1 - alpha/2) * self.std_error,
            self.original_stat + scipy_stats.norm.ppf(1 - alpha/2) * self.std_error
        )
    
    def summary(self) -> Dict:
        """Return bootstrapping summary"""
        return {
            'original_statistic': self.original_stat,
            'bootstrap_mean': np.mean(self.bootstrap_stats),
            'bias': self.bias,
            'std_error': self.std_error,
            f'confidence_interval_{self.confidence_level*100:.0f}%': {
                'percentile': self.percentile_ci,
                'basic': self.basic_ci,
                'normal': self.normal_ci
            },
            'distribution': self.bootstrap_stats
        }
    
    def plot(self, interactive: bool = False, **kwargs):
        """Plot bootstrap distribution"""
        if interactive:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Bootstrap Distribution', 'Q-Q Plot'),
                specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=self.bootstrap_stats, nbinsx=30, name='Distribution'),
                row=1, col=1
            )
            
            # Add vertical line for original statistic
            fig.add_vline(x=self.original_stat, line_color='red', line_dash='dash',
                          annotation_text=f'Original: {self.original_stat:.4f}')
            
            # Q-Q plot
            theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(self.bootstrap_stats)))
            sample_quantiles = np.percentile(self.bootstrap_stats, np.linspace(1, 99, len(self.bootstrap_stats)))
            
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Q-Q'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', name='y=x', line=dict(dash='dash')),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f'Bootstrapping Results ({self.statistic.capitalize()})',
                showlegend=True,
                template='plotly_white'
            )
            
            fig.show()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            axes[0].hist(self.bootstrap_stats, bins=30, edgecolor='black', alpha=0.7)
            axes[0].axvline(self.original_stat, color='red', linestyle='--', 
                           label=f'Original: {self.original_stat:.4f}')
            axes[0].axvline(np.mean(self.bootstrap_stats), color='green', linestyle='--',
                           label=f'Bootstrap mean: {np.mean(self.bootstrap_stats):.4f}')
            axes[0].set_xlabel(f'Bootstrap {self.statistic}')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Bootstrap Distribution ({self.statistic.capitalize()})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Q-Q plot
            scipy_stats.probplot(self.bootstrap_stats, dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def __repr__(self):
        return f"<BootstrappingResult statistic={self.statistic}, n_samples={self.n_samples}, original={self.original_stat:.4f}>"


class ComputationalStats:
    """
    Enhanced class for computational statistics with improved functionality
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        backend: Optional[Literal["pandas", "polars"]] = None,
        seed: Optional[int] = None,
        lang: Literal['es-ES', 'en-US'] = 'es-ES',
    ):
        """
        Initialize ComputationalStats object.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Input data.
        backend : {'pandas', 'polars'}, optional
            Data engine to use. Auto-detects from input type when None.
        seed : int, optional
            Random seed for reproducibility.
        lang : {'es-ES', 'en-US'}, default 'es-ES'
            Language for outputs.
        """
        self._backend = Backend(data, backend=backend)
        self.data = self._backend.df
        
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self._numeric_cols = self._backend.numeric_columns()
        self._categorical_cols = self._backend.categorical_columns()
        self.lang = lang
        
        self._translations = {
            'es-ES': {
                'regression': 'Regresión',
                'polynomial': 'Polinomial',
                'interpolation': 'Interpolación',
                'bootstrapping': 'Remuestreo Bootstrap'
            },
            'en-US': {
                'regression': 'Regression',
                'polynomial': 'Polynomial',
                'interpolation': 'Interpolation',
                'bootstrapping': 'Bootstrapping'
            }
        }

    @classmethod
    def from_file(
        cls,
        path: str,
        backend: str = "pandas",
        sep: str = ",",
        seed: Optional[int] = None,
        lang: Literal['es-ES', 'en-US'] = 'es-ES',
    ) -> "ComputationalStats":
        """Load data from a file and return a ComputationalStats instance."""
        from .datasets import load_dataset
        return cls(
            load_dataset(path, backend=backend, sep=sep),
            backend=backend,
            seed=seed,
            lang=lang,
        )
    
    @property
    def backend(self):
        """Return the active data engine ('pandas' or 'polars')."""
        return self._backend.type

    @property
    def backend_engine(self) -> Backend:
        """Return the internal Backend wrapper."""
        return self._backend
    
    def regression(self, X: Union[List[str], str], y: str, 
                   degree: int = 1, interaction_terms: bool = False) -> RegressionResult:
        """
        Perform polynomial regression
        
        Parameters:
        -----------
        X : str or list of str
            Independent variable(s)
        y : str
            Dependent variable
        degree : int
            Polynomial degree (only for single variable)
        interaction_terms : bool
            Whether to include interaction terms (for multiple variables)
        
        Returns:
        --------
        RegressionResult object
        """
        pdf = self._backend.to_pandas()
        x_cols = [X] if isinstance(X, str) else list(X)
        X_data = pdf[x_cols]
        y_data = pdf[y]
        
        return RegressionResult(X_data, y_data, degree=degree, interaction_terms=interaction_terms)
    
    def linear_regression(self, X: Union[List[str], str], y: str) -> RegressionResult:
        """Perform linear regression (wrapper for regression with degree=1)"""
        return self.regression(X, y, degree=1)
    
    def polynomial_regression(self, X: str, y: str, degree: int = 2) -> RegressionResult:
        """Perform polynomial regression"""
        return self.regression(X, y, degree=degree)
    
    def find_best_degree(self, X: str, y: str, max_degree: int = 5, 
                         metric: Literal['r2', 'aic', 'bic'] = 'r2') -> Dict:
        """
        Find the best polynomial degree based on specified metric
        
        Parameters:
        -----------
        X : str
            Independent variable
        y : str
            Dependent variable
        max_degree : int
            Maximum degree to test
        metric : str
            Metric to optimize ('r2', 'aic', 'bic')
        
        Returns:
        --------
        Dictionary with results for each degree
        """
        results = []
        
        for degree in range(1, max_degree + 1):
            model = self.regression(X, y, degree=degree)
            metrics = model.summary()['metrics']
            
            results.append({
                'degree': degree,
                'r2': metrics['R2'],
                'adj_r2': metrics['Adjusted R2'],
                'aic': metrics['AIC'],
                'bic': metrics['BIC'],
                'rmse': metrics['RMSE'],
                'model': model
            })
        
        # Find best based on metric
        if metric == 'r2':
            best_idx = np.argmax([r['r2'] for r in results])
        elif metric in ['aic', 'bic']:
            best_idx = np.argmin([r[metric] for r in results])
        else:
            best_idx = np.argmax([r[metric] for r in results])
        
        best_result = results[best_idx]
        best_result['best_metric'] = metric
        best_result['all_results'] = results
        
        return best_result
    
    def interpolation(self, points: List[Tuple[float, float]], 
                     method: Literal['lagrange', 'newton', 'spline'] = 'lagrange',
                     spline_degree: int = 3) -> InterpolationResult:
        """
        Perform interpolation on given points
        
        Parameters:
        -----------
        points : list of tuples
            List of (x, y) points
        method : str
            Interpolation method ('lagrange', 'newton', 'spline')
        spline_degree : int
            Degree for spline interpolation (1-5)
        
        Returns:
        --------
        InterpolationResult object
        """
        return InterpolationResult(points, method=method, spline_degree=spline_degree)
    
    def bootstrapping(self, column: str, n_samples: int = 1000,
                     statistic: Literal['mean', 'median', 'std', 'custom'] = 'mean',
                     confidence_level: float = 0.95,
                     custom_func: Optional[callable] = None) -> BootstrappingResult:
        """
        Perform bootstrapping on a column
        
        Parameters:
        -----------
        column : str
            Column name to bootstrap
        n_samples : int
            Number of bootstrap samples
        statistic : str
            Statistic to compute ('mean', 'median', 'std', 'custom')
        confidence_level : float
            Confidence level for intervals
        custom_func : callable, optional
            Custom function for statistic
        
        Returns:
        --------
        BootstrappingResult object
        """
        data = self._backend.col_numpy(column)
        data = data[~np.isnan(data)]
        return BootstrappingResult(data, n_samples, statistic, confidence_level, custom_func)
    
    def k_means(self, k: int, max_iters: int = 100, 
                init_method: Literal['random', 'kmeans++'] = 'kmeans++') -> Dict:
        """
        Perform K-means clustering
        
        Parameters:
        -----------
        k : int
            Number of clusters
        max_iters : int
            Maximum number of iterations
        init_method : str
            Initialization method ('random' or 'kmeans++')
        
        Returns:
        --------
        Dictionary with clustering results
        """
        X = np.column_stack([self._backend.col_numpy(c) for c in self._numeric_cols])
        
        # K-means++ initialization
        if init_method == 'kmeans++':
            centroids = [X[np.random.choice(len(X))]]
            for _ in range(1, k):
                distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
                probabilities = distances / np.sum(distances)
                next_centroid = X[np.random.choice(len(X), p=probabilities)]
                centroids.append(next_centroid)
            centroids = np.array(centroids)
        else:  # random
            np.random.seed(self.seed)
            centroids = X[np.random.choice(len(X), k, replace=False)]
        
        for iteration in range(max_iters):
            # Assign clusters
            distances = np.linalg.norm(X[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 
                                      else centroids[i] for i in range(k)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        # Compute inertia (within-cluster sum of squares)
        inertia = np.sum([np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(k)])
        
        # Compute silhouette score if possible
        if len(np.unique(labels)) > 1:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = -1
        
        return {
            'centroids': centroids,
            'labels': labels,
            'inertia': inertia,
            'silhouette': silhouette,
            'n_iterations': iteration + 1
        }
    
    def elbow_method(self, max_k: int = 10) -> Dict:
        """
        Compute elbow method for K-means clustering
        
        Parameters:
        -----------
        max_k : int
            Maximum number of clusters to test
        
        Returns:
        --------
        Dictionary with inertias for each k
        """
        inertias = []
        silhouettes = []
        
        for k in range(2, max_k + 1):
            result = self.k_means(k)
            inertias.append(result['inertia'])
            silhouettes.append(result['silhouette'])
        
        return {
            'k_values': list(range(2, max_k + 1)),
            'inertias': inertias,
            'silhouettes': silhouettes
        }
    
    def correlation_analysis(self, method: Literal['pearson', 'spearman', 'kendall'] = 'pearson') -> Dict:
        """
        Perform correlation analysis on numeric columns
        
        Parameters:
        -----------
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
        --------
        Dictionary with correlation matrix and p-values
        """
        corr_matrix = self._backend.corr(method=method)
        
        # Compute p-values for Pearson correlation
        p_values = None
        if method == 'pearson':
            n = self._backend.shape[0]
            p_values = pd.DataFrame(index=self._numeric_cols, columns=self._numeric_cols)
            for i in range(len(self._numeric_cols)):
                for j in range(len(self._numeric_cols)):
                    if i <= j:
                        corr, p = scipy_stats.pearsonr(
                            self._backend.col_numpy(self._numeric_cols[i]),
                            self._backend.col_numpy(self._numeric_cols[j])
                        )
                        p_values.iloc[i, j] = p
                        p_values.iloc[j, i] = p
        
        return {
            'correlation_matrix': corr_matrix,
            'p_values': p_values,
            'method': method
        }
    
    def plot_correlation_heatmap(self, method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
                                 annot: bool = True, interactive: bool = False, **kwargs):
        """
        Plot correlation heatmap
        
        Parameters:
        -----------
        method : str
            Correlation method
        annot : bool
            Whether to show correlation values
        interactive : bool
            Whether to use interactive plot
        """
        corr_result = self.correlation_analysis(method)
        corr_matrix = corr_result['correlation_matrix']
        
        if interactive:
            fig = px.imshow(corr_matrix, text_auto=annot, aspect="auto",
                           color_continuous_scale='RdBu', zmin=-1, zmax=1,
                           title=f'Correlation Heatmap ({method.capitalize()})')
            fig.update_layout(template='plotly_white')
            fig.show()
        else:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=annot, cmap='RdBu', center=0,
                       square=True, linewidths=0.5, **kwargs)
            plt.title(f'Correlation Heatmap ({method.capitalize()})')
            plt.tight_layout()
            plt.show()
    
    def descriptive_statistics(self, by: Optional[str] = None) -> pd.DataFrame:
        """
        Compute descriptive statistics
        
        Parameters:
        -----------
        by : str, optional
            Column to group by
        
        Returns:
        --------
        DataFrame with descriptive statistics
        """
        pdf = self._backend.to_pandas()
        if by is None or by not in self._categorical_cols:
            return pdf[self._numeric_cols].describe()
        return pdf.groupby(by)[self._numeric_cols].describe()
    
    def plot_distribution(self, column: str, by: Optional[str] = None, 
                          kind: Literal['hist', 'box', 'violin'] = 'hist',
                          interactive: bool = False, **kwargs):
        """
        Plot distribution of a column
        
        Parameters:
        -----------
        column : str
            Column name
        by : str, optional
            Column to group by
        kind : str
            Type of plot ('hist', 'box', 'violin')
        interactive : bool
            Whether to use interactive plot
        """
        pdf = self._backend.to_pandas()
        if interactive:
            if kind == 'hist':
                fig = px.histogram(pdf, x=column, color=by, marginal='box', **kwargs)
            elif kind == 'box':
                fig = px.box(pdf, x=by, y=column, **kwargs) if by else px.box(pdf, y=column)
            elif kind == 'violin':
                fig = px.violin(pdf, x=by, y=column, box=True, **kwargs) if by else px.violin(pdf, y=column)
            fig.update_layout(title=f'Distribution of {column}', template='plotly_white')
            fig.show()
        else:
            plt.figure(figsize=(10, 6))
            if kind == 'hist':
                if by:
                    for category in pdf[by].unique():
                        subset = pdf[pdf[by] == category]
                        plt.hist(subset[column], alpha=0.5, label=str(category), **kwargs)
                    plt.legend()
                else:
                    plt.hist(pdf[column], edgecolor='black', alpha=0.7, **kwargs)
            elif kind == 'box':
                if by:
                    pdf.boxplot(column=column, by=by, **kwargs)
                else:
                    pdf.boxplot(column=column, **kwargs)
            elif kind == 'violin':
                if by:
                    sns.violinplot(data=pdf, x=by, y=column, **kwargs)
                else:
                    sns.violinplot(data=pdf, y=column, **kwargs)
            
            plt.title(f'Distribution of {column}')
            plt.tight_layout()
            plt.show()

    def help(self) -> None:
        """Display help for ComputationalStats methods."""
        text = """
ComputationalStats — regression, interpolation, bootstrapping, clustering.

Methods:
  .regression(X, y, degree=1, interaction_terms=False)
  .linear_regression(X, y)
  .polynomial_regression(X, y, degree=2)
  .find_best_degree(X, y, max_degree=5, metric='r2')
  .interpolation(points, method='lagrange', spline_degree=3)
  .bootstrapping(column, n_samples=1000, statistic='mean', confidence_level=0.95)
  .k_means(k, max_iters=100, init_method='kmeans++')
  .elbow_method(max_k=10)
  .correlation_analysis(method='pearson')
  .plot_correlation_heatmap(method='pearson')
  .descriptive_statistics(by=None)
  .plot_distribution(column, by=None, kind='hist')

Accepts pandas or polars DataFrames via Backend.
"""
        print(text)
