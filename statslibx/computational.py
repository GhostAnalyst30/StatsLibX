from itertools import combinations
from typing import Union, Optional, Literal, List, Tuple, Any, Dict, Callable
import pandas as pd
import numpy as np
import sympy as sp
from scipy import stats as scipy_stats
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import logging

from .backend import Backend
from ._stats_utils import vectorized_bootstrap

logger = logging.getLogger(__name__)


def _viewx_export_mixin():
    """Lazy import to keep ViewX optional and avoid circular imports."""
    from statslibx.viewx.export import ViewXExportMixin
    return ViewXExportMixin


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _sns():
    import seaborn as sns
    return sns


def _px():
    import plotly.express as px
    return px


def _go():
    import plotly.graph_objects as go
    return go


def _make_subplots():
    from plotly.subplots import make_subplots
    return make_subplots


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
class RegressionResult(_viewx_export_mixin(), BaseResult):
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
        """Compute regression metrics and classical OLS inference."""
        n = self.n_samples
        p = len(self.coefficients) - 1

        # Basic metrics
        self.mse = np.mean(self.residuals ** 2)
        self.rmse = np.sqrt(self.mse)
        self.mae = np.mean(np.abs(self.residuals))
        with np.errstate(divide="ignore", invalid="ignore"):
            nonzero = self.y_values != 0
            self.mape = (
                float(np.mean(np.abs(self.residuals[nonzero] / self.y_values[nonzero])) * 100)
                if nonzero.any() else float("nan")
            )

        # R-squared and adjusted R-squared
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.y_values - np.mean(self.y_values)) ** 2)
        self.r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        self.r2_adj = 1 - (1 - self.r2) * (n - 1) / (n - p - 1) if n > p + 1 else self.r2

        # Information criteria (Gaussian log-likelihood up to a constant)
        self.aic = n * np.log(self.mse) + 2 * (p + 1)
        self.bic = n * np.log(self.mse) + (p + 1) * np.log(n)

        # Standard errors and t-statistics using the unbiased residual
        # variance sigma^2 = SSE / (n - p - 1); using SSE/n would bias
        # the standard errors (and p-values) downward.
        dof = n - p - 1
        try:
            if dof <= 0:
                raise np.linalg.LinAlgError
            sigma2 = ss_res / dof
            XTX_inv = np.linalg.pinv(self.X_design.T @ self.X_design)
            self.std_errors = np.sqrt(np.clip(np.diag(XTX_inv) * sigma2, 0, None))
            with np.errstate(divide="ignore", invalid="ignore"):
                self.t_stats = self.coefficients / self.std_errors
            self.p_values = 2 * scipy_stats.t.sf(np.abs(self.t_stats), dof)
        except np.linalg.LinAlgError:
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

    def __repr__(self):
        from .formatting import format_number, format_pvalue
        s = self.summary()
        lines = [
            "=" * 80,
            "REGRESSION RESULT".center(80),
            "=" * 80,
            f"Formula: {s['model_info']['formula']}",
            f"n={s['model_info']['n_samples']}  degree={s['model_info']['degree']}  "
            f"features={s['model_info']['n_features']}",
            "-" * 80,
            f"{'R²':<20} {format_number(self.r2)}",
            f"{'Adj. R²':<20} {format_number(self.r2_adj)}",
            f"{'RMSE':<20} {format_number(self.rmse)}",
            f"{'AIC':<20} {format_number(self.aic)}",
            f"{'BIC':<20} {format_number(self.bic)}",
            "-" * 80,
            f"{'Term':<20} {'Coef':>12} {'Std Err':>12} {'t':>10} {'P>|t|':>12}",
            "-" * 80,
        ]
        for name, coef, se, t, p in zip(
            self.all_feature_names, self.coefficients, self.std_errors, self.t_stats, self.p_values
        ):
            lines.append(
                f"{str(name):<20} {format_number(coef):>12} {format_number(se):>12} "
                f"{format_number(t, 3):>10} {format_pvalue(p):>12}"
            )
        lines.append("=" * 80)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown, format_number
        s = self.summary()
        coef = s["coefficients"].reset_index().rename(columns={"index": "term"})
        return records_to_markdown(coef.to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        s = self.summary()
        payload = {
            "model_info": s["model_info"],
            "metrics": s["metrics"],
            "coefficients": s["coefficients"].reset_index().rename(
                columns={"index": "term"}
            ).to_dict(orient="records"),
            "latex": str(s["formula"].get("latex")),
        }
        return dumps_json(payload, indent=indent)
    
    def plot(self, plot_type: Literal['scatter', 'residuals', 'qq', 'all'] = 'all', 
             interactive: bool = False, **kwargs):
        """Plot regression results"""
        plt = _plt()
        go = _go()

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
        plt = _plt()
        go = _go()
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
class MonteCarloResult(_viewx_export_mixin(), BaseResult):
    """Result of a Monte Carlo simulation."""
    name: str = "Monte Carlo"
    simulations: np.ndarray = field(default_factory=lambda: np.array([]))
    point_estimate: float = 0.0
    confidence: float = 0.95
    params: Optional[Dict] = None
    extra: Optional[Dict] = None

    def __post_init__(self):
        BaseResult.__init__(self, self.name)
        if self.params is None:
            self.params = {}
        if self.extra is None:
            self.extra = {}
        alpha = (1 - self.confidence) / 2
        self.ci = (
            float(np.quantile(self.simulations, alpha)),
            float(np.quantile(self.simulations, 1 - alpha)),
        )
        self.mean = float(np.mean(self.simulations))
        self.std = float(np.std(self.simulations))
        self._fitted = True

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "point_estimate": self.point_estimate,
            "simulation_mean": self.mean,
            "simulation_std": self.std,
            "ci": self.ci,
            "confidence": self.confidence,
            "n_simulations": len(self.simulations),
            "params": self.params,
            "extra": self.extra,
        }

    def plot(self, **kwargs):
        plt = _plt()
        plt.hist(self.simulations, bins=40, alpha=0.75, edgecolor="black")
        plt.axvline(self.point_estimate, color="red", linestyle="--", label="point estimate")
        plt.axvline(self.ci[0], color="green", linestyle=":", label="CI")
        plt.axvline(self.ci[1], color="green", linestyle=":")
        plt.title(self.name)
        plt.legend()
        plt.show()

    def to_dict(self) -> Dict:
        s = self.summary()
        s["simulations"] = self.simulations.tolist()
        return s

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown, format_ci
        return records_to_markdown([{
            "name": self.name,
            "point_estimate": self.point_estimate,
            "mean": self.mean,
            "std": self.std,
            "ci": format_ci(*self.ci, self.confidence),
        }])

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)

    def __repr__(self):
        from .formatting import format_number, format_ci
        return "\n".join([
            "=" * 60,
            self.name.center(60),
            "=" * 60,
            f"{'Point estimate':<30} {format_number(self.point_estimate)}",
            f"{'Sim mean':<30} {format_number(self.mean)}",
            f"{'Sim std':<30} {format_number(self.std)}",
            f"{'CI':<30} {format_ci(*self.ci, self.confidence)}",
            f"{'n_simulations':<30} {len(self.simulations)}",
            "=" * 60,
        ])


@dataclass
class JackknifeResult(_viewx_export_mixin(), BaseResult):
    """Jackknife bias / SE estimate."""
    statistic: str = "mean"
    point_estimate: float = 0.0
    bias: float = 0.0
    std_error: float = 0.0
    leave_one_out: np.ndarray = field(default_factory=lambda: np.array([]))
    params: Optional[Dict] = None

    def __post_init__(self):
        BaseResult.__init__(self, "Jackknife")
        if self.params is None:
            self.params = {}
        self._fitted = True

    def summary(self) -> Dict:
        return {
            "statistic": self.statistic,
            "point_estimate": self.point_estimate,
            "bias": self.bias,
            "std_error": self.std_error,
            "params": self.params,
        }

    def plot(self, **kwargs):
        plt = _plt()
        plt.hist(self.leave_one_out, bins=30, alpha=0.75, edgecolor="black")
        plt.axvline(self.point_estimate, color="red", linestyle="--")
        plt.title(f"Jackknife leave-one-out ({self.statistic})")
        plt.show()

    def to_dict(self) -> Dict:
        s = self.summary()
        s["leave_one_out"] = self.leave_one_out.tolist()
        return s

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown([self.summary()])

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)

    def __repr__(self):
        from .formatting import format_number
        return "\n".join([
            "=" * 60,
            f"JACKKNIFE ({self.statistic})".center(60),
            "=" * 60,
            f"{'Point estimate':<30} {format_number(self.point_estimate)}",
            f"{'Bias':<30} {format_number(self.bias)}",
            f"{'Std. error':<30} {format_number(self.std_error)}",
            "=" * 60,
        ])


@dataclass
class BootstrappingResult(_viewx_export_mixin(), BaseResult):
    """
    Bootstrap resampling result.

    Attributes
    ----------
    bootstrap_stats : numpy.ndarray
        Bootstrap replicates of the statistic.
    original_stat : float
        Statistic on the original sample.
    bias : float
        Bootstrap bias estimate (mean of replicates minus original).
    std_error : float
        Bootstrap standard error (sample std of replicates, ddof=1).
    percentile_ci, basic_ci, normal_ci, bca_ci : tuple of float
        Confidence intervals. BCa (bias-corrected and accelerated) is
        the recommended default (Efron, 1987).
    """
    data: np.ndarray
    n_samples: int = 1000
    statistic: Literal['mean', 'median', 'std', 'custom'] = 'mean'
    confidence_level: float = 0.95
    custom_func: Optional[callable] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        super().__init__("Bootstrapping")
        self._compute_bootstrap()
        self._compute_confidence_intervals()
        self._fitted = True

    def _stat_func(self):
        if self.statistic == 'mean':
            return np.mean
        elif self.statistic == 'median':
            return np.median
        elif self.statistic == 'std':
            return np.std
        elif self.statistic == 'custom' and self.custom_func is not None:
            return self.custom_func
        raise ValueError(f"Unknown statistic: {self.statistic}")

    def _compute_bootstrap(self):
        """Perform vectorized bootstrap resampling."""
        stat_func = self._stat_func()

        self.bootstrap_stats = vectorized_bootstrap(
            self.data, stat_func,
            n_resamples=self.n_samples,
            random_state=self.random_state,
        )
        self.original_stat = float(stat_func(self.data))
        self.bias = float(np.mean(self.bootstrap_stats) - self.original_stat)
        self.std_error = float(np.std(self.bootstrap_stats, ddof=1))

    def _compute_confidence_intervals(self):
        """Compute percentile, basic, normal and BCa intervals."""
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        self.percentile_ci = (
            float(np.percentile(self.bootstrap_stats, lower_percentile)),
            float(np.percentile(self.bootstrap_stats, upper_percentile))
        )

        self.basic_ci = (
            2 * self.original_stat - self.percentile_ci[1],
            2 * self.original_stat - self.percentile_ci[0]
        )

        self.normal_ci = (
            self.original_stat - scipy_stats.norm.ppf(1 - alpha/2) * self.std_error,
            self.original_stat + scipy_stats.norm.ppf(1 - alpha/2) * self.std_error
        )

        self.bca_ci = self._compute_bca_ci(alpha)

    def _compute_bca_ci(self, alpha: float):
        """
        Bias-corrected and accelerated (BCa) interval.

        The bias correction z0 comes from the fraction of replicates
        below the original statistic; the acceleration a comes from a
        jackknife of the statistic (Efron, 1987).
        """
        stat_func = self._stat_func()
        boots = self.bootstrap_stats
        n = len(self.data)

        prop_less = np.mean(boots < self.original_stat)
        if prop_less <= 0 or prop_less >= 1:
            # Degenerate bootstrap distribution; fall back to percentile.
            return self.percentile_ci
        z0 = scipy_stats.norm.ppf(prop_less)

        # Jackknife acceleration
        leave_one = np.empty(n, dtype=float)
        for i in range(n):
            leave_one[i] = stat_func(np.delete(self.data, i))
        jack_mean = leave_one.mean()
        diffs = jack_mean - leave_one
        denom = 6.0 * (np.sum(diffs ** 2) ** 1.5)
        a = float(np.sum(diffs ** 3) / denom) if denom > 0 else 0.0

        z_lo = scipy_stats.norm.ppf(alpha / 2)
        z_hi = scipy_stats.norm.ppf(1 - alpha / 2)

        def adjusted_quantile(z_alpha):
            adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
            return scipy_stats.norm.cdf(adj)

        q_lo = adjusted_quantile(z_lo)
        q_hi = adjusted_quantile(z_hi)
        return (
            float(np.quantile(boots, q_lo)),
            float(np.quantile(boots, q_hi)),
        )

    def summary(self) -> Dict:
        """Return bootstrapping summary."""
        return {
            'original_statistic': self.original_stat,
            'bootstrap_mean': float(np.mean(self.bootstrap_stats)),
            'bias': self.bias,
            'std_error': self.std_error,
            f'confidence_interval_{self.confidence_level*100:.0f}%': {
                'percentile': self.percentile_ci,
                'basic': self.basic_ci,
                'normal': self.normal_ci,
                'bca': self.bca_ci,
            },
            'distribution': self.bootstrap_stats
        }

    def to_dict(self) -> Dict:
        s = self.summary()
        s = dict(s)
        s['distribution'] = self.bootstrap_stats.tolist()
        return s

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown, format_ci
        row = {
            'statistic': self.statistic,
            'original': self.original_stat,
            'bias': self.bias,
            'std_error': self.std_error,
            'percentile_ci': format_ci(*self.percentile_ci, self.confidence_level),
        }
        return records_to_markdown([row])

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)

    def __repr__(self):
        from .formatting import format_number, format_ci
        lines = [
            "=" * 70,
            f"BOOTSTRAP ({self.statistic})".center(70),
            "=" * 70,
            f"{'Original':<30} {format_number(self.original_stat)}",
            f"{'Bootstrap mean':<30} {format_number(np.mean(self.bootstrap_stats))}",
            f"{'Bias':<30} {format_number(self.bias)}",
            f"{'Std. error':<30} {format_number(self.std_error)}",
            f"{'BCa CI (recommended)':<30} {format_ci(*self.bca_ci, self.confidence_level)}",
            f"{'Percentile CI':<30} {format_ci(*self.percentile_ci, self.confidence_level)}",
            f"{'Basic CI':<30} {format_ci(*self.basic_ci, self.confidence_level)}",
            f"{'Normal CI':<30} {format_ci(*self.normal_ci, self.confidence_level)}",
            f"{'n_samples':<30} {self.n_samples}",
            "=" * 70,
        ]
        return "\n".join(lines)
    
    def plot(self, interactive: bool = False, **kwargs):
        """Plot bootstrap distribution"""
        plt = _plt()
        go = _go()
        make_subplots = _make_subplots()
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
            # Reference line: theoretical quantiles mapped through the
            # bootstrap mean and standard deviation.
            boot_mean = float(np.mean(self.bootstrap_stats))
            boot_std = float(np.std(self.bootstrap_stats, ddof=1))
            line_x = np.array([-3.0, 3.0])
            fig.add_trace(
                go.Scatter(x=line_x, y=boot_mean + boot_std * line_x,
                           mode='lines', name='Normal reference', line=dict(dash='dash')),
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


class ComputationalStats:
    """
    Computational statistics: regression, interpolation, resampling
    (bootstrap, jackknife, Monte Carlo, permutation), cross-validation
    and clustering, over a pandas or polars DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame, polars.DataFrame or numpy.ndarray
        Dataset to analyze.
    backend : {'pandas', 'polars'}, optional
        Data engine. Auto-detected when None.
    seed : int, optional
        Seed for all stochastic methods. Every method also accepts a
        ``random_state`` argument that overrides the instance seed for
        that single call.

    Examples
    --------
    >>> from statslibx import ComputationalStats
    >>> from statslibx.datasets import load_iris
    >>> cs = ComputationalStats(load_iris(), seed=42)
    >>> boot = cs.bootstrap("sepal_length", n_samples=2000)
    >>> boot.bca_ci  # doctest: +SKIP
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        backend: Optional[Literal["pandas", "polars"]] = None,
        seed: Optional[int] = None,
        lang: Optional[str] = None,
    ):
        if lang is not None:
            warnings.warn(
                "The 'lang' parameter is deprecated and ignored; "
                "all output is in English.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._backend = Backend(data, backend=backend)
        self.data = self._backend.df

        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._numeric_cols = self._backend.numeric_columns()
        self._categorical_cols = self._backend.categorical_columns()

    def _rng_for(self, random_state: Optional[int]) -> np.random.Generator:
        """Per-call RNG: explicit random_state wins, else the instance RNG."""
        if random_state is not None:
            return np.random.default_rng(random_state)
        return self._rng

    def _seed_for(self, random_state: Optional[int]) -> Optional[int]:
        """
        Integer seed for APIs that take a seed instead of a Generator.
        Derives a child seed from the instance RNG so the constructor
        seed makes results reproducible.
        """
        if random_state is not None:
            return random_state
        if self.seed is not None:
            return int(self._rng.integers(0, 2**31 - 1))
        return None

    @classmethod
    def from_file(
        cls,
        path: str,
        backend: str = "pandas",
        sep: str = ",",
        seed: Optional[int] = None,
        lang: Optional[str] = None,
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
        """Active data engine ('pandas' or 'polars')."""
        return self._backend.type

    @property
    def backend_engine(self) -> Backend:
        """Internal Backend wrapper."""
        return self._backend
    
    def regression(self, X: Union[List[str], str], y: str,
                   degree: int = 1, interaction_terms: bool = False,
                   cv_folds: Optional[int] = None,
                   random_state: Optional[int] = None) -> Union[RegressionResult, Dict]:
        """
        Ordinary least squares regression, optionally polynomial or with
        pairwise interaction terms.

        Parameters
        ----------
        X : str or list of str
            Independent variable(s).
        y : str
            Dependent variable.
        degree : int, default 1
            Polynomial degree (single predictor only; ignored with a
            warning for multiple predictors).
        interaction_terms : bool, default False
            Include pairwise interaction terms (multiple predictors).
        cv_folds : int, optional
            When set, also run k-fold cross-validation and return
            ``{'model': ..., 'cv': ...}``.
        random_state : int, optional
            Seed for the CV shuffling.

        Returns
        -------
        RegressionResult or dict
        """
        pdf = self._backend.to_pandas()
        x_cols = [X] if isinstance(X, str) else list(X)
        if degree > 1 and len(x_cols) > 1:
            warnings.warn(
                "Polynomial degree > 1 is only applied for a single predictor; "
                "fitting a linear model in the given predictors instead.",
                UserWarning,
                stacklevel=2,
            )
        X_data = pdf[x_cols]
        y_data = pdf[y]

        model = RegressionResult(X_data, y_data, degree=degree, interaction_terms=interaction_terms)
        if cv_folds is None:
            return model
        cv = self.k_fold_cv(X, y, n_folds=cv_folds, degree=degree, random_state=random_state)
        return {"model": model, "cv": cv}

    def linear_regression(self, X: Union[List[str], str], y: str) -> RegressionResult:
        """Linear regression (wrapper for :meth:`regression` with degree=1)."""
        return self.regression(X, y, degree=1)

    def polynomial_regression(self, X: str, y: str, degree: int = 2) -> RegressionResult:
        """Polynomial regression on a single predictor."""
        return self.regression(X, y, degree=degree)

    def bootstrap_regression(
        self,
        X: Union[List[str], str],
        y: str,
        n_samples: int = 1000,
        degree: int = 1,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> Dict:
        """
        Bootstrap the regression coefficients by resampling rows
        (case resampling).

        Parameters
        ----------
        X : str or list of str
            Independent variable(s).
        y : str
            Dependent variable.
        n_samples : int, default 1000
            Number of bootstrap resamples.
        degree : int, default 1
            Polynomial degree (single predictor only).
        confidence_level : float, default 0.95
            Level for the percentile confidence intervals.
        random_state : int, optional
            Seed (overrides the instance seed for this call).

        Returns
        -------
        dict
            'coefficients': DataFrame with one row per term (estimate,
            bootstrap SE and percentile CI); 'distributions': array of
            shape (n_samples, n_terms) with the replicates.
        """
        pdf = self._backend.to_pandas()
        x_cols = [X] if isinstance(X, str) else list(X)
        data = pdf[x_cols + [y]].dropna()
        n = len(data)
        rng = self._rng_for(random_state)

        base_model = RegressionResult(data[x_cols], data[y], degree=degree)
        n_terms = len(base_model.coefficients)
        draws = np.empty((n_samples, n_terms), dtype=float)
        for i in range(n_samples):
            idx = rng.integers(0, n, size=n)
            sample = data.iloc[idx]
            model = RegressionResult(sample[x_cols], sample[y], degree=degree)
            draws[i] = model.coefficients

        alpha = 1 - confidence_level
        lower = np.percentile(draws, alpha / 2 * 100, axis=0)
        upper = np.percentile(draws, (1 - alpha / 2) * 100, axis=0)
        table = pd.DataFrame({
            "term": base_model.all_feature_names,
            "estimate": base_model.coefficients,
            "boot_se": draws.std(axis=0, ddof=1),
            "ci_lower": lower,
            "ci_upper": upper,
        })
        return {"coefficients": table, "distributions": draws,
                "n_samples": n_samples, "confidence_level": confidence_level}

    def find_best_degree(self, X: str, y: str, max_degree: int = 5,
                         metric: Literal['cv_rmse', 'cv_r2', 'aic', 'bic', 'r2'] = 'cv_rmse',
                         n_folds: int = 5,
                         random_state: Optional[int] = None) -> Dict:
        """
        Select the polynomial degree by cross-validation (default) or an
        information criterion.

        Parameters
        ----------
        X : str
            Independent variable.
        y : str
            Dependent variable.
        max_degree : int, default 5
            Highest degree to evaluate.
        metric : {'cv_rmse', 'cv_r2', 'aic', 'bic', 'r2'}, default 'cv_rmse'
            Selection criterion. 'cv_*' use k-fold cross-validation;
            'r2' is in-sample and systematically favors higher degrees
            (kept for reference only, a warning is issued).
        n_folds : int, default 5
            Folds for the CV metrics.
        random_state : int, optional
            Seed for the CV shuffling.

        Returns
        -------
        dict
            Best entry (degree, metrics, fitted model) plus
            'all_results' with every evaluated degree.
        """
        if metric == 'r2':
            warnings.warn(
                "metric='r2' is computed in-sample and always favors higher "
                "degrees (overfitting); prefer 'cv_rmse' or 'cv_r2'.",
                UserWarning,
                stacklevel=2,
            )

        seed = self._seed_for(random_state)
        results = []
        for degree in range(1, max_degree + 1):
            model = self.regression(X, y, degree=degree)
            metrics = model.summary()['metrics']
            entry = {
                'degree': degree,
                'r2': metrics['R2'],
                'adj_r2': metrics['Adjusted R2'],
                'aic': metrics['AIC'],
                'bic': metrics['BIC'],
                'rmse': metrics['RMSE'],
                'model': model,
            }
            if metric in ('cv_rmse', 'cv_r2'):
                cv = self.k_fold_cv(X, y, n_folds=n_folds, degree=degree, random_state=seed)
                entry['cv_r2'] = cv['mean_r2']
                entry['cv_rmse'] = cv['mean_rmse']
            results.append(entry)

        if metric in ('aic', 'bic', 'cv_rmse'):
            best_idx = int(np.argmin([r[metric] for r in results]))
        else:  # r2, cv_r2
            best_idx = int(np.argmax([r[metric] for r in results]))

        best_result = dict(results[best_idx])
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
    
    def bootstrap(self, column: str, n_samples: int = 1000,
                  statistic: Literal['mean', 'median', 'std', 'custom'] = 'mean',
                  confidence_level: float = 0.95,
                  custom_func: Optional[callable] = None,
                  random_state: Optional[int] = None) -> BootstrappingResult:
        """
        Bootstrap a statistic of a column.

        Parameters
        ----------
        column : str
            Numeric column to resample (NaN dropped).
        n_samples : int, default 1000
            Number of bootstrap resamples.
        statistic : {'mean', 'median', 'std', 'custom'}, default 'mean'
            Statistic to bootstrap.
        confidence_level : float, default 0.95
            Level for the confidence intervals.
        custom_func : callable, optional
            Custom statistic (used with ``statistic='custom'``).
        random_state : int, optional
            Seed (overrides the instance seed for this call).

        Returns
        -------
        BootstrappingResult
            Includes percentile, basic, normal and BCa intervals; BCa
            is the recommended interval (Efron, 1987).

        References
        ----------
        Efron, B. (1987). Better bootstrap confidence intervals.
        JASA 82, 171-185.
        """
        data = self._backend.col_clean(column)
        return BootstrappingResult(
            data, n_samples, statistic, confidence_level, custom_func,
            random_state=self._seed_for(random_state),
        )

    def bootstrapping(self, column: str, n_samples: int = 1000,
                     statistic: Literal['mean', 'median', 'std', 'custom'] = 'mean',
                     confidence_level: float = 0.95,
                     custom_func: Optional[callable] = None,
                     random_state: Optional[int] = None) -> BootstrappingResult:
        """Deprecated alias of :meth:`bootstrap`."""
        warnings.warn(
            "bootstrapping() is deprecated; use bootstrap() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.bootstrap(
            column, n_samples=n_samples, statistic=statistic,
            confidence_level=confidence_level, custom_func=custom_func,
            random_state=random_state,
        )

    def k_means(self, k: int, max_iters: int = 100,
                init_method: Literal['random', 'kmeans++'] = 'kmeans++',
                engine: Literal['native', 'sklearn'] = 'native',
                random_state: Optional[int] = None) -> Dict:
        """
        K-means clustering on all numeric columns.

        Parameters
        ----------
        k : int
            Number of clusters.
        max_iters : int, default 100
            Maximum iterations.
        init_method : {'kmeans++', 'random'}, default 'kmeans++'
            Centroid initialization.
        engine : {'native', 'sklearn'}, default 'native'
            'sklearn' requires scikit-learn and uses 10 restarts.
        random_state : int, optional
            Seed (overrides the instance seed for this call).

        Returns
        -------
        dict
            Keys: centroids, labels, inertia, silhouette, n_iterations,
            engine.
        """
        X = np.column_stack([self._backend.col_numpy(c) for c in self._numeric_cols])
        # Drop rows with NaN
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]

        if engine == 'sklearn':
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
            except ImportError as e:
                raise ImportError(
                    "engine='sklearn' requires scikit-learn. "
                    "pip install statslibx[sklearn]"
                ) from e
            km = KMeans(
                n_clusters=k,
                init='k-means++' if init_method == 'kmeans++' else 'random',
                max_iter=max_iters,
                n_init=10,
                random_state=self._seed_for(random_state),
            )
            labels = km.fit_predict(X)
            silhouette = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else -1.0
            return {
                'centroids': km.cluster_centers_,
                'labels': labels,
                'inertia': float(km.inertia_),
                'silhouette': silhouette,
                'n_iterations': int(km.n_iter_),
                'engine': 'sklearn',
            }

        # Native implementation
        rng = self._rng_for(random_state)
        if init_method == 'kmeans++':
            centroids = [X[rng.integers(len(X))]]
            for _ in range(1, k):
                distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
                probabilities = distances / np.sum(distances)
                next_centroid = X[rng.choice(len(X), p=probabilities)]
                centroids.append(next_centroid)
            centroids = np.array(centroids)
        else:  # random
            centroids = X[rng.choice(len(X), k, replace=False)]
        
        for iteration in range(max_iters):
            distances = np.linalg.norm(X[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 
                                      else centroids[i] for i in range(k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        inertia = np.sum([np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(k)])
        
        if len(np.unique(labels)) > 1:
            try:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(X, labels)
            except ImportError:
                silhouette = -1
        else:
            silhouette = -1
        
        return {
            'centroids': centroids,
            'labels': labels,
            'inertia': inertia,
            'silhouette': silhouette,
            'n_iterations': iteration + 1,
            'engine': 'native',
        }
    
    def elbow_method(self, max_k: int = 10, random_state: Optional[int] = None) -> Dict:
        """
        Elbow diagnostics for choosing the number of K-means clusters.

        Parameters
        ----------
        max_k : int, default 10
            Largest k to evaluate (starts at 2).
        random_state : int, optional
            Seed (overrides the instance seed for this call).

        Returns
        -------
        dict
            Keys: k_values, inertias, silhouettes.
        """
        seed = self._seed_for(random_state)
        inertias = []
        silhouettes = []

        for k in range(2, max_k + 1):
            result = self.k_means(k, random_state=seed)
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
        """
        corr_matrix = self._backend.corr(method=method)
        
        p_values = None
        if method == 'pearson':
            cols = self._numeric_cols
            n_cols = len(cols)
            # Build numeric matrix once
            mat = np.column_stack([self._backend.col_numpy(c) for c in cols]).astype(float)
            # Pairwise complete: use scipy on matrix via loop but cached columns
            p_values = pd.DataFrame(np.ones((n_cols, n_cols)), index=cols, columns=cols)
            for i in range(n_cols):
                for j in range(i, n_cols):
                    a, b = mat[:, i], mat[:, j]
                    mask = ~(np.isnan(a) | np.isnan(b))
                    if mask.sum() < 3:
                        p = 1.0
                    else:
                        _, p = scipy_stats.pearsonr(a[mask], b[mask])
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
        """
        plt = _plt()
        sns = _sns()
        px = _px()
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
        Deprecated: use :class:`statslibx.DescriptiveStats` (``summary()``
        and ``summary_by()``) instead.
        """
        warnings.warn(
            "ComputationalStats.descriptive_statistics() is deprecated; "
            "use DescriptiveStats.summary() / DescriptiveStats.summary_by() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        pdf = self._backend.to_pandas()
        if by is None or by not in self._categorical_cols:
            return pdf[self._numeric_cols].describe()
        return pdf.groupby(by)[self._numeric_cols].describe()
    
    def plot_distribution(self, column: str, by: Optional[str] = None, 
                          kind: Literal['hist', 'box', 'violin'] = 'hist',
                          interactive: bool = False, **kwargs):
        """Plot distribution of a column"""
        plt = _plt()
        sns = _sns()
        px = _px()
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

    # ============= MONTE CARLO / JACKKNIFE / CV =============

    def monte_carlo_mean(
        self,
        column: str,
        n_simulations: int = 10000,
        sample_size: Optional[int] = None,
        confidence: float = 0.95,
        random_state: Optional[int] = None,
    ) -> 'MonteCarloResult':
        """
        Simulate the sampling distribution of the mean by resampling the
        empirical distribution (a bootstrap of the mean).

        Parameters
        ----------
        column : str
            Numeric column (NaN dropped).
        n_simulations : int, default 10000
            Number of simulated samples.
        sample_size : int, optional
            Size of each simulated sample (defaults to the data size).
        confidence : float, default 0.95
            Level for the percentile interval.
        random_state : int, optional
            Seed (overrides the instance seed for this call).

        Returns
        -------
        MonteCarloResult
        """
        vals = self._backend.col_clean(column)
        n = sample_size or len(vals)
        rng = self._rng_for(random_state)
        # Resample from empirical distribution
        draws = rng.choice(vals, size=(n_simulations, n), replace=True)
        sim = draws.mean(axis=1)
        return MonteCarloResult(
            name="Monte Carlo Mean",
            simulations=sim,
            point_estimate=float(np.mean(vals)),
            confidence=confidence,
            params={"column": column, "n": n, "n_simulations": n_simulations},
        )

    def monte_carlo_regression(
        self,
        X: Union[List[str], str],
        y: str,
        n_simulations: int = 1000,
        confidence: float = 0.95,
        random_state: Optional[int] = None,
    ) -> 'MonteCarloResult':
        """
        Monte Carlo prediction bands by resampling regression residuals.

        Fits an OLS model and simulates new responses as
        ``fitted + resampled residual``; the ``extra['prediction_bands']``
        entry holds pointwise percentile bands.
        """
        model = self.regression(X, y, degree=1)
        resid = model.residuals
        rng = self._rng_for(random_state)
        # Simulate new responses = fitted + resampled residual
        preds = []
        fitted = model.y_pred
        for _ in range(n_simulations):
            noise = rng.choice(resid, size=len(resid), replace=True)
            preds.append(fitted + noise)
        sims = np.asarray(preds)
        # Store mean predicted path distribution at each point — use overall mean of sims
        return MonteCarloResult(
            name="Monte Carlo Regression",
            simulations=sims.mean(axis=1),
            point_estimate=float(np.mean(fitted)),
            confidence=confidence,
            params={
                "X": X, "y": y,
                "n_simulations": n_simulations,
                "rmse": float(model.rmse),
            },
            extra={"prediction_bands": np.percentile(
                sims, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100], axis=0
            ).tolist()},
        )

    def simulate_distribution(
        self,
        dist: Literal['normal', 't', 'chi2', 'binomial', 'uniform'] = 'normal',
        n_simulations: int = 10000,
        size: int = 100,
        confidence: float = 0.95,
        random_state: Optional[int] = None,
        **dist_params,
    ) -> 'MonteCarloResult':
        """
        Simulate the sampling distribution of the mean of a named
        distribution.

        Parameters
        ----------
        dist : {'normal', 't', 'chi2', 'binomial', 'uniform'}, default 'normal'
            Distribution to sample from.
        n_simulations : int, default 10000
            Number of simulated samples.
        size : int, default 100
            Size of each simulated sample.
        confidence : float, default 0.95
            Level for the percentile interval.
        random_state : int, optional
            Seed (overrides the instance seed for this call).
        **dist_params
            Distribution parameters (loc/scale, df, n/p, low/high).

        Returns
        -------
        MonteCarloResult
        """
        rng = self._rng_for(random_state)
        if dist == 'normal':
            loc = dist_params.get('loc', 0.0)
            scale = dist_params.get('scale', 1.0)
            samples = rng.normal(loc, scale, size=(n_simulations, size))
        elif dist == 't':
            df = dist_params.get('df', 10)
            samples = rng.standard_t(df, size=(n_simulations, size))
        elif dist == 'chi2':
            df = dist_params.get('df', 5)
            samples = rng.chisquare(df, size=(n_simulations, size))
        elif dist == 'binomial':
            n = dist_params.get('n', 10)
            p = dist_params.get('p', 0.5)
            samples = rng.binomial(n, p, size=(n_simulations, size))
        elif dist == 'uniform':
            low = dist_params.get('low', 0.0)
            high = dist_params.get('high', 1.0)
            samples = rng.uniform(low, high, size=(n_simulations, size))
        else:
            raise ValueError(f"Unknown dist: {dist}")
        sim_means = samples.mean(axis=1)
        return MonteCarloResult(
            name=f"Simulate {dist}",
            simulations=sim_means,
            point_estimate=float(np.mean(sim_means)),
            confidence=confidence,
            params={"dist": dist, "size": size, "n_simulations": n_simulations, **dist_params},
        )

    def jackknife(
        self,
        column: str,
        statistic: Literal['mean', 'median', 'std'] = 'mean',
        d: int = 1,
        n_subsets: int = 1000,
        random_state: Optional[int] = None,
    ) -> 'JackknifeResult':
        """
        Jackknife estimate of bias and standard error.

        Parameters
        ----------
        column : str
            Numeric column (NaN dropped).
        statistic : {'mean', 'median', 'std'}, default 'mean'
            Statistic to jackknife.
        d : int, default 1
            Number of observations to delete per replicate. ``d=1`` is
            the exact leave-one-out jackknife; ``d>1`` (delete-d) uses
            random subsets and is recommended for non-smooth statistics
            such as the median.
        n_subsets : int, default 1000
            Number of random subsets for the delete-d jackknife.
        random_state : int, optional
            Seed for the delete-d subsets.

        Returns
        -------
        JackknifeResult

        References
        ----------
        Efron, B. & Tibshirani, R. (1993). An Introduction to the
        Bootstrap. Chapman & Hall. (Ch. 11 for delete-d.)
        """
        vals = self._backend.col_clean(column)
        n = len(vals)
        stat_fn = getattr(np, statistic)
        theta = float(stat_fn(vals))

        if d <= 1:
            leave_one = np.empty(n, dtype=float)
            for i in range(n):
                leave_one[i] = stat_fn(np.delete(vals, i))
            theta_bar = float(np.mean(leave_one))
            bias = (n - 1) * (theta_bar - theta)
            se = float(np.sqrt(((n - 1) / n) * np.sum((leave_one - theta_bar) ** 2)))
            replicates = leave_one
            params = {"column": column, "n": n, "d": 1}
        else:
            if d >= n:
                raise ValueError("d must be smaller than the sample size")
            rng = self._rng_for(random_state)
            replicates = np.empty(n_subsets, dtype=float)
            for i in range(n_subsets):
                keep = rng.choice(n, size=n - d, replace=False)
                replicates[i] = stat_fn(vals[keep])
            theta_bar = float(np.mean(replicates))
            bias = (n - d) / d * (theta_bar - theta) * d / (n - d)  # first-order
            se = float(np.sqrt((n - d) / (d * n_subsets) * np.sum((replicates - theta_bar) ** 2)))
            params = {"column": column, "n": n, "d": d, "n_subsets": n_subsets}

        return JackknifeResult(
            statistic=statistic,
            point_estimate=theta,
            bias=float(bias),
            std_error=se,
            leave_one_out=replicates,
            params=params,
        )

    def k_fold_cv(
        self,
        X: Union[List[str], str],
        y: str,
        n_folds: int = 5,
        degree: int = 1,
        stratify: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> Dict:
        """
        K-fold cross-validation for polynomial/linear regression.

        Parameters
        ----------
        X : str or list of str
            Independent variable(s).
        y : str
            Dependent variable.
        n_folds : int, default 5
            Number of folds.
        degree : int, default 1
            Polynomial degree (single predictor only).
        stratify : str, optional
            Categorical column; folds preserve its class proportions
            (stratified CV).
        random_state : int, optional
            Seed (overrides the instance seed for this call).

        Returns
        -------
        dict
            Keys: folds (per-fold r2/rmse/mae), mean_r2, mean_rmse,
            mean_mae, n_folds.
        """
        pdf = self._backend.to_pandas()
        x_cols = [X] if isinstance(X, str) else list(X)
        n = len(pdf)
        rng = self._rng_for(random_state)

        if stratify is not None:
            if stratify not in pdf.columns:
                raise ValueError(f"Stratification column '{stratify}' not found.")
            # Assign fold ids round-robin within each class after shuffling.
            fold_ids = np.empty(n, dtype=int)
            for _, idx in pdf.groupby(stratify).indices.items():
                idx = np.asarray(idx)
                shuffled = rng.permutation(idx)
                fold_ids[shuffled] = np.arange(len(shuffled)) % n_folds
            folds = [np.where(fold_ids == i)[0] for i in range(n_folds)]
        else:
            indices = rng.permutation(n)
            folds = np.array_split(indices, n_folds)

        scores = []
        for i in range(n_folds):
            test_idx = folds[i]
            if len(test_idx) == 0:
                continue
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            train = pdf.iloc[train_idx]
            test = pdf.iloc[test_idx]
            model = RegressionResult(train[x_cols], train[y], degree=degree)
            y_hat = model.predict(test[x_cols].values)
            y_true = test[y].values
            ss_res = np.sum((y_true - y_hat) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_hat)))
            scores.append({
                "fold": i + 1, "r2": float(r2), "rmse": rmse,
                "mae": mae, "n_test": len(test_idx),
            })
        return {
            "folds": scores,
            "mean_r2": float(np.mean([s["r2"] for s in scores])),
            "mean_rmse": float(np.mean([s["rmse"] for s in scores])),
            "mean_mae": float(np.mean([s["mae"] for s in scores])),
            "n_folds": n_folds,
        }

    def loo_cv(
        self,
        X: Union[List[str], str],
        y: str,
        degree: int = 1,
    ) -> Dict:
        """
        Leave-one-out cross-validation (k-fold with k = n).

        Deterministic: every observation is the test set exactly once.

        Returns
        -------
        dict
            Same structure as :meth:`k_fold_cv`.
        """
        n = len(self._backend.to_pandas())
        return self.k_fold_cv(X, y, n_folds=n, degree=degree, random_state=0)

    def bootstrap_validation(
        self,
        X: Union[List[str], str],
        y: str,
        n_bootstrap: int = 200,
        degree: int = 1,
        random_state: Optional[int] = None,
    ) -> Dict:
        """
        Out-of-bag bootstrap validation for regression.

        Fits on bootstrap resamples and evaluates on the out-of-bag
        rows; reports mean OOB R-squared and RMSE.
        """
        pdf = self._backend.to_pandas()
        x_cols = [X] if isinstance(X, str) else list(X)
        n = len(pdf)
        rng = self._rng_for(random_state)
        oob_r2, oob_rmse = [], []
        for _ in range(n_bootstrap):
            boot_idx = rng.integers(0, n, size=n)
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[np.unique(boot_idx)] = False
            if oob_mask.sum() < 2:
                continue
            train = pdf.iloc[boot_idx]
            test = pdf.iloc[oob_mask]
            model = RegressionResult(train[x_cols], train[y], degree=degree)
            y_hat = model.predict(test[x_cols].values)
            y_true = test[y].values
            ss_res = np.sum((y_true - y_hat) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            oob_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
            oob_rmse.append(float(np.sqrt(np.mean((y_true - y_hat) ** 2))))
        return {
            "mean_oob_r2": float(np.mean(oob_r2)) if oob_r2 else None,
            "mean_oob_rmse": float(np.mean(oob_rmse)) if oob_rmse else None,
            "n_bootstrap": n_bootstrap,
            "n_valid": len(oob_r2),
        }

    # ============= PERMUTATION / POWER (facade over InferentialStats) =============

    def permutation_test(
        self,
        column1: str,
        column2: str,
        statistic: Literal['mean', 'median'] = 'mean',
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        alpha: float = 0.05,
        n_permutations: int = 10000,
        random_state: Optional[int] = None,
    ):
        """
        Permutation test for the difference of means/medians.

        Thin facade over
        :meth:`statslibx.InferentialStats.permutation_test` using this
        instance's data and seed.
        """
        from .inferential import InferentialStats
        return InferentialStats(self.data, backend=self.backend).permutation_test(
            column1, column2, statistic=statistic, alternative=alternative,
            alpha=alpha, n_permutations=n_permutations,
            random_state=self._seed_for(random_state),
        )

    def power_ttest(self, *args, **kwargs):
        """
        Analytical t-test power. Facade over
        :meth:`statslibx.InferentialStats.power_ttest`.
        """
        from .inferential import InferentialStats
        return InferentialStats(self.data, backend=self.backend).power_ttest(*args, **kwargs)

    def help(self) -> None:
        """Print a quick reference of the ComputationalStats API."""
        text = """
================================================================================
ComputationalStats - quick reference
================================================================================
All stochastic methods honor the constructor seed; pass random_state to
override for a single call.

Regression:
  .regression(X, y, degree=1, interaction_terms=False, cv_folds=None)
  .linear_regression(X, y) / .polynomial_regression(X, y, degree=2)
  .bootstrap_regression(X, y, n_samples=1000)   Coefficient bootstrap CIs
  .find_best_degree(X, y, max_degree=5, metric='cv_rmse')

Resampling and simulation:
  .bootstrap(column, n_samples=1000, statistic='mean')
      -> BootstrappingResult (.bca_ci recommended, .percentile_ci,
         .basic_ci, .normal_ci, .std_error, .bias)
  .jackknife(column, statistic='mean', d=1)     Delete-1 or delete-d
  .monte_carlo_mean(column, n_simulations=10000)
  .monte_carlo_regression(X, y, n_simulations=1000)
  .simulate_distribution(dist='normal', n_simulations=10000, size=100)
  .permutation_test(col1, col2, statistic='mean')

Validation:
  .k_fold_cv(X, y, n_folds=5, stratify=None)    R2 / RMSE / MAE per fold
  .loo_cv(X, y)                                 Leave-one-out
  .bootstrap_validation(X, y, n_bootstrap=200)  Out-of-bag metrics

Clustering:
  .k_means(k, init_method='kmeans++', engine='native'|'sklearn')
  .elbow_method(max_k=10)

Other:
  .interpolation(points, method='lagrange'|'newton'|'spline')
  .correlation_analysis(method='pearson')       Matrix + p-values
  .power_ttest(effect_size, n=...)              Facade to InferentialStats

Loading:
  ComputationalStats.from_file(path, backend='pandas'|'polars', seed=...)
================================================================================
For details: help(ComputationalStats.<method>)
"""
        print(text)
