import { TrendingUp } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function TimeSeriesDocs() {
  return (
    <>
      <DocHeader
        title="TimeSeriesStats"
        description="Autocorrelation structure, stationarity testing (ADF + KPSS), classical decomposition, rolling statistics and lag features for chronologically ordered data."
        icon={<TrendingUp className="w-6 h-6" />}
        version="0.4.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          <code className="code-inline">TimeSeriesStats</code> analyses a single
          numeric column in chronological order. Stationarity is diagnosed with{" "}
          <strong>two tests with opposite null hypotheses</strong> — ADF (H₀:
          unit root) and KPSS (H₀: stationary) — so a robust verdict falls out
          of combining them. All statistics are implemented with numpy/scipy:
          no statsmodels dependency.
        </p>
        <div className="code-block mt-6">
          <CodeBlock
            title="Quick start"
            code={`from statslibx import TimeSeriesStats
import numpy as np

rng = np.random.default_rng(0)
t = np.arange(144)
df = pd.DataFrame({
    "value": 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 12) + rng.normal(scale=0.3, size=144),
})

ts = TimeSeriesStats(df, column="value")
ts.acf().is_white_noise        # False — clear seasonality
ts.stationarity_test()         # {'adf': ..., 'kpss': ...}
ts.decompose(period=12)         # trend / seasonal / residual
ts.summary()`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Methods</h2>
        <div className="method-list">
          <MethodCard
            name="acf"
            signature="acf(column=None, nlags=None, conf_level=0.95) -> AutocorrelationResult"
            description="Sample autocorrelation function for lags 0..nlags with white-noise confidence bands and a Ljung-Box portmanteau test for overall autocorrelation."
            returns="AutocorrelationResult"
            parameters={[
              { name: "nlags", type: "int, optional", description: "Number of lags (default min(10·log₁₀n, n−3))." },
              { name: "conf_level", type: "float", description: "Confidence level of the bands.", default: "0.95" },
            ]}
            example={`res = ts.acf(nlags=12)
print(res.ljung_box_pvalue)   # < 0.05 → autocorrelated
print(res.significant_lags)   # lags escaping the band
res.to_dataframe()`}
          />
          <MethodCard
            name="pacf"
            signature="pacf(column=None, nlags=None, conf_level=0.95) -> AutocorrelationResult"
            description="Partial autocorrelation function computed with the Durbin-Levinson recursion — the right tool for AR(p) order selection."
            returns="AutocorrelationResult"
            example={`res = ts.pacf(nlags=8)
# AR(p): pacf cuts off after lag p
print(res.values.round(3))`}
          />
          <MethodCard
            name="stationarity_test"
            signature="stationarity_test(column=None, method='both', regression='c', alpha=0.05, max_lag=None)"
            description="Augmented Dickey-Fuller (H₀: unit root) and/or KPSS (H₀: stationary). Published critical values are used (MacKinnon 1994; Kwiatkowski et al. 1992), so no statsmodels is required. ADF rejects + KPSS does not → stationary; ADF does not reject + KPSS rejects → difference the series."
            returns="StationarityResult or dict"
            parameters={[
              { name: "method", type: "{'adf', 'kpss', 'both'}", description: "Which test(s) to run.", default: "'both'" },
              { name: "regression", type: "{'c', 'ct', 'n'}", description: "ADF deterministic terms: constant, constant+trend, none.", default: "'c'" },
            ]}
            example={`both = ts.stationarity_test(method="both")
print(both["adf"].stationary)   # True  → no unit root
print(both["kpss"].stationary)  # False → level not stationary
print(both["kpss"].interpretation)`}
          />
          <MethodCard
            name="decompose"
            signature="decompose(column=None, period=None, model='additive') -> DecompositionResult"
            description="Classical decomposition into trend (centered moving average), seasonal (per-phase detrended averages, tiled) and residual components, for additive or multiplicative models."
            returns="DecompositionResult"
            parameters={[
              { name: "period", type: "int", description: "Seasonal period (e.g. 12 for monthly data with yearly seasonality). Must be between 2 and n//2. Required unless inferable from the index." },
              { name: "model", type: "{'additive', 'multiplicative'}", description: "y = trend + seasonal + resid or y = trend · seasonal · resid.", default: "'additive'" },
            ]}
            example={`decomp = ts.decompose(period=12)
print(decomp.seasonal_strength)  # 0..1
print(decomp.trend_strength)
decomp.to_dataframe().head()
decomp.plot()  # stacked component panels (matplotlib)`}
          />
          <MethodCard
            name="rolling_stats"
            signature="rolling_stats(column=None, window=5, stats=('mean', 'std', 'min', 'max')) -> DataFrame"
            description="Moving-window statistics of the series — one column per requested statistic."
            returns="DataFrame"
            example={`ts.rolling_stats(window=6, stats=("mean", "std")).head()`}
          />
          <MethodCard
            name="lag_matrix"
            signature="lag_matrix(column=None, max_lag=5, drop_na=True) -> DataFrame"
            description="Lag feature matrix [lag_0 … lag_max] for autoregressive feature engineering."
            returns="DataFrame"
            example={`features = ts.lag_matrix(max_lag=3)
features.head()`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Result objects</h2>
        <div className="method-list">
          <MethodCard
            name="AutocorrelationResult / StationarityResult / DecompositionResult"
            signature="to_dataframe() · to_dict() · to_markdown() · to_json()"
            description="Every result ships the standard exporters plus rich console __repr__ and the optional ViewX export mixin (.to_html() / .to_presentation() with pip install statslibx[viewx])."
            example={`res = ts.acf(nlags=12)
print(res)                # rich summary
res.to_dataframe()        # lag table
res.to_json()             # full payload`}
          />
        </div>
      </section>
    </>
  );
}
