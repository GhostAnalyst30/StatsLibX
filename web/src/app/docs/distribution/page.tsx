import { Spline } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";

export default function DistributionDocs() {
  return (
    <>
      <DocHeader
        title="Distribution Fitting"
        description="Maximum-likelihood fitting of candidate scipy distributions with AIC/BIC ranking, quantile-quantile data and Kolmogorov-Smirnov goodness-of-fit."
        icon={<Spline className="w-6 h-6" />}
        version="0.4.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          Module-level functions <code className="code-inline">fit_distributions</code>,{" "}
          <code className="code-inline">qq_data</code> and{" "}
          <code className="code-inline">goodness_of_fit</code> answer the two
          classic questions of univariate modelling: <em>what distribution does
          my data follow?</em> and <em>is this specific distribution a good
          fit?</em> All fitting uses scipy&apos;s MLE estimators — no new
          dependencies.
        </p>
        <div className="method-list mt-6">
          <MethodCard
            name="fit_distributions"
            signature="fit_distributions(data, column=None, candidates=None, criterion='aic') -> DistributionFitResult"
            description="Fit every candidate scipy distribution by maximum likelihood, then rank them by AIC or BIC. Fit failures (wrong support, divergence) are skipped gracefully. The result stores the ranked table, the best parameters and a frozen scipy distribution."
            returns="DistributionFitResult"
            parameters={[
              { name: "data", type: "DataFrame, Series, array or list", description: "Numeric observations (NaN dropped). Needs ≥ 10 values." },
              { name: "candidates", type: "sequence of str, optional", description: "scipy distribution names. Default: norm, lognorm, expon, gamma, beta, uniform, weibull_min, logistic, t." },
              { name: "criterion", type: "{'aic', 'bic'}", description: "Information criterion for ranking.", default: "'aic'" },
            ]}
            example={`from statslibx import fit_distributions
import numpy as np

rng = np.random.default_rng(0)
res = fit_distributions(rng.gamma(2.0, size=600))
print(res.best_name)        # 'gamma'
print(res.best_params)
res.to_dataframe()           # ranked AIC/BIC table
res.plot()                  # histogram + best-fit density (matplotlib)`}
          />
          <MethodCard
            name="qq_data"
            signature="qq_data(data, column=None, dist='norm', params=None, n_points=100) -> DataFrame"
            description="Quantile-quantile pairs: theoretical quantiles against empirical quantiles. For dist='norm' the sample mean/std are used when params are omitted. The returned frame carries an r² attribute measuring QQ-linearity."
            returns="DataFrame (attrs: r2, distribution)"
            example={`from statslibx import qq_data, load_iris

sl = load_iris()["sepal_length"]
qq = qq_data(sl, dist="norm", n_points=40)
print(f"QQ linearity R² = {qq.attrs['r2']:.4f}")`}
          />
          <MethodCard
            name="goodness_of_fit"
            signature="goodness_of_fit(data, column=None, dist='norm', params=None, alpha=0.05) -> TestResult"
            description="Kolmogorov-Smirnov goodness-of-fit test against a fully specified distribution. When parameters are estimated from the sample, a warning is raised: the classical KS p-value is anti-conservative in that case (use InferentialStats.normality_test() for the Lilliefors-corrected test)."
            returns="TestResult"
            example={`from statslibx import goodness_of_fit, load_iris

sl = load_iris()["sepal_length"]
res = goodness_of_fit(sl, dist="norm", params=(float(sl.mean()), float(sl.std(ddof=1))))
print(res.statistic, res.pvalue)`}
            note="Estimating parameters from the sample invalidates the classical KS p-value — goodness_of_fit warns about this and points to the Lilliefors-corrected normality_test."
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">DistributionFitResult</h2>
        <div className="method-list">
          <MethodCard
            name="DistributionFitResult"
            signature="best_name · best_params · best_dist · criterion · table · to_dataframe() · to_dict() · to_markdown() · to_json()"
            description="best_dist is a frozen scipy distribution (pdf/cdf/ppf ready to use). The table lists distribution, loglik, k, aic, bic and the delta versus the best model."
            example={`res = fit_distributions(data, candidates=["norm", "gamma", "expon"])
res.best_dist.cdf(10)      # cumulative probability at 10
res.best_dist.rvs(100)     # sample 100 draws from the fit`}
          />
        </div>
      </section>
    </>
  );
}
