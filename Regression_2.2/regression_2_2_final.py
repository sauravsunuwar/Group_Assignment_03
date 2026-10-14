"""Regression 2.2 - predict goals_scored (208 team-match rows, 104 matches).
Baseline: Multiple Linear Regression (locked 8 predictors).
HD extension: competing scikit-learn regressors on the SAME grouped splits.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------ CONFIG ------------------------------
CSV_PATH = "/mnt/user-data/uploads/regression_2_2_dataset.csv"
OUT = "/mnt/user-data/outputs/"
TARGET, GROUP, SEED, N_REPEATS = "goals_scored", "match_id", 42, 200
# NOTE: the CSV contains avg_goal_difference_before, NOT matches_played_before
# (which is in the Teams message). Confirm with the group which is final.
FEATURES = [
    "avg_goals_scored_before", "opponent_avg_goals_conceded_before",
    "team_fifa_points", "opponent_fifa_points", "win_rate_before",
    "opponent_scoring_match_rate_before", "avg_goal_difference_before",
    "opponent_clean_sheet_rate_before",
]
# --------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
assert len(df) == 208 and df[GROUP].nunique() == 104
assert df[FEATURES + [TARGET]].isna().sum().sum() == 0
X, y, g = df[FEATURES], df[TARGET], df[GROUP]


def nrmse(t, p):  # RMSE / range of observed target (state this in the report)
    return np.sqrt(mean_squared_error(t, p)) / (t.max() - t.min())


# ---- 1. EDA evidence ----
corr = df[FEATURES + [TARGET]].corr()
Xc = sm.add_constant(X)
vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])], FEATURES)
print("Target mean/var:", round(y.mean(), 3), round(y.var(), 3))
print("\nCorrelation with target:\n", corr[TARGET].drop(TARGET).round(3))
print("\nMax |r| between predictors:",
      round(corr.loc[FEATURES, FEATURES].where(~np.eye(8, dtype=bool)).abs().max().max(), 3))
print("VIF:\n", vif.round(3))

# ---- 2. Models (same splits, same metrics) ----
models = {
    "Linear Regression (baseline)": LinearRegression(),
    "Poisson Regression": make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000)),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                                  subsample=0.8, random_state=SEED),
}


def split_eval(name, model, test_size, seed):
    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(df, y, g))
    m = clone(model).fit(X.iloc[tr], y.iloc[tr])
    ptr, pte = m.predict(X.iloc[tr]), m.predict(X.iloc[te])
    return dict(model=name, split=f"{round((1-test_size)*100)}/{round(test_size*100)}",
                R2_train=r2_score(y.iloc[tr], ptr), R2_test=r2_score(y.iloc[te], pte),
                NRMSE_train=nrmse(y.iloc[tr], ptr), NRMSE_test=nrmse(y.iloc[te], pte))


single = pd.DataFrame([split_eval(n, m, ts, SEED) for n, m in models.items() for ts in (0.2, 0.3)])
print("\n=== SINGLE SPLIT (seed 42, grouped by match) ===\n", single.round(4).to_string(index=False))

rep = pd.DataFrame([split_eval(n, m, ts, s) for s in range(N_REPEATS)
                    for n, m in models.items() for ts in (0.2, 0.3)])
summ = rep.groupby(["model", "split"]).agg(["mean", "std"]).round(4)
print(f"\n=== {N_REPEATS} REPEATED GROUPED SPLITS (mean, std) ===\n", summ.to_string())
single.to_csv(OUT + "results_single_split.csv", index=False)
summ.to_csv(OUT + "results_repeated_splits.csv")

# ---- 3. Diagnostics for the baseline (full data) ----
ols = sm.OLS(y, Xc).fit()
ols_cl = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": g})
print("\n", ols_cl.summary().tables[1])
res, fit = ols.resid, ols.fittedvalues
print("R2 / adj R2 (full data):", round(ols.rsquared, 4), round(ols.rsquared_adj, 4))
print("Breusch-Pagan p:", round(het_breuschpagan(res, Xc)[1], 4),
      "| Shapiro p:", round(stats.shapiro(res)[1], 4),
      "| Durbin-Watson:", round(sm.stats.durbin_watson(res), 3))
print("Negative predictions:", int((fit < 0).sum()), "of", len(fit))
# residual correlation between the two teams in the same match
r = res.groupby(g).apply(lambda s: s.values)
pairs = np.array([v for v in r if len(v) == 2])
print("Within-match residual correlation:", round(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1], 3))
cooks = ols.get_influence().cooks_distance[0]
print("Cook's D > 4/n rows:", np.where(cooks > 4 / len(df))[0].tolist())

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].scatter(fit, res, alpha=.6); ax[0, 0].axhline(0, c="r"); ax[0, 0].set(title="Residuals vs fitted")
sm.qqplot(res, line="45", fit=True, ax=ax[0, 1]); ax[0, 1].set_title("Q-Q plot")
ax[1, 0].hist(res, bins=20); ax[1, 0].set_title("Residual histogram")
ax[1, 1].stem(cooks, markerfmt=","); ax[1, 1].axhline(4 / len(df), c="r", ls="--"); ax[1, 1].set_title("Cook's distance")
plt.tight_layout(); plt.savefig(OUT + "diagnostics_2_2.png", dpi=150); plt.close()

# ---- 4. Tuning competitor with GroupKFold on training data only (80/20) ----
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, y, g))
grid = GridSearchCV(GradientBoostingRegressor(subsample=0.8, random_state=SEED),
                    {"n_estimators": [50, 100, 200], "max_depth": [1, 2, 3], "learning_rate": [0.02, 0.05, 0.1]},
                    cv=GroupKFold(5), scoring="neg_mean_squared_error")
grid.fit(X.iloc[tr], y.iloc[tr], groups=g.iloc[tr])
gb = grid.best_estimator_
print("\nTuned GB params:", grid.best_params_, "| test R2:", round(r2_score(y.iloc[te], gb.predict(X.iloc[te])), 4),
      "| test NRMSE:", round(nrmse(y.iloc[te], gb.predict(X.iloc[te])), 4))
imp = pd.Series(gb.feature_importances_, FEATURES).sort_values(ascending=False)
print("GB feature importance:\n", imp.round(3))
lin = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
print("Linear coef (raw):\n", pd.Series(lin.coef_, FEATURES).round(5))
