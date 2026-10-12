"""Regression 2.1 — aligned with teammate's Regression 2.2 protocol.

Required: 70/30 and 80/20 random match-level splits, seed 42, both train and
held-out test R² and NRMSE, 200 repeated random splits, residual diagnostics.

Input: match-level CSV, eight LOCKED predictors.
Run from this folder:
  python regression_2_1_modeling.py --data regression_2_1_dataset_corrected.csv --output results_aligned
Optional comparison (confirm with Saurab first): add --competitors
Optional expensive repeated comparisons: add --repeat-competitors (implies --competitors)
Requires: pandas numpy scikit-learn scipy statsmodels matplotlib
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.api import add_constant, OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

FEATURES = [
    'fifa_points_difference',
    'points_per_match_difference',
    'avg_goals_scored_difference',
    'avg_goals_conceded_difference',
    'scoring_match_rate_difference',
    'scoring_consistency_difference',
    'clean_sheet_rate_difference',
    'host_country_indicator',
]
TARGET = 'goal_difference'
SEED = 42
SPLITS = [('70/30', 0.30), ('80/20', 0.20)]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
COMMON = ['model', 'split', 'R2_train', 'R2_test', 'NRMSE_train', 'NRMSE_test']


def validate(df: pd.DataFrame, rankings: Path | None):
    fields = ['match_id', 'home_team', 'away_team', 'home_goals', 'away_goals', TARGET, *FEATURES]
    missing = sorted(set(fields) - set(df.columns))
    if missing:
        raise ValueError(f'Missing mandatory columns: {missing}')
    if df[fields].isna().any().any():
        raise ValueError('Missing values in mandatory columns. Do not silently impute.')
    if not df.match_id.is_unique:
        raise ValueError('Regression 2.1 requires exactly one row per match_id')
    numeric = df[[TARGET, 'home_goals', 'away_goals', *FEATURES]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError('Invalid nonfinite values in the modelling data')
    if not np.allclose(df[TARGET], df.home_goals - df.away_goals):
        raise ValueError('Target must equal home_goals - away_goals (same team order)')
    if not set(df.host_country_indicator.unique()).issubset({-1, 0, 1}):
        warnings.warn('Unrecognised host_country_indicator values; inspect upstream encoding')
    audit = {
        'rows': len(df), 'unique_match_ids': int(df.match_id.nunique()),
        'missing_modelling_cells': int(df[fields].isna().sum().sum()),
        'target_order_verified': True,
        'host_indicator_minus_one': int((df.host_country_indicator == -1).sum()),
        'host_indicator_zero': int((df.host_country_indicator == 0).sum()),
        'host_indicator_plus_one': int((df.host_country_indicator == 1).sum()),
        'fifa_difference_verified': 'not checked (rankings CSV absent)',
        'other_six_difference_variables_verified': 'not fully independently verifiable without upstream team-match feature table',
    }
    if rankings is not None and rankings.exists():
        ranks = pd.read_csv(rankings)
        if {'team', 'fifa_points'}.issubset(ranks.columns) and ranks.team.is_unique:
            score = ranks.set_index('team').fifa_points
            if not set(df.home_team).union(df.away_team).issubset(set(score.index)):
                raise ValueError('Cannot verify FIFA feature: team missing from rankings')
            implied = df.home_team.map(score) - df.away_team.map(score)
            if not np.allclose(implied, df.fifa_points_difference, atol=1e-7):
                raise ValueError('FIFA points differences are NOT consistently home minus away')
            audit['fifa_difference_verified'] = 'yes, 104 rows home minus away (ranking reference)'
    return audit


def nrmse(target, prediction):
    """Observed TARGET range of the respective train or test partition."""
    y = np.asarray(target, dtype=float)
    rng = float(np.ptp(y))
    if rng <= 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y, prediction)) / rng)


def fit_models(X_train, y_train, seed: int, competitors: bool):
    """All scaling and optimization remain inside the training partition."""
    out = {}
    out['Mean baseline'] = DummyRegressor(strategy='mean').fit(X_train, y_train)
    out['Multiple linear regression'] = make_pipeline(
        StandardScaler(), LinearRegression()
    ).fit(X_train, y_train)
    # Efficient leave-one-out RidgeCV tuned ONLY on this training partition.
    # Scaling uses this outer training partition; no held-out test labels are used.
    ridge = make_pipeline(
        StandardScaler(), RidgeCV(alphas=ALPHAS, cv=None,
                                   scoring='neg_mean_squared_error')
    )
    ridge.fit(X_train, y_train)
    out['Ridge regression (tuned)'] = ridge
    ridge_inner = ridge.named_steps['ridgecv']
    choices = {'Ridge regression (tuned)': {
        'selected_alpha': float(ridge_inner.alpha_),
        'cv_rmse_goals': float(np.sqrt(-ridge_inner.best_score_)),
    }}
    if competitors:
        lasso_search = GridSearchCV(
            make_pipeline(StandardScaler(), Lasso(max_iter=20000, random_state=seed)),
            {'lasso__alpha': ALPHAS},
            scoring='neg_root_mean_squared_error',
            cv=KFold(5, shuffle=True, random_state=seed),
            n_jobs=1,
        )
        lasso_search.fit(X_train, y_train)
        out['Lasso regression (tuned)'] = lasso_search.best_estimator_
        choices['Lasso regression (tuned)'] = {
            'selected_alpha': float(lasso_search.best_params_['lasso__alpha']),
            'cv_rmse_goals': float(-lasso_search.best_score_),
        }
        out['Random forest'] = RandomForestRegressor(
            n_estimators=120, min_samples_leaf=3, random_state=seed, n_jobs=1
        ).fit(X_train, y_train)
        out['Gradient boosting'] = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            min_samples_leaf=3, random_state=seed
        ).fit(X_train, y_train)
    return out, choices


def model_score(model, X_train, y_train, X_test, y_test, split):
    ptrain = model.predict(X_train)
    ptest = model.predict(X_test)
    return ({'model': '', 'split': split,
             'R2_train': float(r2_score(y_train, ptrain)),
             'R2_test': float(r2_score(y_test, ptest)),
             'NRMSE_train': nrmse(y_train, ptrain),
             'NRMSE_test': nrmse(y_test, ptest)},
            {'RMSE_train_goals': float(np.sqrt(mean_squared_error(y_train, ptrain))),
             'RMSE_test_goals': float(np.sqrt(mean_squared_error(y_test, ptest))),
             'train_observed_target_range': float(np.ptp(y_train)),
             'test_observed_target_range': float(np.ptp(y_test)),
             'n_train': len(y_train), 'n_test': len(y_test)},
            ptrain, ptest)


def create_eda(df, output):
    X = df[FEATURES]
    cor = X.corr()
    cor.to_csv(output / 'regression_2_1_predictor_correlations.csv')
    Xsm = add_constant(X.astype(float), has_constant='add')
    vif_rows = []
    for i, feature in enumerate(FEATURES, start=1):
        try:
            vif = float(variance_inflation_factor(Xsm.to_numpy(), i))
        except Exception:
            vif = float('inf')
        vif_rows.append({'feature': feature, 'VIF': vif})
    vifs = pd.DataFrame(vif_rows)
    vifs.to_csv(output / 'regression_2_1_vif.csv', index=False)
    data_summary = pd.DataFrame({
        'feature': FEATURES,
        'missing': [int(X[c].isna().sum()) for c in FEATURES],
        'mean': [float(X[c].mean()) for c in FEATURES],
        'sd': [float(X[c].std()) for c in FEATURES],
        'min': [float(X[c].min()) for c in FEATURES],
        'max': [float(X[c].max()) for c in FEATURES],
    })
    data_summary.to_csv(output / 'regression_2_1_eda_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(cor.values, vmin=-1, vmax=1, cmap='coolwarm')
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    ax.set_xticks(range(len(FEATURES)), [f.replace('_difference','_diff') for f in FEATURES], rotation=60, ha='right')
    ax.set_yticks(range(len(FEATURES)), [f.replace('_difference','_diff') for f in FEATURES])
    ax.set_title('Regression 2.1: correlation between locked predictors')
    fig.tight_layout()
    fig.savefig(output / 'regression_2_1_correlation_heatmap.png', dpi=150)
    plt.close(fig)
    return vifs, cor


def diagnostics(X_train, y_train, output):
    # OLS only on training rows; leverage and residual tests do not use the held-out labels.
    Xsm = add_constant(X_train.astype(float), has_constant='add')
    ols = OLS(np.asarray(y_train, dtype=float), Xsm).fit()
    fitted = np.asarray(ols.fittedvalues)
    residuals = np.asarray(ols.resid)
    bp_lm, bp_lmp, bp_f, bp_fp = het_breuschpagan(residuals, Xsm)
    influences = OLSInfluence(ols)
    cooks = np.asarray(influences.cooks_distance[0])
    threshold = 4 / len(y_train)
    row = {
        'diagnostic_split': '80/20 training data',
        'OLS_intercept_raw_units': float(ols.params.iloc[0]),
        'OLS_intercept_pvalue': float(ols.pvalues.iloc[0]),
        'Breusch_Pagan_LM_pvalue': float(bp_lmp),
        'Breusch_Pagan_F_pvalue': float(bp_fp),
        'Cook_threshold_4_over_n': float(threshold),
        'n_Cook_above_threshold': int(np.sum(cooks > threshold)),
        'max_Cook_distance': float(cooks.max()),
    }
    pd.DataFrame([row]).to_csv(output / 'regression_2_1_diagnostic_tests.csv', index=False)
    top = pd.DataFrame({'match_id': X_train.index.to_numpy(), 'fitted_goal_difference': fitted,
                        'residual': residuals, 'Cooks_distance': cooks})
    top = top.sort_values('Cooks_distance', ascending=False)
    top.to_csv(output / 'regression_2_1_influence.csv', index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fitted, residuals, s=20)
    ax.axhline(0, linestyle='--')
    ax.set(xlabel='OLS fitted goal difference', ylabel='Residual (actual − fitted)',
           title='80/20 train: OLS residuals vs fitted')
    fig.tight_layout()
    fig.savefig(output / 'regression_2_1_residuals_vs_fitted.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    stats.probplot(residuals, dist='norm', plot=ax)
    ax.set(title='80/20 train: OLS residual Q–Q plot')
    fig.tight_layout()
    fig.savefig(output / 'regression_2_1_qq_plot.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stem(np.arange(1, len(cooks)+1), cooks, markerfmt='.')
    ax.axhline(threshold, linestyle='--', label='4 / n threshold')
    ax.set(xlabel='Training observation number', ylabel="Cook's distance",
           title='80/20 train: influential observations')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'regression_2_1_cooks_distance.png', dpi=150)
    plt.close(fig)
    return row


def run(path, output, rankings, repeats, competitors, repeat_competitors):
    if repeat_competitors:
        competitors = True
    df = pd.read_csv(path)
    audit = validate(df, rankings)
    df = df.copy().set_index('match_id', drop=False)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([audit]).to_csv(output / 'regression_2_1_data_integrity.csv', index=False)
    vifs, correlations = create_eda(df, output)
    X, y = df[FEATURES].astype(float), df[TARGET].astype(float)
    primary = []
    details, predictions, coeffs, all_repeat_scores, repeat_details = [], [], [], [], []
    diagnostic = None
    for split_name, test_size in SPLITS:
        tr, te = train_test_split(np.arange(len(df)), test_size=test_size,
                                  random_state=SEED, shuffle=True)
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        fits, selected = fit_models(Xtr, ytr, SEED, competitors)
        for name, fitted_model in fits.items():
            row, extra, predtrain, predtest = model_score(fitted_model, Xtr, ytr, Xte, yte, split_name)
            row['model'] = name
            primary.append(row)
            details.append({**row, **extra, **selected.get(name, {})})
            for idx, actual, pred in zip(Xte.index, yte, predtest):
                predictions.append({'model': name, 'split': split_name, 'match_id': idx,
                                    'actual_goal_difference': float(actual),
                                    'predicted_goal_difference': float(pred),
                                    'residual_actual_minus_predicted': float(actual - pred)})
            if split_name == '80/20' and name in ('Multiple linear regression', 'Ridge regression (tuned)'):
                scaler = fitted_model.named_steps['standardscaler']
                est = fitted_model.steps[-1][1]
                raw_weights = est.coef_ / scaler.scale_
                raw_intercept = float(est.intercept_ - np.dot(raw_weights, scaler.mean_))
                for f, c, b in zip(FEATURES, est.coef_, raw_weights):
                    coeffs.append({'model': name, 'split': split_name, 'feature': f,
                                   'standardized_feature_coefficient': float(c),
                                   'original_units_coefficient': float(b),
                                   'raw_scale_intercept': raw_intercept})
        if split_name == '80/20':
            diagnostic = diagnostics(Xtr, ytr, output)

        # 200 random splits; each model refitted from scratch. No test-set tuning.
        for i in range(repeats):
            repeat_seed = SEED + i
            tri, tei = train_test_split(np.arange(len(df)), test_size=test_size,
                                        random_state=repeat_seed, shuffle=True)
            Xi, Xe = X.iloc[tri], X.iloc[tei]
            yi, ye = y.iloc[tri], y.iloc[tei]
            fits, _ = fit_models(Xi, yi, repeat_seed, competitors and repeat_competitors)
            for name, fitted_model in fits.items():
                row, extra, _, _ = model_score(fitted_model, Xi, yi, Xe, ye, split_name)
                row['model'] = name
                all_repeat_scores.append({**row, 'repeat_id': i+1, 'seed': repeat_seed})
                repeat_details.append({**row, **extra, 'repeat_id': i+1, 'seed': repeat_seed})
        print(f'Completed {split_name}: {len(fits)} repeated models per seed x {repeats} repeats', flush=True)

    pd.DataFrame(primary)[COMMON].to_csv(output / 'regression_2_1_results.csv', index=False)
    pd.DataFrame(details).to_csv(output / 'regression_2_1_model_details.csv', index=False)
    pd.DataFrame(predictions).to_csv(output / 'regression_2_1_predictions.csv', index=False)
    pd.DataFrame(coeffs).to_csv(output / 'regression_2_1_coefficients.csv', index=False)
    repeated = pd.DataFrame(all_repeat_scores)
    repeated[COMMON + ['repeat_id','seed']].to_csv(output / 'regression_2_1_repeated_splits.csv', index=False)
    pd.DataFrame(repeat_details).to_csv(output / 'regression_2_1_repeated_details.csv', index=False)
    grp = repeated.groupby(['model', 'split'], sort=False)
    summ = grp[['R2_train', 'R2_test', 'NRMSE_train', 'NRMSE_test']].agg(['mean','std'])
    summ.columns = ['_'.join(item) for item in summ.columns.to_flat_index()]
    summ = summ.reset_index()
    summ['n_repeats'] = repeats
    summ.to_csv(output / 'regression_2_1_repeated_summary.csv', index=False)
    print('\nLOCKED 42-SEED SPLITS: NRMSE expressed as a fraction; multiply by 100 for %')
    print(pd.DataFrame(primary)[COMMON].round(4).to_string(index=False))
    print(f'\nREPEATED TEST METRICS: {repeats} shuffle seeds (overlapping match sets)')
    print(summ[['model','split','R2_test_mean','R2_test_std','NRMSE_test_mean','NRMSE_test_std']].round(4).to_string(index=False))
    print('\nQUALITY CHECK: host indicator counts =', {k: audit[f'host_indicator_{s}'] for k, s in [(-1,'minus_one'), (0,'zero'), (1,'plus_one')]})
    print('Highest predictor VIF:', float(vifs.VIF.max()))
    print('80/20 train OLS diagnostics:', diagnostic)
    print(f'Output location: {output.resolve()}')
    return pd.DataFrame(primary), summ


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('regression_2_1_dataset_corrected.csv'))
    parser.add_argument('--output', type=Path, default=Path('results_aligned'))
    parser.add_argument('--fifa-rankings', type=Path, default=Path('fifa_rankings_11_june_2026.csv'))
    parser.add_argument('--repeats', type=int, default=200)
    parser.add_argument('--competitors', action='store_true', help='Optional competitors, pending team agreement')
    parser.add_argument('--repeat-competitors', action='store_true', help='Also repeat competitors 200 times (slower)')
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('At least two repeats are required to calculate sample SD')
    run(args.data, args.output, args.fifa_rankings, args.repeats, args.competitors, args.repeat_competitors)
