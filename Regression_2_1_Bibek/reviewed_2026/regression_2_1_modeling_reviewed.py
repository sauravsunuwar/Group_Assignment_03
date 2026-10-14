"""HIT140 Regression 2.1 — targeted methodological review (2026-10-13).

Preserves: corrected 104-row dataset, exactly eight locked predictors, seed=42,
70/30 and 80/20 match-level splits, observed-target-range NRMSE, 200 repetitions.

Changes from original: scale INSIDE every inner CV fold (full Pipeline inside
GridSearchCV), evaluate 6 model families on identical holdouts and all repeated
splits, tune ALL non-baseline families strictly on training data, add MAE,
feature-target EDA, visual diagnostics and metadata on hyperparameter choices.

Run:
  python regression_2_1_modeling_reviewed.py --data regression_2_1_dataset_corrected.csv --output results_reviewed --repeats 200

N.B. 200 x 2 nested-CV comparisons with forests and boosting can take minutes.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import __version__ as skl_version
from joblib import Parallel, delayed
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.api import add_constant, OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

FEATURES = [
    'fifa_points_difference', 'points_per_match_difference',
    'avg_goals_scored_difference', 'avg_goals_conceded_difference',
    'scoring_match_rate_difference', 'scoring_consistency_difference',
    'clean_sheet_rate_difference', 'host_country_indicator',
]
TARGET = 'goal_difference'
SEED = 42
SPLITS = [('70/30', 0.30), ('80/20', 0.20)]
COMMON = ['model', 'split', 'R2_train', 'R2_test', 'NRMSE_train', 'NRMSE_test']
MODELS = ['Mean baseline', 'Multiple linear regression', 'Ridge regression (tuned)',
          'Lasso regression (tuned)', 'Random forest (tuned)', 'Gradient boosting (tuned)']
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def validate(df: pd.DataFrame, rankings: Path | None, upstream: Path | None) -> dict:
    fields = ['match_id', 'match_date', 'home_team', 'away_team', 'home_goals',
              'away_goals', TARGET] + FEATURES
    missing = sorted(set(fields) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required fields: {missing}')
    if len(df) != 104 or df['match_id'].nunique() != 104:
        raise ValueError('Expected 104 unique match rows, without duplicate match_id')
    if df[fields].isna().any().any():
        raise ValueError('Missing values in required input/target fields')
    numeric = df[['home_goals', 'away_goals', TARGET] + FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite numeric inputs')
    if not np.allclose(df[TARGET], df.home_goals - df.away_goals):
        raise ValueError('goal_difference does not match home_goals - away_goals')
    if (df.home_team == df.away_team).any():
        raise ValueError('Some rows have identical home and away teams')
    if not set(df.host_country_indicator.unique()).issubset({-1, 0, 1}):
        raise ValueError('Expected signed host-country indicator in {-1,0,+1}')
    audit = {
        'n_rows':len(df), 'unique_match_ids':int(df.match_id.nunique()),
        'missing_required_cells':int(df[fields].isna().sum().sum()),
        'target_matches_home_minus_away': True,
        'host_minus_one':int((df.host_country_indicator==-1).sum()),
        'host_zero':int((df.host_country_indicator==0).sum()),
        'host_plus_one':int((df.host_country_indicator==1).sum()),
        'fifa_home_minus_away_verified':False,
        'other_six_features_independently_verified':False,
        'notes':'Other six historical-form inputs need original point-in-time feature building script for full verification.'
    }
    if rankings and rankings.exists():
        ranks=pd.read_csv(rankings)
        mapping = ranks.set_index('team').fifa_points
        if not set(df.home_team).union(df.away_team).issubset(mapping.index):
            raise ValueError('Missing team in FIFA reference table')
        diff = df.home_team.map(mapping)-df.away_team.map(mapping)
        if not np.allclose(diff,df.fifa_points_difference,atol=1e-7):
            raise ValueError('FIFA points difference not consistently Team A minus Team B')
        audit['fifa_home_minus_away_verified']=True
    if upstream is not None and upstream.exists():
        original=pd.read_csv(upstream)
        if original.shape!=df.shape or not original.match_id.equals(df.match_id):
            raise ValueError('Original/adjusted dataset shape or match order differs')
        diff=(df[FEATURES]-original[FEATURES]).abs()>1e-8
        audit['changed_feature_cells_vs_original']=int(diff.sum().sum())
        audit['changed_match_rows_vs_original']=int(diff.any(axis=1).sum())
        if (df[TARGET]!=original[TARGET]).any():
            raise ValueError('Correction unexpectedly changed target values')
    return audit


def calc_metrics(y, pred):
    y = np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float)
    rmse=float(np.sqrt(mean_squared_error(y,pred)))
    observed_range=float(np.ptp(y))
    return {'R2':float(r2_score(y,pred)), 'NRMSE':rmse/observed_range if observed_range>0 else float('nan'),
            'RMSE_goals':rmse,'MAE_goals':float(mean_absolute_error(y,pred)),
            'observed_range':observed_range}


def inner_cv(seed):
    # Fixed 5-fold CV, SHUFFLED using seed. All folds wholly inside outer train.
    return KFold(n_splits=5,shuffle=True,random_state=seed)


def make_candidates(seed):
    return {
        'Mean baseline': (DummyRegressor(strategy='mean'),None),
        'Multiple linear regression': (make_pipeline(StandardScaler(),LinearRegression()),None),
        'Ridge regression (tuned)': (make_pipeline(StandardScaler(), Ridge()), {'ridge__alpha':ALPHAS}),
        'Lasso regression (tuned)': (make_pipeline(StandardScaler(), Lasso(max_iter=20000,random_state=seed)), {'lasso__alpha':ALPHAS}),
        # Tight, pre-specified grid for small n; fold CV performed for both trees.
        'Random forest (tuned)': (RandomForestRegressor(n_estimators=60,random_state=seed,n_jobs=1),
                   {'min_samples_leaf':[2,5], 'max_features':[0.7]}),
        'Gradient boosting (tuned)': (GradientBoostingRegressor(random_state=seed,learning_rate=0.05,n_estimators=80),
                   {'max_depth':[1,2], 'min_samples_leaf':[3]}),
    }


def fit_all(Xtr,ytr,seed):
    fits={}; tune_meta={}
    for name,(est,params) in make_candidates(seed).items():
        if params is None:
            fitted=est.fit(Xtr,ytr)
            tune_meta[name]={'selected_params':'not tuned','inner_cv_rmse_goals':float('nan')}
        else:
            # The entire StandardScaler+estimator is inside GridSearchCV. Thus
            # every fold refits scaling using its OWN training indices only.
            cv = inner_cv(seed)  # Identical 5-fold training-only CV protocol for every tuned family
            search=GridSearchCV(est,param_grid=params,cv=cv,
                     scoring='neg_root_mean_squared_error',refit=True,n_jobs=1,error_score='raise')
            search.fit(Xtr,ytr)
            fitted=search.best_estimator_
            tune_meta[name]={'selected_params':json.dumps(search.best_params_,sort_keys=True),
                             'inner_cv_rmse_goals':float(-search.best_score_)}
        fits[name]=fitted
    return fits,tune_meta


def eda(df,out):
    X=df[FEATURES].astype(float); y=df[TARGET].astype(float)
    desc=X.describe().T; desc['missing']=X.isna().sum(); desc.to_csv(out/'regression_2_1_eda_summary.csv')
    pd.DataFrame({'metric':['mean','sd','median','min','max','skew'],
        'goal_difference':[y.mean(),y.std(),y.median(),y.min(),y.max(),y.skew()]}).to_csv(out/'regression_2_1_target_summary.csv',index=False)
    corr=X.corr();corr.to_csv(out/'regression_2_1_predictor_correlations.csv')
    effect=[]
    for f in FEATURES:
        pear=stats.pearsonr(X[f],y)
        sp=stats.spearmanr(X[f],y)
        effect.append({'feature':f,'pearson_r':float(pear.statistic),'pearson_p_exploratory':float(pear.pvalue),
                       'spearman_rho':float(sp.statistic),'spearman_p_exploratory':float(sp.pvalue)})
    effect=pd.DataFrame(effect).sort_values('pearson_r',key=lambda x:x.abs(),ascending=False)
    effect.to_csv(out/'regression_2_1_feature_target_correlations.csv',index=False)
    A=add_constant(X,has_constant='add').to_numpy()
    vif=[]
    for i,feature in enumerate(FEATURES,start=1):
        vif.append({'feature':feature,'VIF':float(variance_inflation_factor(A,i))})
    vif=pd.DataFrame(vif);vif.to_csv(out/'regression_2_1_vif.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,7));im=ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm')
    labels=[f.replace('_difference','_diff') for f in FEATURES]
    ax.set_xticks(range(8),labels,rotation=55,ha='right');ax.set_yticks(range(8),labels)
    fig.colorbar(im,ax=ax,label='Pearson r');ax.set_title('Locked predictors: Pearson relationships')
    fig.tight_layout();fig.savefig(out/'regression_2_1_correlation_heatmap.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.barh(effect.feature.iloc[::-1],effect.pearson_r.iloc[::-1]);ax.axvline(0,lw=1)
    ax.set(xlabel='Pearson r with observed goal difference (descriptive only)',title='Feature-to-target relationships (full-sample EDA)')
    fig.tight_layout();fig.savefig(out/'regression_2_1_feature_target.png',dpi=150);plt.close(fig)
    return effect,vif


def diagnostics(df,Xtr,ytr,out):
    # 80/20 OLS training partition, not the held-out test labels.
    raw=add_constant(Xtr.astype(float),has_constant='add')
    ols=OLS(np.asarray(ytr,dtype=float),raw).fit()
    pred=np.asarray(ols.fittedvalues);res=np.asarray(ols.resid)
    bp=het_breuschpagan(res,raw);infl=OLSInfluence(ols)
    cooks=np.asarray(infl.cooks_distance[0]);threshold=4/len(ytr)
    report={'split':'80/20 training','n_train':len(ytr),'bp_LM_p':float(bp[1]),
            'bp_F_p':float(bp[3]),'n_cooks_above_4_n':int((cooks>threshold).sum()),
            'max_cooks_distance':float(cooks.max()),'cooks_threshold':threshold,
            'ols_intercept':float(ols.params.iloc[0]),'ols_intercept_p':float(ols.pvalues.iloc[0]),
            'host_effect_coefficient':float(ols.params['host_country_indicator']),
            'host_effect_p':float(ols.pvalues['host_country_indicator']),
            'train_residual_skew':float(stats.skew(res)),
            'train_residual_shapiro_p_exploratory':float(stats.shapiro(res).pvalue)}
    pd.DataFrame([report]).to_csv(out/'regression_2_1_diagnostic_tests.csv',index=False)
   ranking = pd.DataFrame({
'match_id': Xtr.index.to_numpy(),
'fitted_goal_difference': pred,
'residual': res,
'cooks_distance': cooks
})
      
    ranking.sort_values('cooks_distance',ascending=False).to_csv(out/'regression_2_1_influence.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.5,4.6));ax.scatter(pred,res,s=22);ax.axhline(0,ls='--');
    ax.set(xlabel='OLS fitted',ylabel='Actual - fitted',title='80/20 train: residuals vs fitted')
    fig.tight_layout();fig.savefig(out/'regression_2_1_residuals_vs_fitted.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.6));stats.probplot(res,dist='norm',plot=ax);
    ax.set_title('80/20 train: OLS Q-Q');fig.tight_layout();fig.savefig(out/'regression_2_1_qq_plot.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.stem(np.arange(len(cooks)),cooks,markerfmt='.');ax.axhline(threshold,ls='--',label='4/n screening')
    ax.set(xlabel='Training row index',ylabel="Cook's distance",title='80/20 train: OLS influence');ax.legend()
    fig.tight_layout();fig.savefig(out/'regression_2_1_cooks_distance.png',dpi=150);plt.close(fig)
    return report


def plot_primary(preds,details,coefficients,out):
    D=pd.DataFrame(details);p=pd.DataFrame(preds)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,split in zip(axes,['70/30','80/20']):
        d=D[D.split.eq(split)];x=np.arange(len(d))
        ax.bar(x,d.R2_test);ax.set_xticks(x,d.model,rotation=65,ha='right',fontsize=8)
        ax.axhline(0,ls='--',lw=.8);ax.set(title=split+' held-out R²',ylabel='R²')
    fig.tight_layout();fig.savefig(out/'regression_2_1_comparison_r2.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,split in zip(axes,['70/30','80/20']):
        d=D[D.split.eq(split)];x=np.arange(len(d))
        ax.bar(x,100*d.NRMSE_test);ax.set_xticks(x,d.model,rotation=65,ha='right',fontsize=8)
        ax.set(title=split+' held-out range-normalised error',ylabel='NRMSE (%)')
    fig.tight_layout();fig.savefig(out/'regression_2_1_comparison_nrmse.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for i,split in enumerate(['70/30','80/20']):
        for j,model in enumerate(['Ridge regression (tuned)','Lasso regression (tuned)']):
            ax=axes[j,i]
            sub=p[(p.split==split)&(p.model==model)]
            ax.scatter(sub.actual_goal_difference,sub.predicted_goal_difference,s=28)
            bounds=[min(sub.actual_goal_difference.min(),sub.predicted_goal_difference.min())-.25,
                    max(sub.actual_goal_difference.max(),sub.predicted_goal_difference.max())+.25]
            ax.plot(bounds,bounds,ls='--')
            ax.set(xlabel='Actual goal difference',ylabel='Predicted',title=f'{split} holdout: {model.split()[0]}')
    fig.tight_layout();fig.savefig(out/'regression_2_1_actual_vs_predicted.png',dpi=150);plt.close(fig)
    c=pd.DataFrame(coefficients)
    if not c.empty:
        s=c[c.split.eq('80/20')].pivot(index='feature',columns='model',values='standardized_input_coefficient').reindex(FEATURES)
        ax=s.plot(kind='barh',figsize=(9,5.4))
        ax.axvline(0,lw=1);ax.set(xlabel='Coefficient per one training-SD predictor increase',
                                  title='80/20 train: conditional standardized coefficients (not causal)')
        plt.tight_layout();plt.savefig(out/'regression_2_1_coefficients_comparison.png',dpi=150);plt.close('all')


def one_repeat(rep,X,y,ratio,tst):
    seed=SEED+rep
    tr,te=train_test_split(np.arange(len(X)),test_size=tst,random_state=seed,shuffle=True)
    Xi,Xe=X.iloc[tr],X.iloc[te];yi,ye=y.iloc[tr],y.iloc[te]
    trained,meta_i=fit_all(Xi,yi,seed)
    scores=[];params=[]
    for model,est in trained.items():
        a=calc_metrics(yi,est.predict(Xi));b=calc_metrics(ye,est.predict(Xe))
        scores.append({'model':model,'split':ratio,'repeat_id':rep+1,'seed':seed,
                'R2_train':a['R2'],'R2_test':b['R2'],'NRMSE_train':a['NRMSE'],
                'NRMSE_test':b['NRMSE'],'RMSE_train_goals':a['RMSE_goals'],
                'RMSE_test_goals':b['RMSE_goals'],'MAE_train_goals':a['MAE_goals'],
                'MAE_test_goals':b['MAE_goals']})
        params.append({'model':model,'split':ratio,'repeat_id':rep+1,**meta_i[model]})
    return scores,params

def run(data,out,rankings,original,repeats,jobs):
    df=pd.read_csv(data);audit=validate(df,rankings,original)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([audit]).to_csv(out/'regression_2_1_data_integrity.csv',index=False)
    eda_train_indices,_=train_test_split(np.arange(len(df)),test_size=0.20,random_state=SEED,shuffle=True)
    # Feature-target EDA is training-only to avoid peeking at held-out labels.
    effect,vif=eda(df.iloc[eda_train_indices],out)
    # Unique match_id indexed for traceable predictions and influential observations.
    df=df.set_index('match_id',drop=False)
    X=df[FEATURES].astype(float);y=df[TARGET].astype(float)
    main=[];details=[];preds=[];coefficients=[];repeated=[];repeat_params=[];diagnostic=None
    for ratio,tst in SPLITS:
        train,test=train_test_split(np.arange(len(df)),test_size=tst,random_state=SEED,shuffle=True)
        Xtr,Xte=X.iloc[train],X.iloc[test];ytr,yte=y.iloc[train],y.iloc[test]
        fits,meta=fit_all(Xtr,ytr,SEED)
        for model,est in fits.items():
            if model in ('Ridge regression (tuned)','Lasso regression (tuned)'):
                scaler=est.named_steps['standardscaler']
                reg=est.steps[-1][1]
                rawcoef=reg.coef_/scaler.scale_
                rawintercept=float(reg.intercept_-np.dot(rawcoef,scaler.mean_))
                for feature,stdcoef,unscaled in zip(FEATURES,reg.coef_,rawcoef):
                    coefficients.append({'model':model,'split':ratio,'feature':feature,
                       'standardized_input_coefficient':float(stdcoef),
                       'original_units_coefficient':float(unscaled),
                       'raw_intercept':rawintercept,'effectively_zero_in_lasso':bool(abs(stdcoef)<1e-9) if model.startswith('Lasso') else False})
            p0,p1=est.predict(Xtr),est.predict(Xte)
            a,b=calc_metrics(ytr,p0),calc_metrics(yte,p1)
            scores={'model':model,'split':ratio,'R2_train':a['R2'],'R2_test':b['R2'],
                    'NRMSE_train':a['NRMSE'],'NRMSE_test':b['NRMSE']}
            main.append(scores)
            details.append({**scores,'RMSE_train_goals':a['RMSE_goals'],'RMSE_test_goals':b['RMSE_goals'],
                   'MAE_train_goals':a['MAE_goals'],'MAE_test_goals':b['MAE_goals'],
                   'n_train':len(Xtr),'n_test':len(Xte),'train_target_range':a['observed_range'],
                   'test_target_range':b['observed_range'],**meta[model]})
            for mid,t,pred in zip(Xte.index,yte.to_numpy(),p1):
                preds.append({'model':model,'split':ratio,'match_id':mid,'actual_goal_difference':float(t),
                         'predicted_goal_difference':float(pred),'residual_actual_minus_predicted':float(t-pred)})
        if ratio=='80/20':diagnostic=diagnostics(df,Xtr,ytr,out)
        # Runs are isolated match-level splits. Training + inner CV never sees outer holdout.
        for chunk_start in range(0,repeats,25):
            batch=range(chunk_start,min(chunk_start+25,repeats))
            results=Parallel(n_jobs=jobs,prefer='processes')(
                 delayed(one_repeat)(rep,X,y,ratio,tst) for rep in batch)
            for rows,params in results:
                repeated.extend(rows);repeat_params.extend(params)
            print(f'{ratio}: completed {min(chunk_start+25,repeats)}/{repeats} repeated splits',flush=True)
    pd.DataFrame(main)[COMMON].to_csv(out/'regression_2_1_results.csv',index=False)
    pd.DataFrame(details).to_csv(out/'regression_2_1_model_details.csv',index=False)
    pd.DataFrame(preds).to_csv(out/'regression_2_1_predictions.csv',index=False)
    pd.DataFrame(coefficients).to_csv(out/'regression_2_1_coefficients.csv',index=False)
    pd.DataFrame(repeated).to_csv(out/'regression_2_1_repeated_splits.csv',index=False)
    pd.DataFrame(repeat_params).to_csv(out/'regression_2_1_repeated_tuning.csv',index=False)
    rep=pd.DataFrame(repeated);g=rep.groupby(['model','split'],sort=False)
    all_metrics=['R2_train','R2_test','NRMSE_train','NRMSE_test','RMSE_train_goals',
                 'RMSE_test_goals','MAE_train_goals','MAE_test_goals']
    sm=g[all_metrics].agg(['mean','std']).reset_index()
    sm.columns=['_'.join(v).strip('_') for v in sm.columns]
    sm['n_repeats']=repeats;sm.to_csv(out/'regression_2_1_repeated_summary.csv',index=False)
    plot_primary(preds,details,coefficients,out)
    pd.DataFrame([{'seed':SEED,'n_repeats_per_ratio':repeats,'predictor_count':len(FEATURES),
        'cv':'5-fold shuffled CV for all tuned model families, confined to each outer training partition','nrmse':'RMSE / (observed max - observed min) within the scored partition',
        'tuning':'training-only GridSearchCV; all hyperparameters refit within each repeated training split',
        'feature_target_eda':'seed42 80/20 training rows only',
        'python_version':sys.version.split()[0],'sklearn_version':skl_version,
        'platform':platform.platform()}]).to_csv(out/'run_metadata.csv',index=False)
    print('\nSEED-42 HOLDOUT:');print(pd.DataFrame(details)[['model','split','R2_train','R2_test','RMSE_test_goals','NRMSE_test','MAE_test_goals','selected_params']].round(5).to_string(index=False))
    print('\n200-REPEAT TEST SUMMARY:');print(sm[['model','split','R2_test_mean','R2_test_std','NRMSE_test_mean','NRMSE_test_std']].round(5).to_string(index=False))
    print('\nEDA strongest target correlations:');print(effect[['feature','pearson_r']].head(4).to_string(index=False))
    print('Largest predictor VIF:',round(vif.VIF.max(),3),'\nOLS diagnostics:',diagnostic)
    print('Saved in',out.resolve(),flush=True)
    return pd.DataFrame(details),sm

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path,default=Path('regression_2_1_dataset_corrected.csv'))
    a.add_argument('--output',type=Path,default=Path('results_reviewed'))
    a.add_argument('--fifa-rankings',type=Path,default=Path('fifa_rankings_11_june_2026.csv'))
    a.add_argument('--original',type=Path,default=Path('regression_2_1_dataset.csv'))
    a.add_argument('--repeats',type=int,default=200)
    a.add_argument('--jobs',type=int,default=2,help='Parallel repeated-split processes (1 for slower laptops)')
    args=a.parse_args()
    if args.repeats<2:a.error('--repeats must be >=2')
    run(args.data,args.output,args.fifa_rankings,args.original,args.repeats,args.jobs)
