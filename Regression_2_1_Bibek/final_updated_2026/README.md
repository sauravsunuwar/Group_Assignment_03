HIT140 Regression 2.1 - Updated model run (15 October 2026)
This folder contains an updated report and reproducible modelling outputs from `regression_2_1_dataset_latest.csv`. The previous `results_reviewed` data and report are superseded for current model-performance values; preserve them as an archive rather than mixing tables from the two runs.
What is included
`Regression_2_1_Updated_Validated_Report.docx`: updated model comparison, uncertainty, assumptions and provenance notes.
`regression_2_1_modeling_reviewed.py`: actual updated modelling script (including match-ID diagnostic correction).
`regression_2_1_dataset_latest.csv`: current 104-match, eight-predictor dataset used for the new run.
`results_latest/`: all 25 generated output charts, CSVs, and run metadata.
`source_evidence/`: corrected 240-row five-match history, recalculated 48-team baseline CSV, ranking/master snapshots and supplied upstream Python scripts. Keep these source files distinct from the older same-named CSVs present in prior project ZIP files.
Running from PowerShell
Open PowerShell in this folder and run:
```powershell
python regression_2_1_modeling_reviewed.py --data regression_2_1_dataset_latest.csv --output results_latest --repeats 200
```
Required Python libraries: numpy, pandas, scikit-learn, scipy, statsmodels, matplotlib and joblib.
Verified numerical summary (200 repetitions per train/test ratio)
Lasso, 70/30: mean held-out R2 = 0.3210; SD = 0.1781; NRMSE = 18.70%.
Lasso, 80/20: mean held-out R2 = 0.3153; SD = 0.2209; NRMSE = 20.54%.
Fixed seed 42: Lasso leads the 70/30 holdout (R2 = 0.307), Random Forest leads the 80/20 holdout (R2 = 0.365).
Scientific and handoff caveats
The 104 rows represent World Cup matches, not independent teams. The 200 train/test resamples overlap; their SDs describe split sensitivity, not independent-replication confidence intervals.
Eight input columns remain available to Lasso, but some fitted Lasso coefficients are exactly zero. If the rubric demands eight nonzero effects, Ridge is the alternative regularised model; confirm with lecturer/group.
The FIFA publication time (11 June 2026) is supported by an official FIFA release published before the competition began; the project CSV's 48 numeric entries have not each been independently checked against an archived official ranking table.
The historical five-match CSV and recalculated baselines in `source_evidence/` are the newer source files. The old ones from earlier project ZIPs contain post-opening-date matches and must not be labelled as the direct sources of this updated model.
The team may need to document the provenance of `international_results.csv` and ensure full end-to-end reconstruction uses the approved shared dataset.
GitHub suggestion
For safe collaboration, upload this as a separate latest version within `Regression_2_1_Bibek` or merge only the agreed updated files, retaining the old `reviewed_2026` directory as an archive. Do not blindly overwrite teammates' `data/` folders.
Official FIFA publication source: https://grassroots.fifa.com/fifa-rankings/world-ranking/men/news/argentina-reclaim-top-spot-mens-world-ranking-fifa-coca-cola
