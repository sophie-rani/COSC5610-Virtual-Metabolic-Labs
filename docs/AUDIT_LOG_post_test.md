# Audit Log, Part 2: after the single test run (2026-11-19)
- The freeze manifest hashes were verified by `final.py` before it ran. The test set was built and evaluated once.
- No method, model, threshold or feature was changed after the run. The only post-test edits are presentational: figure axis ticks and titles.

| # | Issue found on proofreading test output | Sev | Action | Changes conclusion? |
|---|---|---|---|---|
| P1 | The `tee` log file was not written because the `final/` directory did not exist yet. The CSV outputs and console output are complete | L | CSVs are the record of the run. Not re-run, to preserve the "exactly once" rule | No |
| P2 | Lactate estimates are biased low in every model, test bias −0.51 (GBM) and −0.83 (AG + HCO3). Cause: models fit on log-lactate and back-transformed with exp(), which estimates the geometric, not arithmetic, mean (retransformation bias). Test prevalence was also higher (46% vs 33–42% in dev/val) | M | Reported, not corrected. It affects MAE and the fixed-threshold sensitivity of model and baseline alike. AUROC is unaffected. Future work: smearing correction, prespecified | No for discrimination. The fixed-threshold sensitivity is understated |
| P3 | The iCa test effect is smaller than on val (ΔAUROC +0.042 vs +0.077). The MAE reduction CI touches 0 (0.000–0.008) | M | Reported as is. The primary discrimination hypothesis holds; the MAE gain is marginal | Tempers H1: discrimination yes, magnitude modest |
| P4 | Lactate pH < 7.30 subgroup: val ΔAUROC +0.18, test −0.002 (n = 76, 45 patients) | L | Reported as a non-replicating subgroup. In acidemia, AG + HCO3 already discriminates well (AUROC 0.89) | Removes a val-only claim |
| P5 | iCa one-pair-per-patient-day sensitivity CI touches 0 (−0.000 to 0.069) | L | Reported. The point estimate is consistent | No |
| P6 | Corrected (0.8) vs total Ca on test: ΔAUROC −0.057 (−0.12, 0.01), NS. Payne vs total: −0.089 (−0.17, −0.01), significant | L | Wording: correction "never improved and sometimes worsened" discrimination | No |
| P7 | Naive Bayes did well for lactate on test (AUROC 0.835) but is badly calibrated (slope 0.29) | L | Not interpreted | No |
| P8 | Measurement selection on test panels: AUROC 0.67 (iCa) and 0.73 (lactate). Selection is stronger than in val; IPW results are similar | M | Results apply to panels where clinicians obtained a blood gas | Scope limitation |
