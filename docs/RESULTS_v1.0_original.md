# Virtual Metabolic Laboratories in Severe TBI: Estimating Ionized Calcium and Lactate from Routine Chemistry
**COSC 5610 final results** · MIMIC-IV v3.1 · Protocol v1.0 frozen 2026-11-19 (SHA-256 manifest) · test set evaluated once

## Bottom line
- **Lactate (strong, robust):** a gradient-boosting soft sensor built from routine chemistry discriminated hyperlactatemia (> 2 mmol/L) clearly better than calibrated anion gap + HCO3.
  - Test AUROC 0.87 vs 0.77; ΔAUROC +0.10 (95% CI 0.05–0.15).
  - MAE reduction 0.28 mmol/L (0.14–0.47).
  - The gain was largest where the anion gap is blind. In normal-AG observations (AG < 16), the model flagged 62% of hyperlactatemia at 89% specificity; the AG ≥ 16 rule flags 0% there by definition.
- **iCa (modest, real):** a ridge soft sensor beat calibrated total Ca.
  - Test AUROC 0.84 vs 0.80; ΔAUROC +0.042 (0.006–0.079).
  - The MAE gain was marginal: 0.053 vs 0.057 mmol/L, CI 0.000–0.008.
- **Albumin correction misleads:**
  - Used as a bedside rule, it falsely reassured in **81% (0.8 formula) and 93% (Payne)** of hypoalbuminemic patients with true ionized hypocalcemia. Total Ca falsely reassured in 19% and the soft sensor in 12%.
  - Even after recalibration, correction never improved and sometimes worsened discrimination relative to uncorrected total Ca.
- **What ML adds:** the gain comes from *combining routine ions in a physiologically coherent way*, not from model complexity.
  - Ridge matched GBM for iCa; a single tree and Naive Bayes never beat the baselines.
  - Neither soft sensor is accurate enough to replace a blood gas. Its realistic role is triage: deciding when a blood gas is worth drawing, and flagging misleading surrogates.

## Methods (summary; full protocol in `PROTOCOL_v1.0_FROZEN`)
- **Cohort and split:**
  - 1,334 adult sTBI ICU patients (frozen v3 cohort rules), one ICU stay each.
  - Patient-level split: dev 815, val 259, test 260.
- **References and pairing:**
  - Blood-gas iCa (50808) and lactate (50813) obtained during the ICU stay.
  - Each was paired to the nearest complete chemistry panel within ±60 min. Phosphate, Mg and CBC were taken within ±6 h; albumin within ±24 h, optional.
  - Pairs with active renal replacement therapy were excluded.
- **Predictors:** routine chemistry plus 6-h treatment exposures (transfusion, IV calcium, vasopressor). No blood-gas variable or ICU time was used. pH served only as a stratifier.
- **Baselines** were linearly calibrated on the training data, which makes them *stronger* than raw bedside formulas:
  - iCa: total Ca; albumin-corrected Ca (0.8 and Payne 1.0) on albumin-measured observations.
  - Lactate: AG; AG + HCO3.
- **Models:** Ridge, depth-4 tree, RBF-SVR and HistGB regressors; logistic, Gaussian NB, tree and HistGB classifiers.
  - Primary models were chosen by dev patient-grouped CV MAE: Ridge for iCa, GBM for lactate.
  - Final models were refit on dev + val.
- **Thresholds** were fixed a priori: iCa < 1.12 mmol/L; lactate > 2 mmol/L.
- **Inference:** 2,000-resample patient-cluster bootstrap, with all comparisons on identical observations.
- **Evaluation data:**
  - Test: iCa 557 pairs / 147 patients (44.7% < 1.12); lactate 471 pairs / 137 patients (45.9% > 2).
  - Development: iCa 2,298 pairs / 567 patients; lactate 1,980 / 564.

## Table 1. Primary results, untouched test set (95% patient-cluster CI)
| Task | Estimator | n | MAE (mmol/L) | Bias | AUROC | AUPRC | Sens | Spec |
|---|---|---|---|---|---|---|---|---|
| iCa | **Ridge soft sensor** | 557 | **0.053** (0.043–0.064) | 0.000 | **0.839** (0.804–0.869) | 0.808 | 0.77 | 0.79 |
| iCa | GBM soft sensor | 557 | 0.053 | 0.002 | 0.830 | 0.810 | 0.76 | 0.78 |
| iCa | Total Ca, calibrated | 557 | 0.057 (0.047–0.069) | 0.001 | 0.797 (0.757–0.833) | 0.770 | 0.70 | 0.76 |
| iCa | Corrected Ca 0.8, calibrated* | 172 | 0.063 | −0.006 | 0.745 (0.656–0.827) | 0.691 | 0.71 | 0.64 |
| iCa | Payne Ca 1.0, calibrated* | 172 | 0.064 | −0.008 | 0.713 (0.622–0.797) | 0.649 | 0.71 | 0.61 |
| Lactate | **GBM soft sensor** | 471 | **0.98** (0.79–1.20) | −0.51 | **0.874** (0.831–0.911) | 0.856 | 0.72 | 0.87 |
| Lactate | SVR soft sensor | 471 | 0.99 | −0.53 | 0.881 | 0.870 | 0.74 | 0.88 |
| Lactate | Ridge soft sensor | 471 | 1.08 | −0.62 | 0.852 | 0.848 | 0.67 | 0.85 |
| Lactate | AG + HCO3, calibrated | 471 | 1.27 (0.98–1.62) | −0.83 | 0.774 (0.716–0.832) | 0.760 | 0.57 | 0.84 |
| Lactate | AG alone, calibrated | 471 | 1.37 | −0.86 | 0.710 | 0.693 | 0.49 | 0.82 |

\*albumin-measured subset. Sensitivity and specificity are at the fixed analyte threshold applied to the continuous estimate. All models are in `final/test_*_models.csv`.

## Table 2. Head-to-head differences on identical observations (test)
| Comparison | n | ΔAUROC (95% CI) | MAE reduction, mmol/L (95% CI) |
|---|---|---|---|
| iCa: Ridge vs total Ca **(primary H1)** | 557 | **+0.042 (0.006, 0.079)** | 0.004 (0.000, 0.008) |
| iCa: Ridge vs corrected Ca 0.8 (key secondary) | 172 | +0.084 (0.002, 0.168) | 0.007 (−0.005, 0.015) |
| iCa: corrected Ca 0.8 vs total Ca | 172 | −0.057 (−0.120, 0.010) | −0.004 (−0.010, 0.004) |
| iCa: Payne Ca 1.0 vs total Ca | 172 | −0.089 (−0.166, −0.014) | −0.005 (−0.012, 0.004) |
| Lactate: GBM vs AG + HCO3 **(primary H2)** | 471 | **+0.100 (0.048, 0.155)** | 0.28 (0.14, 0.47) |
| Lactate: AG alone vs AG + HCO3 | 471 | −0.064 (−0.116, −0.013) | −0.11 (−0.18, −0.04) |

The same direction held in dev CV and val for both primary hypotheses: iCa +0.051 / +0.077; lactate +0.062 / +0.076.

## Table 3. Uncalibrated bedside rules, iCa, albumin-measured test observations (n = 172; 75 hypocalcemic; 42 with albumin < 3.0 and hypocalcemia)
| Rule | Sensitivity | Specificity | False reassurance when albumin < 3.0 |
|---|---|---|---|
| Total Ca < 8.5 mg/dL | 80% | 58% | 19% |
| Corrected Ca (0.8) < 8.5 mg/dL | 37% | 94% | **81%** |
| Payne Ca (1.0) < 8.5 mg/dL | 25% | 95% | **93%** |
| Soft sensor < 1.12 mmol/L | 80% | 72% | 12% |

The same pattern appeared in dev (75% / 90% false reassurance) and val (83% / 97%).

## Subgroups and robustness (Figure 3)
- **iCa:**
  - The gain was consistent in direction across subgroups but individually imprecise.
  - The gain was clearest where albumin was *not measured* (+0.050, 0.013–0.088): exactly where clinicians cannot apply a correction formula.
  - No gain in albumin ≥ 3.0 (+0.005). There, total Ca is already a decent surrogate.
- **Lactate:**
  - Gains in HCO3 < 22 (+0.13), AG ≤ 12 (+0.12), AG < 16 (+0.13) and Cr ≥ 1.5 (+0.16).
  - The pH < 7.30 subgroup gain seen in val (+0.18) **did not replicate** on test (−0.002). In overt acidemia, AG + HCO3 already performs well (AUROC 0.89).
- **Severe hyperlactatemia (> 4 mmol/L):** AUROC 0.90 vs 0.86.
- **Prespecified sensitivity analyses:** all stayed positive for both analytes, with ΔAUROC between +0.035 and +0.051 for iCa and +0.079 and +0.117 for lactate. They were:
  - no missingness indicators;
  - chemistry only (no treatment flags);
  - basic metabolic panel only;
  - chemistry drawn at or before the blood gas;
  - one pair per patient-day;
  - inverse-probability weighting for who gets tested.
- **Ordering patterns:** the signal is not driven by ordering patterns. Removing treatment flags and missingness indicators left the gain unchanged.

## What the models learned (supporting analyses; association only)
- **iCa: chloride is the second-strongest predictor after total Ca** (test permutation importance: Ca 0.038, Cl 0.035, Na 0.018).
  - In dev and val regressions adjusted for total Ca, albumin, AG and phosphate, a higher Na − Cl was associated with lower iCa: −0.005 and −0.008 mmol/L per mEq/L.
  - Adding blood-gas pH attenuated this by 25–33%.
  - Na − Cl approximates the strong-ion difference. A more alkaline acid–base state increases albumin binding of Ca, lowering the ionized fraction.
  - **The model partly recovers acid–base information that albumin correction ignores.** Two-thirds of the effect is unexplained by measured pH; this is a hypothesis, not a demonstrated mechanism.
- **PCA (10 routine analytes, dev):**
  - PC2, a Na/Cl axis, carried the iCa information *beyond total Ca*: correlation with the residual r = 0.25 dev, 0.32 val.
  - PC3, an HCO3/Ca/Mg axis, correlated with iCa itself (r ≈ 0.4) but not with the residual, so the baseline already captures it.
  - For lactate, PC3 (HCO3) tracked lactate (r ≈ −0.4) but was largely absorbed by AG + HCO3. The residual information was spread across glucose, platelets, creatinine and phosphate, consistent with the GBM importances (AG, HCO3, glucose, platelets, Cr).
- **Association rules (dev discovery → val replication):**
  - Lactate > 2 co-occurred with AG ≥ 16, phosphate > 4.5, transfusion and vasopressor exposure (val lift 1.4–2.1).
  - iCa < 1.12 co-occurred with AG ≥ 16, Cr ≥ 1.5 and phosphate > 4.5 (val lift 1.4–1.9).
  - The transfusion → low iCa association did **not** replicate (val lift 0.97).
  - Next-day sequential rules (treatment day d → new abnormality day d+1) were underpowered (≤ 64 exposed patient-days) and inconclusive.
  - These are co-occurrence patterns subject to confounding by indication, not treatment effects.

## Interpretation: what ML adds to conventional metabolic monitoring
1. **It replaces a harmful heuristic with a better one.** Albumin-corrected calcium, a reflexive bedside habit, hid ionized hypocalcemia in most hypoalbuminemic patients. A transparent linear model using routine ions cut false reassurance to about 12%.
   - The defensible clinical message is *not* "use ML"; it is "don't trust corrected calcium, and if you need a surrogate, combine Ca with Na/Cl."
2. **It sees lactate the anion gap misses.** Normal-AG hyperlactatemia is common in this cohort: 157 of 216 test hyperlactatemic observations had AG < 16. The model identified about 60% of them.
   - This is the clearest case for a "virtual lab": prompting a lactate draw when routine chemistry looks unremarkable.
3. **Complexity buys little.** Ridge ≈ GBM for iCa; GBM ≈ SVR > Ridge for lactate (modest non-linearity).
   - The added value is information fusion across analytes, which classic single-formula surrogates cannot do.
4. **It does not replace measurement.** The residual iCa error (~0.05 mmol/L) is about half the normal range; lactate sensitivity at 2 mmol/L is about 0.72. Both are triage or flagging tools.

## Limitations
- **Single center, no external validation:** MIMIC-IV is one institution. Analyzer-specific relations (e.g. the Na/Cl–iCa link) may not transport.
- **Selection:** only panels where clinicians obtained a blood gas are labeled (24% of ICU panels for iCa, 19% for lactate; selection AUROC 0.67–0.73 on test). IPW did not change conclusions, but results describe *tested* patients.
- **Lactate retransformation bias:** models fit on log-lactate underestimate the arithmetic mean (test bias −0.5 mmol/L for the model, −0.8 for the baseline). AUROC is unaffected, but fixed-threshold sensitivity is understated. Not corrected post hoc, by design.
- **Albumin is measured in ~35% of pairs:** the corrected-Ca comparisons rest on 172 test observations from 69 patients.
- **Effect sizes:** the iCa gain shrank from val to test (+0.077 → +0.042), and the MAE improvement is marginal. Pairing within ±60 min admits some chemistry drawn after the blood gas (9–13%); restricting to prior chemistry did not change results.
- **No sTBI-specific claim:** the cohort is the setting. Nothing here shows that these relations differ in brain injury.
- **Prior work:** ML estimation of iCa has been reported (e.g. Kumari 2025); the contribution here is a leakage-audited, baseline-fair comparison with a frozen, once-used test set.

## Key references
- Payne RB et al. *BMJ* 1973;4:643. doi:10.1136/bmj.4.5893.643
- Dickerson RN et al. *JPEN* 2004;28:133. doi:10.1177/0148607104028003133
- Özdemir et al. *PLoS One* 2026. doi:10.1371/journal.pone.0354233
- Iberti TJ et al. *Crit Care Med* 1990;18:275. doi:10.1097/00003246-199003000-00005
- Rocktaeschel J et al. *Crit Care Med* 2003;31:2131. doi:10.1097/01.CCM.0000079819.27515.8E
- Singer M et al. *JAMA* 2016;315:801. doi:10.1001/jama.2016.0287
- Giancarelli A et al. *J Surg Res* 2016;202:182. doi:10.1016/j.jss.2015.12.036
- Kumari et al. *Indian J Clin Biochem* 2025. doi:10.1007/s12291-024-01182-3

## Reproducibility
- Code: `config.py`, `build.py`, `models.py`, `metrics.py`, `run_dev.py`, `aux.py`, `phase3.py`, `final.py`, `figures.py`.
- Order of runs: `build.py dev,val` → `run_dev.py` → `aux.py` → `phase3.py` → freeze → `build.py test` → `final.py` → `figures.py`.
- Seed 20261119. The freeze manifest with SHA-256 hashes is checked inside `final.py` before any test computation.
