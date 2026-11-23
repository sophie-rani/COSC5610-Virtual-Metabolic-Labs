# Virtual Metabolic Laboratories in Severe TBI — Protocol v1.0 (frozen 2026-11-19, before test-set access)

## 1. Rationale
Ionized calcium (iCa) and lactate need a blood-gas sample. Routine chemistry (BMP ± Ca, phosphate, Mg, CBC, albumin) is drawn far more often. Bedside surrogates are known to be poor:
- **iCa:** albumin-"corrected" total Ca (Payne 1973) misclassifies calcium status in ICU and trauma patients (Dickerson 2004; Özdemir 2026).
- **Lactate:** anion gap is insensitive to hyperlactatemia (Iberti 1990; Rocktaeschel 2003).

This study asks whether a "soft sensor" (virtual lab) built from routine chemistry estimates iCa and lactate better than the conventional surrogates, measured on identical observations.
- ML estimation of iCa is not novel (e.g. Kumari 2025), so novelty is not claimed.
- The sTBI cohort is the setting, not a claimed mechanism. No sTBI-specific physiology is asserted.

## 2. Hypotheses (two co-primary)
- **H1:** a ridge model from routine chemistry estimates iCa with lower MAE and higher AUROC for iCa < 1.12 mmol/L than calibrated total Ca, on identical observations.
- **H2:** a gradient-boosting model from routine chemistry estimates lactate with lower MAE and higher AUROC for lactate > 2 mmol/L than calibrated anion gap + HCO3, on identical observations.

## 3. Data and cohort
- **Source:** MIMIC-IV v3.1 only. The sTBI cohort uses the frozen v3 rules: 1,334 patients, one ICU stay each.
- **Patient-level split:** salt `discovery-2026-11-18b`, giving dev 815, val 259 and test 260 patients. Every observation from a patient stays in that patient's split.
- **References** (ICU stay window only, [0, ICU LOS]):
  - iCa: itemid 50808, "Free Calcium", blood gas, mmol/L.
  - Lactate: itemid 50813, blood gas, mmol/L. Chemistry lactate (53154) does not occur in this cohort.
- **Pairing:**
  - Each reference is paired with the nearest complete chemistry panel within ±60 min. The panel must have Na, K, Cl, HCO3, BUN, Cr and glucose, and total Ca for the iCa task.
  - Phosphate, Mg, Hgb, WBC and platelets are taken within ±6 h. Albumin is taken within ±24 h and may be missing.
- **Exclusion:** active renal replacement therapy at the reference time (citrate anticoagulation alters iCa).
- **Repeated measures:** all pairs are kept. Inference uses patient-cluster bootstrap (2,000 resamples), and patient-grouped CV is used in dev.

## 4. Predictors (allowed only if knowable without a blood gas)
- **iCa:** total Ca, albumin, Na, K, Cl, HCO3, BUN, Cr, glucose, phosphate, Mg, Hgb, platelets, transfusion ≤6 h, IV calcium ≤6 h.
- **Lactate:** Na, K, Cl, HCO3, AG, BUN, Cr, glucose, Ca, phosphate, Mg, Hgb, WBC, platelets, albumin, vasopressor ≤6 h, transfusion ≤6 h.
- **Never used as predictors:** pH, pCO2, any blood-gas value, ICU time, or the reference itself. pH from the reference blood gas is used only as a subgroup stratifier.
- **Missing values:** median imputation (dev medians) plus missingness indicators. Removing the indicators is a prespecified sensitivity analysis.

## 5. Formulas, thresholds, baselines
| Item | Definition | Source |
|---|---|---|
| Ca unit | mg/dL × 0.2495 = mmol/L | SI conversion |
| Corrected Ca (common) | Ca + 0.8 × (4.0 − albumin) | widely used variant |
| Corrected Ca (Payne) | Ca + 1.0 × (4.0 − albumin) | Payne 1973 |
| Ionized hypocalcemia | iCa < 1.12 mmol/L (severe < 1.00) | Giancarelli 2016 |
| Hyperlactatemia | lactate > 2 mmol/L (severe > 4) | Singer 2016 (Sepsis-3) |
| AG | Na − (Cl + HCO3); AG ≥ 16 rule | Iberti 1990 |

**Baselines**, each linearly calibrated on training data so that the comparison tests information content and not unit offset:
- **iCa:**
  - B1, total Ca (all observations).
  - B2, corrected Ca 0.8, and B3, Payne 1.0, both on albumin-measured observations only.
  - Uncalibrated bedside rules (total or corrected Ca < 8.5 mg/dL) as reported context.
- **Lactate:**
  - B1, AG.
  - B2, AG + HCO3 (log-lactate scale); B2 is the primary comparator.

## 6. Models (compact, each with a reason)
- **Ridge regression:** a transparent linear soft sensor, close to how physiology combines ions. It is the iCa primary model.
- **Regression or classification tree (depth 4):** a human-readable rule set.
- **SVR (RBF):** smooth non-linearity without many hyperparameters.
- **Gradient boosting (HistGB, depth 3):** captures interactions and thresholds. It is the lactate primary model, because lactate is right-skewed and non-linear in AG and HCO3.
- **Logistic regression / Gaussian Naive Bayes / tree / GBM classifiers:** direct threshold classification. Naive Bayes is a weak-independence reference.

Hyperparameters are fixed a priori, apart from the internal alpha path of RidgeCV. Primary models were chosen by dev grouped-CV MAE before the test set was built.

## 7. Metrics
- Regression: MAE, bias (estimate − measured) and calibration slope.
- Classification at fixed thresholds (no threshold tuning): AUROC, AUPRC, sensitivity, specificity, and calibration intercept and slope (classifiers).
- 95% patient-cluster bootstrap CIs.
- Model–baseline differences (ΔAUROC, MAE reduction) on identical observations, with paired bootstrap CIs.

## 8. Prespecified subanalyses
- **iCa:**
  - albumin < 3.0, albumin ≥ 3.0, albumin not measured;
  - transfusion ≤6 h; IV calcium ≤6 h; Cr ≥ 1.5;
  - pH < 7.30 and pH > 7.45 (stratifier only);
  - false reassurance of the uncalibrated corrected-Ca rules in low-albumin hypocalcemia.
- **Lactate:**
  - vasopressor ≤6 h; Cr ≥ 1.5; HCO3 < 22; AG ≤ 12; AG < 16; pH < 7.30;
  - severe (> 4) detection;
  - sensitivity for hyperlactatemia when AG < 16 (the AG rule is 0% sensitive there by definition).
- **Sensitivity analyses:**
  - no missingness indicators;
  - chemistry only (no exposures);
  - basic metabolic panel only;
  - chemistry at or before the reference;
  - one pair per patient-day;
  - IPW for who gets measured (propensity of a blood gas within 1 h of a chemistry panel).

## 9. Supporting analyses
- **PCA** of 10 routine analytes (dev). Each PC is correlated with the analyte and with the residual beyond the conventional baseline.
- **Association rules** (apriori, support ≥ 3%, lift ≥ 1.2) between treatments and abnormalities within 6 h, discovered in dev and replicated in val.
- **Next-day sequential rules.**
- All rules are descriptive associations, not causal effects.

## 10. Freeze and test
1. Hash the code, config and this protocol.
2. Refit primary and secondary models on dev + val.
3. Build test pairs with the same `build.py`.
4. Evaluate once.

No re-tuning, re-selection or threshold change after the test run. All primary outcomes are reported, including negative ones.
