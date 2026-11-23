# Audit Log — Virtual Metabolic Labs (sTBI, MIMIC-IV)
Severity scale: **H** (would invalidate a conclusion), **M** (would bias magnitude), **L** (presentation or robustness).

All entries below were made before the test set was built (dev/val only).

## A. Clinical-validity audit
| # | Issue | Sev | Correction / check | Changes conclusion? |
|---|---|---|---|---|
| A1 | Units: iCa (50808) is mmol/L; total Ca (50893) is mg/dL | H | Converted Ca × 0.2495; checked iCa median 1.12 mmol/L | No (prevented an error) |
| A2 | Corrected-Ca formula: the earlier pipeline used only the 0.8 factor; Payne's original is 1.0 | M | Both formulas tested as B2 and B3 | No. Both are worse than total Ca (C3) |
| A3 | The corrected-Ca comparison must use observations with albumin actually measured | H | B2/B3 are evaluated only where albumin was measured within ±24 h (dev 598, val 201). ΔAUROC is computed on the identical subset | No |
| A4 | Was 50808 reported at the actual pH or normalized to pH 7.4? | M | Adjusted iCa–pH coefficient is negative (dev −0.14, val −0.27 per pH unit), as expected for actual-pH iCa | No |
| A5 | Citrate (RRT, massive transfusion) chelates Ca, an alternative explanation | M | Active-RRT pairs excluded (iCa n −242). Transfusion and IV calcium ≤6 h kept as predictors and subgroups | No |
| A6 | Thresholds | L | iCa < 1.12 (Giancarelli 2016); lactate > 2 (Sepsis-3); AG ≥ 16 (Iberti 1990); corrected Ca < 8.5 mg/dL (conventional lower limit). All fixed, none tuned | No |
| A7 | Pre-ICU references were included in an earlier version, before exposures were recorded | M | Restricted to the ICU window [0, LOS] | No (effect sizes are similar) |
| A8 | Lactate source: chemistry lactate 53154 | L | 0 occurrences in this cohort, so only 50813 is used | No |
| A9 | sTBI-specific claims | M | None made. The cohort is the setting. Results generalize at most to ICU patients with similar testing patterns | Framing only |

## B. Statistical / ML audit
| # | Issue | Sev | Correction / check | Changes conclusion? |
|---|---|---|---|---|
| B1 | Patient overlap across splits | H | 1 ICU stay per subject; split by patient; zero overlap verified | No |
| B2 | Blood-gas leakage (pH, pCO2 from the same ABG) | H | Excluded from predictors. pH is used only as a stratifier | No |
| B3 | Time-in-ICU (`hrs`) feature in an early lactate model, which proxies ordering intensity | M | Dropped | No |
| B4 | Post-reference chemistry (panel drawn after the ABG) | M | 9–13% of pairs. Sensitivity "chemistry at/before reference" gives ΔAUROC iCa +0.084, lactate +0.072 (val) | No |
| B5 | Repeated measures (median 2–3 pairs per patient, max 35) | M | Grouped CV, cluster bootstrap, and a one-pair-per-patient-day sensitivity: iCa +0.084, lactate +0.108 | No |
| B6 | Ordering patterns via missingness indicators | M | Removing the indicators leaves iCa ΔAUROC at +0.076 and lactate at +0.078 | No |
| B7 | Treatment-indication leakage (IV Ca given because iCa was low) | M | Chemistry-only model (no exposures) gives iCa +0.077 and lactate +0.063. The IV-calcium subgroup shows no gain for iCa (−0.002), so the gain does not come from treatment flags | No |
| B8 | Who gets tested (selection) | M | A propensity model of a blood gas within 1 h of chemistry has val AUROC 0.57 (iCa) and 0.65 (lactate): weak selection. IPW-weighted ΔAUROC is iCa +0.076, lactate +0.066 | No. Results apply to *tested* panels |
| B9 | Test set used for selection | H | Primary models (Ridge for iCa, GBM for lactate) chosen by dev grouped-CV MAE. Val was used only to confirm. Test is not built until the freeze | No |
| B10 | Baseline fairness | H | Baselines are linearly calibrated on the same training data, so they are *stronger* than bedside formulas. Comparisons are always on identical observations | No |
| B11 | Calibration of the corrected-Ca baselines was unstable (dev CV slope 0.40, val 1.86) | L | Expected from small subsets plus weak signal. Reported, not fixed | No |
| B12 | Naive Bayes is badly calibrated (slope 0.16–0.30) | L | Kept as a reference model. Not used for conclusions | No |

## C. Phase-4 proofreading of findings (dev/val)
Each finding is checked against six questions: (1) plausible? (2) learned or built into the label? (3) explained by timing, treatment or missingness? (4) baseline fair? (5) clinically meaningful? (6) overstated?

**C1. The iCa soft sensor beats calibrated total Ca** (val ΔAUROC +0.077 [0.04, 0.12]; MAE 0.047 vs 0.056 mmol/L).
- (1) Plausible: total Ca explains ~30% of iCa variance.
- (2) Learned: the label is not a function of the inputs.
- (3) Robust to all sensitivity analyses (B4–B8).
- (4) The baseline is calibrated, so it is fair.
- (5) The MAE gain of ~0.01 mmol/L is small in absolute terms, yet sensitivity at 1.12 rose from 0.73 to 0.80 with similar specificity.
- (6) Do not claim it replaces a blood gas. Residual MAE of ~0.05 mmol/L is about half the width of the normal range.

**C2. Chloride is the top iCa predictor after total Ca** (Ridge: +Cl, −Na).
- Adjusted for total Ca, AG, phosphate and albumin, higher Na − Cl is associated with lower iCa (dev −0.0052, val −0.0078 mmol/L per mEq/L).
- Adding pH attenuates this by ~25–33%.
- Interpretation: Na − Cl approximates the strong-ion difference, and alkalinizing SID increases albumin binding of Ca (Stewart acid–base). The model partly recovers acid–base information that albumin correction ignores.
- About two-thirds of the effect is unexplained by measured pH.
- **Label as a hypothesis-generating mechanism, not proven.**

**C3. Albumin correction misleads.**
- On identical albumin-measured observations, calibrated corrected Ca is *worse* than calibrated total Ca: val ΔAUROC −0.087 (0.8) and −0.129 (Payne). In dev the 0.8 version is similar to total Ca.
- With uncalibrated bedside rules, in patients with albumin < 3.0 and true ionized hypocalcemia, corrected Ca ≥ 8.5 mg/dL ("normal") falsely reassured in 75% (dev) and 83% (val) with the 0.8 formula, and 90% and 97% with Payne. The total Ca rule falsely reassured in 11% and 4%, and the model in 24% and 6%.
- Consistent with Dickerson 2004 and Özdemir 2026.
- Clinically important.
- Caveat: albumin is measured in only ~35% of pairs, a selected subgroup.

**C4. The lactate soft sensor beats calibrated AG + HCO3** (val ΔAUROC +0.076 [0.03, 0.12]; MAE 0.80 vs 0.97 mmol/L).
- The gain is largest where conventional tools are blind: AG ≤ 12 (+0.092) and pH < 7.30 (+0.18).
- With AG < 16, the model flags 57–59% of hyperlactatemia at 84% specificity; the AG ≥ 16 rule flags none by construction.
- (2) AG partly *contains* lactate as an unmeasured anion. This is physiology, not label leakage, because AG is a routine value. Even so, the comparator also uses AG, so the gain is beyond AG.
- (6) Sensitivity of ~0.65–0.70 at 2 mmol/L is not enough to replace measurement. It is a "who should get a lactate" triage tool.

**C5. Association rules.**
- Lactate > 2 associates with AG ≥ 16, transfusion, vasopressors and phosphate > 4.5, replicated in val with lift 1.4–2.1.
- iCa < 1.12 associates with AG ≥ 16, Cr ≥ 1.5 and phosphate > 4.5, replicated (lift 1.4–1.9).
- Transfusion → low iCa did *not* replicate (val lift 0.97).
- Next-day sequential rules are underpowered (≤ 64 exposed patient-days) and inconclusive. Reported as null or inconclusive.
- **Associations only.** Confounding by indication: sicker patients get transfusions and vasopressors.

**C6. Model ranking.**
- Linear (Ridge or logistic) ≈ GBM for iCa; GBM ≈ SVR > Ridge for lactate.
- The single decision tree and Naive Bayes never beat the calibrated baseline.
- The added value comes from combining ions, not from model complexity, except a modest non-linearity gain for lactate.

**C7. The BMP-only iCa model did slightly better on val than the full model** (0.902 vs 0.884).
- Noted. The primary model is **not** changed, to avoid selecting on val.

## D. Decisions frozen before test
- Primary: Ridge (iCa) vs B1 total Ca; GBM (lactate) vs B2 AG + HCO3.
- Key secondary: iCa model vs B2 corrected Ca (0.8) on albumin-measured observations.
- Final models are refit on dev + val. Thresholds stay fixed at 1.12 and 2.0.
