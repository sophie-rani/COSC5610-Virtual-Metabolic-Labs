# QA Report: v1.1 post hoc methodological revision
*Virtual Metabolic Laboratories in sTBI (MIMIC-IV), COSC 5610, 2026-11-19*

**Scope**
- Executed notebook: `VML_sTBI_MIMIC-IV_COSC5610_v1.1_corrected.ipynb`.
- Pre-registered amendment: `docs/PROTOCOL_AMENDMENT_v1.1_posthoc.md`. It was SHA-256-locked before any corrected result existed, and the notebook verifies the hash.

**Preserved**
- Frozen v1.0 code (hash-verified).
- Frozen results and audit logs.
- The executed v1.0 notebook, in `original_v1.0/`.

**Status:** this is a **post hoc revision evaluated on the same held-out patients**, not a new independent test. No threshold, model or variant was chosen on test performance.

## 1. What changed
| # | Issue | Bug or assumption? | Correction | Did it change results? |
|---|---|---|---|---|
| C1 | Phosphate, Mg, CBC and albumin paired to the *nearest* value (±6 h / ±24 h), and chemistry within ±60 min. 9–35% were drawn **after** the blood gas | **Bug** for any real-time claim (temporal leakage) | **V2:** predictors collected ≤ t₀. **V3 (primary corrected):** predictors whose results were *available* (`labevents.storetime`) ≤ t₀, chemistry ≤ 6 h old. Verified: 0% collected or resulted after t₀ | **Yes.** Fewer observations, lower absolute accuracy. Gains vs baseline persist (§2) |
| C2 | Ridge alpha tuned by leave-one-*row*-out (RidgeCV), so one patient's rows sat on both sides of an internal split | **Bug** (repeated-measures leakage in tuning) | GroupKFold(5) by patient, same grid and criterion. Alpha rose from 5.6 to 17.8 for iCa and from 100 to 178 for lactate. No other model has internal tuning; GBM early stopping is verified off | **No.** iCa test ΔAUROC 0.042 → 0.042 |
| C3 | `rrt6` read from `inputevents`, so always 0 | **Bug.** Not a predictor; the exclusion (`rrt_active`, `procedureevents`) was unaffected | Read from `procedureevents` (250 iCa / 192 lactate pairs now flagged before exclusion). Independent check vs CRRT-mode / citrate / CRRT-calcium / filter-change charting within ±2 h: 236/242 excluded iCa pairs had evidence; **61/2,298 retained iCa pairs (2.7%) and 43/1,980 lactate pairs had evidence**. Pre-declared sensitivity analysis excludes them | Small. iCa V3 ΔAUROC 0.058 → 0.054 (CI touches 0); lactate 0.095 → 0.113 |
| C4 | Bootstrap B = 300 (dev/val), 1,000 (test subgroups/sensitivity) vs 2,000 in protocol | **Protocol deviation** | All v1.1 intervals use **B = 2,000** patient-cluster resamples. v1.0 counts documented as run | No (point estimates unchanged) |
| C5 | Lactate estimates biased low | **Expected property**, not a bug | Explained, not corrected (§4) | No |
| C6 | Sequential-rule columns mislabelled (`n_val` = exposed days, `support_dev` = eligible days) | **Reporting bug** | Relabelled (`n_eligible_days`, `n_exposed_days`, `p_abnormal_next_day_if_exposed`, `lift`). Values identical to frozen (checked) | No |
| C7 | Units and missingness not reported | QA | One unit per analyte (from `valueuom`). Required chemistry never missing; ancillary labs imputed with training medians plus indicators (no test information) | No |

**Clinical assumptions** (justified, not bugs):
- **Prediction time** = blood-gas collection time.
- **Treatments** count if started ≤ t₀ and running in the prior 6 h, assuming a clinician knows about an infusion that is already running. Median charting lag is 0–3 min; p90 is 1.4–2.6 h.
- **`storetime`** is treated as the result-availability time. It is never missing; 3 of 290,266 rows (one blood-gas specimen, never a predictor) predate collection; chemistry turnaround median 68 min; MIMIC-IV, Johnson 2023.
- **6-h chemistry lookback in V3.** A 60-min lookback is incompatible with the 68-min chemistry turnaround; 6 h is the frozen window for other routine labs.
- **IV calcium** includes CRRT calcium (227525), as frozen.

## 2. Original vs corrected: primary contrast, test set (95% patient-cluster CI)
| Variant | iCa n | iCa AUROC model vs total Ca | iCa ΔAUROC | Lactate n | Lactate AUROC model vs AG + HCO3 | Lactate ΔAUROC |
|---|---|---|---|---|---|---|
| V0 original (frozen) | 557 | 0.839 vs 0.797 | +0.042 (0.006, 0.079) | 471 | 0.874 vs 0.774 | +0.100 (0.048, 0.155) |
| V1 ML fixes | 557 | 0.839 vs 0.797 | +0.042 (0.008, 0.076) | 471 | 0.874 vs 0.774 | +0.100 (0.049, 0.154) |
| V2 collected ≤ t₀ | 520 | 0.857 vs 0.807 | +0.050 (0.019, 0.083) | 439 | 0.881 vs 0.766 | +0.115 (0.064, 0.166) |
| **V3 resulted ≤ t₀ (primary corrected)** | **307** | **0.763 vs 0.705** | **+0.058 (0.004, 0.115)** | **346** | **0.850 vs 0.754** | **+0.095 (0.033, 0.162)** |

V3 evaluates a different, smaller set of observations (blood gases with a resulted chemistry panel in the prior 6 h; chemistry median about 4 h old), so it is not a paired comparison with V0.

## 3. Do the main results still hold?
**Lactate: yes, robustly.**
- Under realistic availability the GBM soft sensor beats calibrated AG + HCO3 by ΔAUROC +0.095 (0.033, 0.162). The MAE is 1.26 vs 1.59 mmol/L.
- The gain stays positive in every V3 sensitivity analysis (+0.067 to +0.113, all CIs above 0).
- It is largest where the anion gap is blind. With AG < 16, the model flags 65% of hyperlactatemia at 82% specificity.
- The acidemia subgroup (pH < 7.30) again shows no gain.

**iCa: yes, but modest and fragile.**
- V3 ΔAUROC is +0.058 (0.004, 0.115), with the lower bound near zero. Absolute discrimination falls from 0.84 to 0.76 once chemistry must already be resulted.
- Three pre-declared V3 sensitivity analyses cross zero:
  - one pair per patient-day: +0.023 (−0.029, 0.076);
  - excluding independent CRRT/citrate evidence: +0.054 (−0.001, 0.110);
  - IPW for who gets tested: +0.050 (−0.007, 0.113).
- Report it as "consistent direction, weak evidence."
- Calcium and chloride remain the top predictors.

**Albumin correction: directionally unchanged, no longer statistically supported under V3.**
- Corrected Ca still discriminates worse than total Ca: AUROC 0.58 vs 0.65.
- It still falsely reassures most hypoalbuminemic patients with true hypocalcemia: 89% with either formula, vs 32% for total Ca and 21% for the model.
- But only 56 albumin-measured V3 test observations remain, 19 of them hypoalbuminemic hypocalcemic, so all CIs include 0.
- The strong v1.0 evidence came from albumin paired ±24 h, which partly postdated the blood gas.

## 4. Lactate bias (explained, not corrected)
- **Cause:** the model is fit on log(lactate). exp() of a log-scale estimate is a geometric-mean, or median-type, estimate. Lactate is right-skewed, so the mean bias is negative by construction.
- **Dev evidence (out-of-fold):**
  - mean bias −0.50 mmol/L, but median bias −0.06 and log-scale bias −0.03;
  - Duan smearing estimated from training residuals (factor 1.20) removes the dev mean bias (+0.02).
- **Val:** smearing would overshoot, from −0.12 to +0.34, so it is not uniformly better.
- **Test:** mean bias is −0.79. Even with training-derived smearing it would be −0.35, because the test patients have higher lactate (prevalence 59% vs 51% dev and 38% val). The remainder is case-mix shift, not retransformation.
- **What was not done:** smearing was not applied, because it would change the prespecified estimator. AUROC is unaffected by any monotone rescaling. A median-type estimate is also the natural target for MAE.

## 5. Verification
**Reproducibility and leakage checks: all PASS**
- Cohort, split and lab values are identical to frozen v1.0.
- V1 pairs reproduce the frozen pairs exactly (apart from `rrt6`).
- No patient is shared between dev/val and test.
- No V2/V3 predictor was collected or resulted after t₀.
- Units are consistent.

**Agreement between outputs**
- Figure 1 AUROCs equal the table values; Figure 5 is drawn from the comparison table.
- Every number in §2–4 of this report was checked against `results/key_numbers_v1.1.json` from the executed run.

## 6. Remaining concerns that could affect grading or publication
1. **No independent test.** The 260 test patients were analysed in v1.0 before these corrections. The v1.1 test numbers are revised estimates, not confirmation. *For publication this is the main gap.* External validation is needed: eICU-CRD, or MIMIC admissions outside this cohort, with a new pre-registration.
2. **The iCa claim is weak under realistic timing.** It should be presented as modest, with its fragile sensitivity analyses, not as a headline.
3. **The corrected-Ca finding was not re-tested with causally timed albumin** (V2). Only V3 was pre-declared for the bedside-rule analysis, and it is underpowered (n = 56). A V2 check would be a new post hoc analysis and was not run.
4. **V3 changes the population.** Blood gases without a resulted chemistry panel in the prior 6 h are dropped (iCa dev/val 1,205 of 2,298 pairs retained). Results apply to "blood gases drawn when a recent chemistry is already back."
5. **The RRT definition misses some citrate exposure** (2.7% of retained iCa pairs had CRRT/citrate charting). This is handled by the pre-declared sensitivity analysis, not by changing the primary exclusion.
6. **Treatment timing rests on start times**, assuming running infusions are known before they are charted.
7. **Lactate calibration-in-the-large shifts** between splits (case mix). Any deployment would need local recalibration.
8. **Single centre** and observational. Associations (rules, PCA) are not causal, and no sTBI-specific mechanism is claimed.

**For COSC 5610 grading:** concerns 1–3 are disclosed, labelled and quantified, which is what a reviewer should expect. None is a hidden error.
