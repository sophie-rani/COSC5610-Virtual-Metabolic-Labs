# Protocol Amendment v1.1: post hoc methodological revision
**Status:** written and SHA-256-hashed on 2026-11-19, **before** any corrected analysis was run.

**What had been seen at the time of writing:**
- The frozen v1.0 results, including the single test evaluation and its reproduction.
- Dev/val **counts only** for the new pairing rules: no model performance.
- Lab turnaround times and `d_items` labels.

**What this amendment is:**
- A **post hoc correction** of methodological problems found in the v1.0 reproducibility audit.
- **Not** a new independent test. The test set has already been evaluated with v1.0. Corrected test results are reported as *revised estimates on the same held-out patients*.

**What it does not change:**
- the research question;
- cohort, split, references, outcomes and thresholds (iCa < 1.12 mmol/L; lactate > 2 mmol/L);
- predictor lists, baselines, model classes and hyperparameters;
- the prespecified primary models (Ridge for iCa, GBM for lactate).

No threshold, model or variant will be selected on test performance. All variants below are reported, whatever they show.

The frozen v1.0 code, results and audit trail are kept unchanged (`src/frozen/`, `original_v1.0/`).

## 1. Corrections
| # | Problem in v1.0 | Type | v1.1 correction |
|---|---|---|---|
| C1 | Phosphate, Mg, CBC (±6 h) and albumin (±24 h) were paired to the *nearest* value, so 8–35% could postdate the blood gas. Chemistry (±60 min) could too, in 9–13% of pairs | Temporal leakage (design bug for a real-time claim) | New pairing rules, defined in §2 |
| C2 | Ridge alpha chosen by RidgeCV leave-one-row-out, so the same patient's repeated measures could fall on both sides of an internal split | Bug (repeated-measures leakage in tuning) | Alpha chosen by `GroupKFold(5)` on patient id. Same grid (10⁻³–10³, 25 values) and same criterion (MSE). Other models have fixed hyperparameters and no internal validation; this is verified |
| C3 | `rrt6` looked up RRT itemids in `inputevents`, so it was always 0 | Bug (not a predictor; the exclusion was unaffected) | Look up `procedureevents`. Verify the active-RRT exclusion independently against CRRT/citrate charting (§3) |
| C4 | Bootstrap B = 300 (dev/val) and 1,000 (test subgroups/sensitivity) vs 2,000 in the protocol | Reporting/protocol deviation | All v1.1 analyses use **B = 2,000** patient-cluster resamples. v1.0 counts are documented as they were run |
| C5 | Lactate estimates biased low (test −0.51 mmol/L) | Expected property, needs explanation | Investigate only (§4). No recalibration, no use of test data, primary model unchanged |
| C6 | Sequential-rule columns mislabelled | Reporting bug | Relabel. Values unchanged |
| C7 | Units and missingness handling not reported | QA | Report units from `labevents.valueuom` and missingness per predictor |

## 2. Prediction time and availability (C1)
**Prediction time t₀.** The collection time (`charttime`) of the reference blood gas: the moment a clinician decides whether a blood gas is needed. The reference value itself is never available at t₀.

**Variants, all pre-declared and all reported:**
| Variant | Chemistry panel | Phosphate, Mg, CBC | Albumin | Purpose |
|---|---|---|---|---|
| **V0 original** (frozen) | nearest collection within ±60 min | nearest within ±6 h | nearest within ±24 h | Frozen result; preserved |
| **V1 ML fixes only** | as V0 | as V0 | as V0 | Isolates the effect of grouped Ridge tuning |
| **V2 collection-time causal** | latest *collected* in [t₀ − 60 min, t₀] | latest collected in [t₀ − 6 h, t₀] | latest collected in [t₀ − 24 h, t₀] | Retrospective estimation: nothing collected after t₀ |
| **V3 result-available (primary corrected)** | latest panel whose **every component was resulted** (`storetime`) at or before t₀, collected ≤ 6 h before t₀ | latest *resulted* by t₀, collected ≤ 6 h before | latest *resulted* by t₀, collected ≤ 24 h before | Clinically realistic: only results a clinician could see at t₀ |

**Rationale for V3:**
- `storetime` is the time a result became available in the laboratory system (MIMIC-IV; Johnson 2023, *Sci Data* 10:1, doi:10.1038/s41597-022-01899-x).
- In this cohort it is never missing, never precedes collection, and has plausible turnaround times: blood gas median 3 min, CBC 39 min, chemistry 69 min, albumin 77 min. It is therefore treated as reliable.
- A 60-min lookback is incompatible with the 69-min median chemistry turnaround. V3 therefore uses the 6-h window the frozen design already applied to the other routine labs. The ICU chemistry interval is typically 6–12 h.
- The window was fixed from turnaround times and counts, not from performance.
- V3 drops references with no resulted chemistry in the prior 6 h, so its pairs are a different, smaller set. Comparisons with V0 are side by side, not paired.

**Treatments:**
- Exposure = an infusion running at any time in [t₀ − 6 h, t₀] with start ≤ t₀, as in v1.0. Nothing started after t₀ is used.
- Assumption: a clinician knows a transfusion, calcium dose or vasopressor that is *already running* even if the nurse charts it later. Charting lag (`inputevents.storetime − starttime`) is reported, not used.
- This is an assumption, not a bug.

**Other rules:**
- Derived variables (AG, ca_mmol) come from the same available panel.
- Missing ancillary labs are imputed with training medians plus indicators, as frozen.

## 3. RRT (C3)
- `rrt_active` (exclusion) and `rrt6` (6-h exposure) are both taken from `procedureevents`, with the frozen itemids: 225802 CRRT, 225803 CVVHD, 225805 peritoneal, 225809 CVVHDF, 225955 SCUF.
- **Independent check:** CRRT/citrate evidence within ±2 h of each reference:
  - chartevents 227290 (CRRT mode) or 228004 (citrate ACD-A);
  - inputevents 227525 (calcium gluconate CRRT), 227526/227528/227529 (citrate), 230044 (heparin CRRT);
  - procedureevents 225436 (filter change) or 225441 (hemodialysis).
- Concordance with `rrt_active` is reported.
- **Pre-declared sensitivity analysis:** additionally exclude retained pairs with any independent evidence.

## 4. Lactate bias (C5)
- GBM is fit on log-lactate and back-transformed with exp(). This estimates the conditional **median** (geometric mean), not the mean, so mean bias is negative by construction.
- Report on dev (OOF), val and test:
  - mean bias, median bias and log-scale bias;
  - the Duan smearing factor estimated **on training out-of-fold residuals only** (Duan 1983, *JASA* 78:605, doi:10.1080/01621459.1983.10478017), with the mean bias that a smeared estimate *would* have.
- Smearing is explanatory only. It does not replace the primary estimate or its threshold classification.

## 5. Analyses and reporting
- **V3** (primary corrected):
  - dev patient-grouped 5-fold CV and val, all models, B = 2,000;
  - test: all models, primary contrasts, key secondary (vs corrected Ca 0.8), corrected-Ca vs total Ca, bedside rules, the frozen subgroups, severe lactate, AG < 16;
  - sensitivity analyses: no missingness indicators; chemistry only; basic metabolic panel only; one pair per patient-day; IPW for who gets tested; additional exclusion for independent CRRT/citrate evidence.

  All at B = 2,000.
- **V1 and V2:** the test primary contrasts (ΔAUROC, MAE reduction), B = 2,000.
- **V0:** frozen values, shown side by side.
- **PCA and association rules:** descriptive and not prediction-time dependent. Re-run on the frozen dev/val pairs with corrected labels.
- **Interpretation rule:** a main result "still holds" in v1.1 if the V3 test ΔAUROC 95% CI excludes 0 for the prespecified primary contrast. Otherwise it is reported as not confirmed under realistic availability.
