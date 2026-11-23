# Virtual Metabolic Laboratories in Severe Traumatic Brain Injury

COSC 5610 — Data Mining and Machine Learning (Fall 2026)  
Sophie Persaud

## Project

This project estimates ionized calcium and lactate from routinely available laboratory measurements in adults with severe traumatic brain injury, using MIMIC-IV v3.1. It compares regression and classification models with conventional laboratory surrogates and evaluates held-out performance using patient-separated splits.

The main analysis is in [`analysis.ipynb`](analysis.ipynb). The notebook contains cohort extraction, temporal laboratory matching, exploratory analysis, preprocessing, model comparisons, evaluation, and plots. Supporting functions are under `src/`.

## Results

The primary analysis uses laboratory results available at prediction time. On the held-out cohort, ionized calcium classification (threshold 1.12 mmol/L) had AUROC 0.763 for Ridge regression versus 0.705 for the calibrated total-calcium comparator. Lactate classification (threshold 2 mmol/L) had AUROC 0.850 for gradient boosting versus 0.754 for the calibrated anion-gap/bicarbonate comparator. Results are internal and exploratory; they do not establish clinical utility or independent external validation.

Summary outputs are in `results/`; figures are in `figures/`.

## Reproducing the analysis

The raw MIMIC-IV data are restricted and **not included**. Obtain credentialed access to MIMIC-IV v3.1 through PhysioNet, download the required `hosp` and `icu` tables, and place the `.csv.gz` files in a single folder.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
export MIMIC_RAW=/absolute/path/to/mimic-iv-files
jupyter notebook analysis.ipynb
```

Run cells in order from the repository root. The notebook writes intermediate patient-level files under `data/`; do not commit that directory. The original analysis took approximately 40–60 minutes on two CPU cores. Dependencies are pinned in `requirements.txt`.

## Repository structure

- `analysis.ipynb` — full executed analysis and figures
- `src/` — supporting model, metrics, cohort, and notebook-generation code
- `results/` — aggregate output workbook and summary values
- `figures/` — exported visualizations
- `docs/` — protocol, methodological revision record, and reproducibility checks
- `frozen_reference/` — aggregate reference outputs used for the documented method comparison

## Interpretation and limitations

This is an observational, single-center analysis of patients with eligible laboratory measurements. Missingness and selective testing may affect generalizability. The notebook documents a methodological revision correcting time alignment and patient-grouped tuning; revised results use the same held-out patients, rather than an independent new test set. See `docs/QA_REPORT_v1.1.md` for details.

No patient-level records or credentialed source data are included in this repository.
