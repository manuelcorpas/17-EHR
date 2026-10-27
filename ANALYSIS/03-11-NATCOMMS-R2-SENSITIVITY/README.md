# NCOMMS-26-040266-T round 2 sensitivity analyses

Purpose: answer the 23 September 2026 decision. R4.1 asked for a complete-case (86 diseases) analysis, R4.2 for rankings without the +10 Global South penalty, and R3.5 for specificity (does HEIM fail to flag diseases that are not neglected?).
Created: 25 October 2026. Inputs: frozen `docs/data/integrated.json`, `diseases.json` and `clinical_trials.json` (read only).
Regenerate: `python3 01-r2-sensitivity.py`. Outputs: `02-r2-sensitivity-scores.csv` and `03-r2-sensitivity-report.txt`.
Controls: the unified score reproduces the stored value to 1e-14 and the gap function reproduces the stored gap for 169/172 matched diseases. The three exceptions are handled explicitly in the script (Ebola and Zika are manual overrides; Leprosy has DALYs rounded in the JSON).
