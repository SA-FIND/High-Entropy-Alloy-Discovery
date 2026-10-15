# Processed data

## rhea_strength.csv

Yield strengths of refractory multi-principal element alloys, built by `scripts/build_rhea_dataset.py`.

- **Source:** MPEA dataset, C. K. H. Borg et al., "Expanded dataset of mechanical properties and observed phases of multi-principal element alloys", *Scientific Data* 7 (2020) 430, doi:10.1038/s41597-020-00768-9. Repository: github.com/CitrineInformatics/MPEA_dataset, Apache License 2.0. This file is a filtered and reformatted derivative; the original references are kept in the `doi` column.
- **Scope:** alloys containing only Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W (plus Al, Re, flagged outside the physics domain) with a reported yield strength and test temperature.
- **Columns:** `key` (alloy identity from fractions rounded to 0.001), atomic fractions per element, `T_C` (test temperature, C), `YS_MPa`, `test` (C compression, T tension), `microstructure`, `processing`, `single_phase_bcc`, `physics_domain` (single-phase BCC and only the nine elements), `year`, `holdout`, `doi`.
- **Holdout:** `holdout == "future"` marks alloys whose first appearance in the dataset is 2019 (the latest year). They are never used for training or model selection and are scored once at the end, to test whether models trained on older data would have predicted newer alloys.

| Subset | Rows | Alloys | Room temperature | >= 600 C |
|---|---:|---:|---:|---:|
| All refractory (incl. Al, Re) | 492 | 135 | 250 | 210 |
| Physics domain (single-phase BCC) | 277 | 72 | 149 | 104 |
| ... train | 217 | 58 | 131 | 78 |
| ... future holdout | 60 | 14 | 18 | 26 |
