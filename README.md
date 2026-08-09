# MetaForge: Machine Learning-Accelerated High Entropy Alloy Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

## Overview

**MetaForge** is a computational materials science pipeline designed to discover optimal High Entropy Alloys (HEAs) for aerospace, corrosion-resistance, lightweight structural, and high-temperature applications. It combines physics-informed machine learning with genetic algorithms for inverse design and graph neural network structural relaxation via CHGNet.

![MetaForge Web Interface Preview](demo_screenshot.png)

---

## Core Features

- **Combinatorial Engine:** Generates candidate alloy compositions and filters them using established physical metallurgy rules (e.g., lattice strain limits δ < 6.6% and specific VEC phase thresholds).
- **Property Prediction:** Utilizes Random Forest regression models trained on 132 Matminer Magpie descriptors to predict rule-of-mixtures density and solid-solution strengthening proxies (see [Transparency Note](#transparency-note) below).
- **Inverse Design:** A custom genetic algorithm evolves alloy compositions over multiple generations to maximize specific strength (strength-to-weight ratio).
- **Structure Relaxation:** Optimizes SQS supercell blueprints (54-atom BCC / 48-atom FCC) using the CHGNet graph neural network interatomic potential, with post-relaxation space group verification via `SpacegroupAnalyzer`.
- **Web Interface:** A lightweight Flask and React interface with a clean, dark-themed UI for real-time model inference.

---

## Transparency Note

> **Important:** The density and strength targets used to train the surrogate models are **rule-of-mixtures analytical proxies**, not experimental measurements or DFT-derived values.
>
> - **Density** is a composition-weighted average of elemental densities.
> - **Strength** uses a simplified Varvenne-Luque-Curtin (2016) solid-solution strengthening estimate based on the rule-of-mixtures shear modulus plus an atomic size misfit correction.
>
> Because Magpie descriptors include statistics of the same atomic properties used to compute these proxies, the ML models achieve near-perfect R² scores by construction. A LinearRegression sanity check is included in the pipeline output to confirm this, if LR achieves comparable R² to the Random Forest, the target is analytically recoverable from the features.
>
> **Replacing these proxies with grounded data** (Materials Project elastic tensors, experimental measurements, or DFT-computed properties) is the planned next step. See `RESEARCH_PROPOSAL.md` for the validation roadmap.

---

## Model Performance & Discovered Candidates

The pipeline currently supports four distinct HEA categories. The metrics below reflect model fit to **rule-of-mixtures proxy targets** (see [Transparency Note](#transparency-note)), validated with 5-fold cross-validation.

### 1. Refractory Alloys (BCC, 54-atom supercell)
- **Density Model:** RMSE 0.046 g/cm³ | R² 0.998 via 5-fold cross-validation
- **Strength Model:** RMSE 0.740 GPa | R² 0.995 via 5-fold cross-validation
- **Top Candidate:** `W:9.4% - Mo:70.2% - Ta:1.6% - Nb:7.6% - V:11.3%`
- **Specific Strength:** 11.35 GPa/(g/cm³)

### 2. Corrosion-Resistant Alloys (FCC, 48-atom supercell)
- **Density Model:** RMSE 0.013 g/cm³ | R² 0.993 via 5-fold cross-validation
- **Strength Model:** RMSE 0.153 GPa | R² 0.999 via 5-fold cross-validation
- **Top Candidate:** `Co:5.3% - Cr:40.1% - Fe:41.3% - Ni:1.8% - Cu:11.6%`
- **Specific Strength:** 11.40 GPa/(g/cm³)

### 3. Lightweight Alloys (FCC, 48-atom supercell)
- **Density Model:** RMSE 0.023 g/cm³ | R² 0.998 via 5-fold cross-validation
- **Strength Model:** RMSE 0.161 GPa | R² 0.998 via 5-fold cross-validation
- **Top Candidate:** `Al:36.4% - Mg:12.7% - Li:16.2% - Ti:33.4% - Zn:1.2%`
- **Specific Strength:** 9.84 GPa/(g/cm³)

### 4. Aerospace Alloys (FCC, 48-atom supercell)
- **Density Model:** RMSE 0.043 g/cm³ | R² 0.982 via 5-fold cross-validation
- **Strength Model:** RMSE 0.216 GPa | R² 0.982 via 5-fold cross-validation
- **Top Candidate:** `Al:29.9% - Ti:35.1% - Sc:31.1% - Zr:0.7% - V:3.2%`
- **Specific Strength:** 9.35 GPa/(g/cm³)

---

## Structural Validation (CHGNet GNN Relaxations)

To evaluate the physical validity of the Rule-of-Mixtures (RoM) density proxy, we generated Special Quasirandom Structures (SQS) (54-atom BCC for Refractory, 48-atom FCC for others) for the top candidates and relaxed them using the **CHGNet** graph neural network potential. 

To ensure the results represent true equilibrium structures and are not relaxation artifacts, we verified force convergence (< 0.1 eV/Å target, achieved in all runs within 35 steps) and evaluated statistical variance across multiple random SQS atomic shuffles (different random seeds).

| Category | RoM Density (g/cm³) | CHGNet Density (g/cm³) | Gap (%) | Seed Std Dev (g/cm³) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Refractory** | 9.40 | 9.88 (n=3) | +5.1% | ±0.026 (0.3%) | Real, strong margin |
| **Corrosion** | 8.00 | 8.27 (n=3) | +3.4% | ±0.033 (0.4%) | Real, solid margin |
| **Lightweight** | 2.62 | 2.69 (n=8) | +2.8% | ±0.011 (0.4%) | Real, confirmed after widening |
| **Aerospace** | 3.67 | 3.62 (n=3) | -1.4% | ±0.003 (0.1%) | Real, small but clean effect |

### Key Findings:
- **Lattice Relaxation Effects:** The GNN relaxations reveal a clear physical density gap from simple linear rule-of-mixtures predictions, ranging from a +5.1% contraction in Refractory to a -1.4% expansion in Aerospace.
- **Statistical Significance:** The seed-to-seed standard deviation (representing single-SQS realization noise) is consistently $\le 0.4\%$, leaving a comfortable margin that confirms the density gaps represent real physical bonding trends rather than random shuffle noise. For the Lightweight category, the seed count was widened to $n=8$ to tighten the standard error and confirm the significance of the 2.8% gap.

---


## Architecture & Data Flow

1. **Data Sources & Features:** The target variables (density and strength) are rule-of-mixtures analytical proxies, not Materials Project DFT values or a compiled experimental dataset. Elemental properties used to enforce metallurgical constraints (atomic radii, density, VEC) were sourced from the Materials Project API in the interactive notebook (`HEA.ipynb`). The features (132 Magpie descriptors) are computed by Matminer using its own internal elemental property tables, not Materials Project API results. Both features and targets are derived entirely from composition.
2. **Feature Engineering:** Compositions are filtered by phase rules (Zhang's δ parameter, Guo's VEC thresholds) and featurized into 132 Magpie descriptors via Matminer.
3. **Training:** Scikit-learn Random Forests map descriptors to density and strength proxies, validated with 5-fold cross-validation and a LinearRegression sanity check.
4. **Optimization:** Genetic Algorithm maximizes specific strength across the composition simplex.
5. **Blueprint Generation:** BCC (2-atom basis, 54-atom supercell) or FCC (4-atom basis, 48-atom supercell) structures are generated per category and relaxed with CHGNet. Post-relaxation space group is verified with `SpacegroupAnalyzer`.

---

## Project Structure

```text
├── HEA.ipynb            # Interactive notebook with Materials Project API harvesting
├── run_master.py        # Master script: full end-to-end pipeline
├── MetaForge-Web/       # Frontend UI and Flask backend for serving models
├── render.yaml          # Blueprint for direct deployment on Render
└── *.cif                # Generated and relaxed SQS blueprints
```

---

## Tech Stack

- **Core Engineering:** Python 3.13+, Pymatgen, Matminer
- **Machine Learning:** Scikit-learn, CHGNet, NumPy, Pandas, Joblib
- **Web Application:** Flask, React, Vanilla CSS

---

## Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/SA-FIND/High-Entropy-Alloy-Discovery.git
cd High-Entropy-Alloy-Discovery

# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install pymatgen mp-api python-dotenv numpy pandas scikit-learn matplotlib matminer chgnet flask flask-cors joblib
```

### 2. Configure API Keys

Create a `.env` file in the root directory and add your Materials Project API key. This is required for the data harvesting step in `HEA.ipynb`.

```bash
MY_API_KEY=your_materials_project_api_key_here
```

### 3. Execution

You can run the full machine learning pipeline sequentially by executing the master script:

```bash
python run_master.py
```
*Alternatively, you can open and run `HEA.ipynb` for an interactive, cell-by-cell walkthrough of the data harvesting, filtering, and model training process.*

### 4. Local Web Server

To launch the local web server and interact with the trained models:

```bash
cd MetaForge-Web
..\.venv\Scripts\python.exe app.py
```

---

## Deployment

This repository includes a `render.yaml` Blueprint for direct deployment on Render. The static frontend communicates with a Gunicorn-wrapped Flask backend to serve the machine learning models.

---

## License & Attribution

This project is open-source under the **MIT License**.

**Key Dependencies:**
- Materials Project *(Jain et al., APL Materials, 2013)*
- Matminer *(Ward et al., Comput. Mater. Sci., 2018)*
- CHGNet *(Deng et al., Nature Machine Intelligence, 2023)*
- Pymatgen *(Ong et al., Comput. Mater. Sci., 2013)*

**Strengthening Model:**
- Varvenne, Luque & Curtin, *Acta Materialia* 118 (2016) 164-176
