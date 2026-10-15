# MetaForge: Machine Learning-Accelerated High Entropy Alloy Discovery

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-metaforge--web.onrender.com-00c853?style=for-the-badge&logo=render)](https://metaforge-web.onrender.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Framework: ICME](https://img.shields.io/badge/ICME-Multi--Fidelity-teal.svg)](#hierarchical-multi-fidelity-discovery-framework)
[![Materials Informatics](https://img.shields.io/badge/Materials%20Informatics-CHGNet%20%7C%20Pymatgen-purple.svg)](https://github.com/SA-FIND/High-Entropy-Alloy-Discovery)

## Overview

**MetaForge** is a computational materials science pipeline designed to discover optimal High Entropy Alloys (HEAs) for aerospace, corrosion-resistance, lightweight structural, and high-temperature applications. It combines physics-informed machine learning with genetic algorithms for inverse design and graph neural network structural relaxation via CHGNet.

<div align="center">
  <a href="https://metaforge-web.onrender.com/" target="_blank" rel="noopener noreferrer">
    <img src="docs/assets/demo_screenshot.png" alt="MetaForge Interactive 3D Web Interface" width="100%">
  </a>
  <p><em>👉 <b><a href="https://metaforge-web.onrender.com/" target="_blank" rel="noopener noreferrer">Click here to launch the interactive live 3D crystal supercell & ML predictor</a></b></em></p>
</div>

---

## Core Features

- **Combinatorial Engine:** Generates candidate alloy compositions and filters them using established physical metallurgy rules (e.g., lattice strain limits δ < 6.6% and specific VEC phase thresholds).
- **Property Prediction:** Utilizes Random Forest regression models trained on 132 Matminer Magpie descriptors to predict density; yield strength is predicted by a model trained on measured room-temperature compression data (MPEA dataset). The older analytical strength proxy ($\sigma_y = \sigma_0 + M \cdot \tau_{ss}$, ad hoc constants) is kept only as a legacy baseline.
- **Inverse Design:** A custom genetic algorithm evolves alloy compositions over multiple generations to maximize specific strength (strength-to-weight ratio).
- **Structure Relaxation:** Optimizes SQS supercell blueprints (54-atom BCC / 48-atom FCC) using the CHGNet graph neural network interatomic potential, with post-relaxation space group verification via `SpacegroupAnalyzer`.
- **Web Interface:** A lightweight Flask and React interface with a clean, dark-themed UI for real-time model inference.

---

## Hierarchical Multi-Fidelity Discovery Framework

MetaForge employs a **three-tier multi-fidelity screening architecture** designed to balance high-throughput screening with atomistic checks. Tier 3 is a protocol, not yet executed:

```mermaid
graph TD
    subgraph T1["<b>TIER 1: High-Throughput Compositional Screen</b>"]
        A1["• Combinatorial phase filtering: Zhang δ ≤ 6.6%, Guo VEC thresholds"]
        A2["• Miedema mixing enthalpy: -15 ≤ ΔH_mix ≤ +5 kJ/mol, Ω ≥ 1.1"]
        A3["• 132 Magpie feature descriptors + Random Forest surrogate regressors"]
        A4["• Yield strength from measured-data model (MPEA); legacy analytical proxy as baseline"]
    end

    subgraph T2["<b>TIER 2: Atomistic GNN Relaxation & SRO Optimization</b>"]
        B1["• Monte Carlo simulated annealing minimizing Warren-Cowley SRO: α₁, α₂"]
        B2["• 54-atom BCC / 48-atom FCC Special Quasirandom Structures (SQS)"]
        B3["• CHGNet universal neural network potential structural relaxation"]
        B4["• Thermodynamic Formation Energy: ΔE_f relative to pure elements"]
    end

    subgraph T3["<b>TIER 3 (planned, not yet run): DFT & Experimental Validation</b>"]
        C1["• Proposed: DFT (VASP/QE) elastic constants for selected candidates"]
        C2["• Proposed: arc melting, XRD/SEM, hardness (via collaboration)"]
    end

    T1 -->|Top GA candidates| T2
    T2 -->|Optimized Blueprints| T3
```

> **Methodological Note on Surrogates:** In Tier 1, property models operate on analytical proxies (composition-weighted density and Taylor-factor solid-solution yield strength) to evaluate hundreds of thousands of combinations in milliseconds. The LinearRegression baseline confirms that these proxies are linear in Magpie features (R² ≈ 0.99), so a model trained on them only reproduces the formula. Yield strength is therefore now predicted by a model trained on **measured** data (see *Measured-Data Strength Model* below). Tier 2 CHGNet relaxations are benchmarked against measured lattice parameters (see *Structural Validation*).

---

## Status (October 2026)

MetaForge is being re-scoped to **refractory BCC alloys** (Mo-Nb-Ta-V-W family) with every model checked against measured data. The plan is in [`docs/PUBLICATION_PLAN.md`](docs/PUBLICATION_PLAN.md).

The candidate alloys previously listed here (four families) came from an analytical strength proxy and a genetic algorithm that did not enforce the 5-35 at.% bounds. They are withdrawn and kept only in [`archive/`](archive/). The web app still offers all four families, with a coverage warning where measured data is sparse.

---

## Structural Validation (CHGNet GNN Relaxations)

> **Correction (October 2026).** Earlier versions of this README and the accompanying manuscript reported CHGNet density gaps of up to +5.1% ("non-Vegard contraction"). Those numbers were not produced by a relaxation: the `*_Relaxed.cif` files were byte-identical copies of the unrelaxed blueprints, written by a fallback path when CHGNet was not installed. The files and the claim are withdrawn; `run_master.py` now writes a relaxed CIF only after a converged relaxation.

To check whether the relaxation step is trustworthy, `scripts/benchmark_chgnet.py` relaxes SQS supercells (54-atom BCC, 48-atom FCC; atoms **and** cell; fmax = 0.05 eV/Å; 3 SQS seeds each) of alloys with measured lattice parameters. Results: `results/chgnet_benchmark.csv`; structures: `data/structures/benchmark/`.

| Alloy | a, CHGNet (Å) | a vs experiment | Density vs experiment | Volume vs rule of mixtures |
| :--- | :---: | :---: | :---: | :---: |
| NbMoTaW (BCC) | 3.240 | +0.83% | −0.52% | +0.95% |
| NbMoTaWV (BCC) | 3.196 | +0.41% | −0.08% | +0.52% |
| CoCrFeNi (FCC) | 3.518 | −1.47% | n/a | −5.11% |
| CoCrFeMnNi (FCC) | 3.515 | −2.36% | n/a | −6.12% |

Seed-to-seed scatter in *a* is ≤ 0.004 Å. Refractory references: Senkov et al., *Intermetallics* 18 (2010) 1758; FCC values are typical literature XRD values.

### Key Findings
- **Refractory MPEAs show no volume contraction.** CHGNet matches measured lattice parameters within 1% and density within 0.6%; the relaxed volume is slightly *above* the rule of mixtures.
- **The 5–6% shrinkage in 3d FCC alloys is a model artefact, not physics.** CHGNet under-predicts the measured lattice parameter, and its predicted Fe/Co/Mn magnetic moments fall during relaxation, consistent with a magnetovolume error of the potential.
- **The rule-of-mixtures density proxy is sound:** against 47 measured alloys in the MPEA dataset its median error is 0.95%.

## Measured-Data Strength Model

`scripts/train_measured_strength.py` trains yield strength on measured data from the open MPEA dataset (Borg et al., *Scientific Data* 7 (2020) 430): 450 room-temperature compression tests on 314 distinct alloys, featurized with the same 132 Magpie descriptors the web app uses. Validation is 5-fold `GroupKFold` by composition, so no alloy appears in both training and test folds.

| Model | CV MAE (GPa) | CV R² |
| :--- | :---: | :---: |
| Predict the mean (baseline) | 0.412 | 0.00 |
| Ridge | 0.323 | 0.29 |
| Random Forest | 0.260 | 0.54 |
| **HistGradientBoosting (selected)** | **0.258** | **0.56** |

Example: NbMoTaW is predicted at 1.09 GPa; Senkov et al. (2011) measured about 1.06 GPa in compression.

**Coverage limits.** Few measured alloys contain Li (9), Mg (14), Zn (12) or Sc (2). For the Lightweight category the per-category CV R² is negative, so its strength predictions are extrapolations; the web app flags any composition with ≥ 5 at.% of an element present in fewer than 20 measured alloys.

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
*Alternatively, you can open and run `notebooks/HEA_Exploration.ipynb` for an interactive, cell-by-cell walkthrough of the data harvesting, filtering, and model training process.*

### 4. Local Web Server

To launch the local web server and interact with the trained models:

```bash
cd MetaForge-Web
..\.venv\Scripts\python.exe app.py
```

---

## Repository Architecture

```text
MetaForge/
├── data/
│   └── structures/              # SQS blueprints & CHGNet-relaxed CIF crystal supercells
├── docs/
│   └── assets/                  # Documentation figures & interface previews
├── notebooks/
│   └── HEA_Exploration.ipynb    # Interactive combinatorial & ML prototyping notebook
├── MetaForge-Web/               # Production Flask + React Web Application
│   ├── models/                  # Pretrained Random Forest regressor weights (.model)
│   ├── templates/               # Responsive single-page interface with 3Dmol.js viewer
│   └── tests/                   # Automated API & security integration test suite
├── run_master.py                # Standalone master CLI pipeline (training -> GA -> SQS)
├── requirements.txt             # Primary environment dependencies
├── render.yaml                  # Cloud deployment configuration
├── README.md                    # Project documentation
├── docs/PUBLICATION_PLAN.md     # Current plan (refractory scope, validation)
├── archive/                     # Retired material (old proposal, withdrawn candidates)
└── LICENSE                      # MIT Open-Source License
```

---

## Deployment

This repository includes a `render.yaml` Blueprint for direct deployment on Render. The static frontend communicates with a Gunicorn-wrapped Flask backend to serve the machine learning models.

---

## Citation

If you use MetaForge in your research, academic proposals, or materials discovery workflows, please cite:

```bibtex
@software{ahedor2026metaforge,
  author       = {Ahedor, Solomon},
  title        = {MetaForge: Machine Learning-Accelerated High-Entropy Alloy Discovery via Multi-Fidelity ICME and Neural Interatomic Potentials},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/SA-FIND/High-Entropy-Alloy-Discovery}}
}
```

---

## Foundational References & Literature Base

1. **Empirical HEA Phase Prediction & Descriptors:**
   - Zhang, Y., Zuo, T. T., Tang, Z., et al. *Microstructures and properties of high-entropy alloys.* **Prog. Mater. Sci.** 61 (2014) 1–93.
   - Guo, S., & Liu, C. T. *Phase stability in high entropy alloys: Formation of solid-solution phase or amorphous phase.* **Prog. Nat. Sci.: Mater. Int.** 21 (2011) 433–446.
   - Fieser, D., Dewanjee, U., & Hu, A. *HEA-Bench: An AI-Agent-Optimized Calculator of High-Entropy Alloy and Oxide Descriptors and Phase-Prediction Rules.* **Materials** 19(14) (2026) 3075. [DOI: 10.3390/ma19143075](https://doi.org/10.3390/ma19143075).
2. **Dislocation Mechanics & Solid-Solution Strengthening:**
   - Varvenne, C., Luque, A., & Curtin, W. A. *Theory of strengthening in fcc high entropy alloys.* **Acta Materialia** 118 (2016) 164–176.
   - Maresca, F., & Curtin, W. A. *Mechanistic origin of high strength in refractory BCC high entropy alloys up to 1900K.* **Acta Materialia** 182 (2020) 235–249. [DOI: 10.1016/j.actamat.2019.10.015](https://doi.org/10.1016/j.actamat.2019.10.015).
   - Toda-Caraballo, I., & Rivera-Díaz-del-Castillo, P. E. J. *Modelling solid solution hardening in high entropy alloys.* **Acta Materialia** 85 (2015) 14–23.
3. **Data:**
   - Borg, C. K. H., et al. *Expanded dataset of mechanical properties and observed phases of multi-principal element alloys.* **Scientific Data** 7 (2020) 430. [DOI: 10.1038/s41597-020-00768-9](https://doi.org/10.1038/s41597-020-00768-9).
4. **Graph Neural Network Interatomic Potentials:**
   - Deng, B., Zhong, P., Jun, K., et al. *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.* **Nature Machine Intelligence** 5 (2023) 1031–1041.
5. **Materials Informatics Core:**
   - Jain, A., Ong, S. P., Hautier, G., et al. *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.* **APL Materials** 1 (2013) 011002.
   - Ward, L., Dunn, A., Faghaninia, A., et al. *Matminer: An open source toolkit for materials data mining.* **Comput. Mater. Sci.** 152 (2018) 60–69.
   - Ong, S. P., Richards, W. D., Jain, A., et al. *Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis.* **Comput. Mater. Sci.** 68 (2013) 314–319.

---

## License

This project is open-source under the **MIT License**.
