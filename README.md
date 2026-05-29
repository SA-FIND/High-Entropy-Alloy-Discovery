<div align="center">

# MetaForge
**ML-Powered High Entropy Alloy Discovery**

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask 3.1.3](https://img.shields.io/badge/Flask-3.1.3-black?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![React 18](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react&logoColor=black)](https://reactjs.org/)
[![Materials Project](https://img.shields.io/badge/Materials%20Project-API-00c853?style=flat-square)](https://materialsproject.org/)
[![CHGNet](https://img.shields.io/badge/CHGNet-GNN-ff6f00?style=flat-square)](https://github.com/CederGroupHub/chgnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

An end-to-end computational materials science pipeline for discovering optimal High Entropy Alloys (HEAs) for diverse applications including aerospace, corrosion resistance, refractory, and lightweight structural purposes. Combines data harvesting from the Materials Project, physics-informed machine learning, genetic algorithm-driven inverse design and ML interatomic potential relaxation.

</div>

---

## Core Features

> **Combinatorial Engine**<br>
> Generates thousands of theoretical alloy compositions and filters them using physics-based stability rules (lattice strain δ, VEC thresholds).

> **ML Property Prediction**<br>
> Trains RandomForest models on 132 Matminer Magpie descriptors to predict density and shear strength from composition.

> **Genetic Algorithm Inverse Design**<br>
> Evolves alloy compositions over 20 generations to maximize specific strength (strength-to-weight ratio).

> **CHGNet Structure Relaxation**<br>
> Relaxes 54-atom supercell blueprints using a graph neural network interatomic potential.

> **Real-Time Web Interface**<br>
> Flask app with interactive composition sliders, live ML predictions, and a composition donut chart.

---

## Architecture & Data Flow

```mermaid
flowchart TD
    A[Materials Project API] -->|Atomic radii, density, VEC| B(Combinatorial Engine)
    B -->|δ < 6.6 + VEC rules| C(Feature Engineering)
    C -->|132 Magpie descriptors| D(ML Model Training)
    D -->|Density + Strength Predictors| E(Genetic Algorithm)
    E -->|Maximize specific strength| F(Crystal Blueprint)
    F -->|54-atom BCC supercell| G(Structure Relaxation)
    G -->|CHGNet FIRE optimizer| H[Optimized Alloy CIF]
    
    D -.->|Export models| I(Flask Web Backend)
    I <-->|POST /predict| J[React Web Interface]
```

---

## Tech Stack

**Core Engineering**<br>
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pymatgen](https://img.shields.io/badge/Pymatgen-2026.5.4-blue?style=flat-square)
![Matminer](https://img.shields.io/badge/Matminer-0.10.1-blue?style=flat-square)

**Machine Learning**<br>
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![CHGNet](https://img.shields.io/badge/CHGNet-0.4.1-F7931E?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

**Web Application**<br>
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white)

---

## Quickstart

<details open>
<summary><b>1. Environment Setup</b></summary>

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MetaForge.git
cd MetaForge

# Create and activate virtual environment (Windows PowerShell)
python -m venv ----
.\----\Scripts\Activate.ps1

# Install core dependencies
pip install pymatgen mp-api python-dotenv numpy pandas scikit-learn matplotlib matminer chgnet flask flask-cors joblib
```
</details>

<details>
<summary><b>2. Configure API Keys</b></summary>

Create a `.env` file in the root directory and add your Materials Project API key:

```bash
# .env
MY_API_KEY=your_materials_project_api_key_here
```
*(Only required for data harvesting in Jupyter notebooks)*
</details>

<details>
<summary><b>3. Launch Web Application</b></summary>

Navigate to the web directory and start the Flask server using the virtual environment:

```bash
cd MetaForge-Web
..\----\Scripts\python.exe app.py
```
</details>

---

## Physics & Stability Filters

| HEA Family | Crystal Structure | Lattice Strain (δ) | VEC Range |
|------------|-------------------|---------------------|-----------|
| **Refractory** | BCC | δ < 6.6% | 5.0 ≤ VEC ≤ 6.8 |
| **Corrosion-Resistant** | FCC | δ < 6.6% | VEC ≥ 8.0 |
| **Lightweight** | Mixed/HCP | δ < 6.6% | — |

---

## License & Attribution

This project is open-source under the MIT License.

Special thanks to the foundational work by:
- **Materials Project**, Jain et al., APL Materials, 2013
- **Matminer**, Ward et al., Comput. Mater. Sci., 2018
- **CHGNet**, Deng et al., Nature Machine Intelligence, 2023
- **Pymatgen**, Ong et al., Comput. Mater. Sci., 2013
