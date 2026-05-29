# MetaForge — AI-Powered High Entropy Alloy Discovery

MetaForge is an end-to-end computational materials science pipeline for discovering optimal High Entropy Alloys (HEAs) for aerospace applications. It combines data harvesting from the Materials Project, physics-informed machine learning (Matminer + RandomForest), genetic algorithm-driven inverse design, and ML interatomic potential relaxation (CHGNet) — all served through a real-time Flask web interface.

## Key Features

- **Data Harvesting** — Pulls atomic radii, densities, and structural data directly from the [Materials Project API](https://materialsproject.org/) for 15 elements across 3 HEA families
- **Combinatorial Engine** — Generates thousands of theoretical alloy compositions and filters them using physics-based stability rules (lattice strain δ, VEC thresholds)
- **ML Property Prediction** — Trains RandomForest models on 132 Matminer Magpie descriptors to predict density and shear strength from composition
- **Genetic Algorithm Inverse Design** — Evolves alloy compositions over 20 generations to maximize specific strength (strength-to-weight ratio)
- **CHGNet Structure Relaxation** — Relaxes 54-atom supercell blueprints using a graph neural network interatomic potential
- **Real-Time Web Interface** — Flask app with interactive composition sliders, live ML predictions, and a composition donut chart

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [The Discovery Pipeline](#the-discovery-pipeline)
- [Web Interface](#web-interface)
- [Environment Variables](#environment-variables)
- [Available Scripts](#available-scripts)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Language** | Python | 3.13+ |
| **Materials Database** | Materials Project API (`mp-api`) | 0.46.1 |
| **Crystal Structures** | Pymatgen | 2026.5.4 |
| **Feature Engineering** | Matminer (Magpie preset) | 0.10.1 |
| **ML Models** | scikit-learn (RandomForestRegressor) | 1.8.0 |
| **Structure Relaxation** | CHGNet (Graph Neural Network) | 0.4.1 |
| **Web Backend** | Flask + Flask-CORS | 3.1.3 |
| **Web Frontend** | React 18 (CDN) + Vanilla CSS | — |
| **Model Serialization** | joblib | 1.5.3 |
| **Numerical** | NumPy, Pandas | 2.4.5, 2.3.3 |

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.13+** — [Download from python.org](https://www.python.org/downloads/)
- **Git** — [Download from git-scm.com](https://git-scm.com/)
- **Materials Project API Key** — Free at [materialsproject.org](https://materialsproject.org/api). You'll need this to harvest element data from the Materials Project database.
- **~4 GB disk space** — For the virtual environment with all scientific packages

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MetaForge.git
cd MetaForge
```

### 2. Create and Activate the Virtual Environment

All dependencies must be installed inside the project's virtual environment (`pymatgenenv`), not in your global Python.

```bash
# Create the virtual environment
python -m venv pymatgenenv

# Activate it
# Windows (PowerShell):
.\pymatgenenv\Scripts\Activate.ps1

# Windows (Command Prompt):
.\pymatgenenv\Scripts\activate.bat

# macOS/Linux:
source pymatgenenv/bin/activate
```

> **Important:** You will see `(pymatgenenv)` in your terminal prompt when the environment is active. All `pip install` commands below must be run while this environment is active.

### 3. Install Dependencies

```bash
pip install pymatgen mp-api python-dotenv numpy pandas scikit-learn matplotlib
pip install matminer
pip install chgnet
pip install flask flask-cors joblib
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
MY_API_KEY=your_materials_project_api_key_here
```

Get your API key from [Materials Project Dashboard](https://materialsproject.org/api#api-key).

### 5. Run the Discovery Pipeline (Jupyter Notebooks)

Open the notebooks in VS Code or Jupyter and run them in order:

```bash
# Make sure pymatgenenv is active, then:
jupyter notebook
```

- **`HEA.ipynb`** — Original multi-family HEA discovery (Refractory, Corrosion, Lightweight)
- **`HEA2.0.ipynb`** — Refined lightweight alloy pipeline with Matminer features, genetic algorithm inverse design, and CHGNet relaxation

### 6. Launch the Web Interface

```bash
cd MetaForge-Web

# Option A: Use the launch script (Windows)
run.bat

# Option B: Run directly with the venv Python
..\pymatgenenv\Scripts\python.exe app.py
```

Open **http://localhost:5000** in your browser.

> ⚠️ **Do not** run `python app.py` directly — this uses your global Python which doesn't have the scientific packages installed. Always use the virtual environment's Python.

---

## Project Structure

```
MetaForge/
├── .env                                    # API keys (not committed to git)
├── HEA.ipynb                              # v1: Multi-family HEA discovery
├── HEA2.0.ipynb                           # v2: Matminer + Genetic Algorithm pipeline
├── Optimal_AlMgLiTiZn_Blueprint.cif       # Generated CIF from v1 (Lightweight)
├── Optimal_AlTiScZrV_Blueprint.cif        # Generated CIF from v2
├── Relaxed_Optimal_AlTiScZrV_Blueprint.cif # CHGNet-relaxed structure
├── pymatgenenv/                           # Python virtual environment (not committed)
└── MetaForge-Web/                         # Flask web application
    ├── app.py                             # Flask backend + ML prediction API
    ├── ml_density.model                   # Trained RandomForest (density predictor)
    ├── ml_strength.model                  # Trained RandomForest (strength predictor)
    ├── run.bat                            # Windows launch script
    └── templates/
        └── index.html                     # React frontend with glassmorphism UI
```

---

## Architecture

### Pipeline Data Flow

```
Materials Project API
        │
        ▼
┌─────────────────────┐
│   Data Harvesting    │  Atomic radii, density, VEC
│   (mp-api + pymatgen)│  for each element
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Combinatorial Engine │  itertools.product → 1451 compositions
│   (Stability Filter) │  δ < 6.6 + VEC rules
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Feature Engineering │  Composition → 132 Magpie descriptors
│     (Matminer)       │  (electronegativity, atomic radius, etc.)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   ML Model Training  │  RandomForest × 2
│   (scikit-learn)     │  Predicts density + shear strength
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Genetic Algorithm   │  50 population × 20 generations
│  (Inverse Design)    │  Maximizes specific strength
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Crystal Blueprint  │  54-atom BCC supercell
│    (pymatgen CIF)    │  Random site occupation
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Structure Relaxation│  CHGNet FIRE optimizer
│     (CHGNet)         │  fmax < 0.05 eV/Å convergence
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   Model Export       │  joblib.dump → .model files
│   (joblib)           │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Flask Web App       │  POST /predict → JSON response
│  (MetaForge-Web)     │  Composition sliders → real-time predictions
└─────────────────────┘
```

### Web Application Request Lifecycle

1. User adjusts element composition sliders in the browser
2. React debounces the input (300ms) and sends a `POST /predict` request
3. Flask normalizes slider values to sum to 100%
4. Composition is featurized into 132 Magpie descriptors via Matminer
5. Both RandomForest models predict density and shear strength
6. Specific strength score is calculated (strength ÷ density)
7. JSON response is rendered in the UI with animated transitions

### Stability Filters (Physics-Based)

| HEA Family | Crystal Structure | Lattice Strain (δ) | VEC Range |
|------------|-------------------|---------------------|-----------|
| Refractory (W, Mo, Ta, Nb, V) | BCC | δ < 6.6% | 5.0 ≤ VEC ≤ 6.8 |
| Corrosion-Resistant (Co, Cr, Fe, Ni, Cu) | FCC | δ < 6.6% | VEC ≥ 8.0 |
| Lightweight (Al, Ti, Sc, Zr, V) | Mixed/HCP | δ < 6.6% | — |

---

## The Discovery Pipeline

### Notebook 1: `HEA.ipynb` — Multi-Family Discovery

Explores 3 families of HEAs simultaneously:

| Step | What It Does |
|------|-------------|
| **Cell 1** | Import libraries and configure environment |
| **Cell 2** | Harvest element properties from Materials Project for 15 elements across Refractory, Corrosion, and Lightweight families |
| **Cell 3** | Run combinatorial engine (5%–35% fractions, step 5%) with physics stability filters per family |
| **Cell 4** | Train RandomForest per family, optimize for specific strength, generate CIF blueprint |

### Notebook 2: `HEA2.0.ipynb` — Advanced Lightweight Pipeline

Focused deep dive on the Al-Ti-Sc-Zr-V lightweight system:

| Step | What It Does |
|------|-------------|
| **Cell 1** | Import scientific stack |
| **Cell 2** | Harvest from Materials Project, combinatorial engine, RandomForest on raw fractions → **best alloy: Al₂₀Ti₃₅Sc₅Zr₅V₃₅** (Score: 8.14) |
| **Cell 3** | Build 54-atom BCC supercell blueprint, export `.cif` |
| **Cell 4** | Relax structure with CHGNet (32 FIRE steps, final energy: −403.77 eV) |
| **Cell 5** | Upgrade to Matminer Magpie features (132 descriptors), train new models, run Genetic Algorithm → **evolved alloy: Al₂₄Ti₂₉Sc₃₄Zr₁V₁₀** (Score: 8.99) |
| **Cell 6** | Export trained models as `ml_density.model` and `ml_strength.model` via joblib |

---

## Web Interface

### `MetaForge-Web/app.py` — Backend API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the React frontend |
| `/predict` | POST | Accepts element weights as JSON, returns predicted density, strength, score, and normalized composition |

**Example request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Al": 25, "Ti": 30, "Sc": 20, "Zr": 10, "V": 15}'
```

**Example response:**
```json
{
  "density": 4.12,
  "strength": 35.67,
  "score": 8.66,
  "composition": {"Al": 25.0, "Ti": 30.0, "Sc": 20.0, "Zr": 10.0, "V": 15.0}
}
```

### `MetaForge-Web/templates/index.html` — Frontend

- Dark glassmorphism design with animated gradient mesh background
- 5 element-colored composition sliders (Al=blue, Ti=violet, Sc=emerald, Zr=amber, V=rose)
- CSS conic-gradient donut chart showing normalized composition
- Real-time AI predictions with shimmer animation on update
- 300ms debounced API calls to prevent server hammering
- Responsive layout (2-column → 1-column on mobile)

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MY_API_KEY` | Materials Project API key for data harvesting | Yes (notebooks only) |
| `DEBUG` | Enable debug mode | No (default: `TRUE`) |

> **Note:** The web app (`MetaForge-Web/app.py`) does not require API keys — it uses pre-trained models. The API key is only needed when running the Jupyter notebooks to harvest fresh data from Materials Project.

---

## Available Scripts

| Command | Description |
|---------|-------------|
| `run.bat` | Start the Flask web server using the correct venv Python (Windows) |
| `.\pymatgenenv\Scripts\Activate.ps1` | Activate the virtual environment (PowerShell) |
| `..\pymatgenenv\Scripts\python.exe app.py` | Start Flask server from MetaForge-Web directory |
| `jupyter notebook` | Launch Jupyter to run the discovery pipeline notebooks |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'flask'` (or any package)

**Cause:** You're running with the global Python instead of the virtual environment.

**Solution:**
```bash
# Use the venv Python directly:
..\pymatgenenv\Scripts\python.exe app.py

# Or activate the environment first:
.\pymatgenenv\Scripts\Activate.ps1
python app.py
```

### `ModuleNotFoundError` inside Jupyter Notebook

**Cause:** The notebook kernel is not pointing to `pymatgenenv`.

**Solution:** Run inside a notebook cell:
```python
%pip install <package_name>
```

Or register the venv as a Jupyter kernel:
```bash
.\pymatgenenv\Scripts\Activate.ps1
pip install ipykernel
python -m ipykernel install --user --name=pymatgenenv --display-name="pymatgenenv"
```

### Materials Project API errors

**Cause:** Invalid or expired API key.

**Solution:**
1. Verify your key at [materialsproject.org/api](https://materialsproject.org/api)
2. Check `.env` file contains: `MY_API_KEY=your_key_here`
3. Ensure `python-dotenv` is installed: `pip install python-dotenv`

### CHGNet runs slowly

**Cause:** CHGNet defaults to CPU when no CUDA GPU is available.

**Solution:** This is expected on CPU. The 54-atom relaxation typically takes ~30 seconds on a modern CPU. For faster execution, install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Port 5000 already in use

**Cause:** Another Flask instance or process is using the port.

**Solution:**
```bash
# Find the process (Windows):
netstat -ano | findstr :5000

# Kill it:
taskkill /PID <PID> /F
```

---

## License

This project is for academic and research purposes. Please cite the following if you use this work:

- [Materials Project](https://materialsproject.org/) — Jain et al., APL Materials, 2013
- [Matminer](https://hackingmaterials.lbl.gov/matminer/) — Ward et al., Comput. Mater. Sci., 2018
- [CHGNet](https://github.com/CederGroupHub/chgnet) — Deng et al., Nature Machine Intelligence, 2023
- [Pymatgen](https://pymatgen.org/) — Ong et al., Comput. Mater. Sci., 2013
