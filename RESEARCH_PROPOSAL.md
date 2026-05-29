# Research Proposal: Machine Learning-Accelerated Discovery of Multi-Principal Element Alloys for Extreme Environments

**Principal Investigator:** Solomon Ahedor  
**Affiliation:** Kwame Nkrumah University of Science and Technology, Department of Metallurgical & Materials Engineering (Year 4)  
**Project:** MetaForge — ML-Driven High Entropy Alloy Discovery Platform  
**Date:** May 2026

---

## Abstract

High Entropy Alloys (HEAs) represent a paradigm shift in materials design, offering unprecedented combinations of strength, corrosion resistance, and thermal stability. However, the vast combinatorial space of multi-principal element alloys makes traditional trial-and-error discovery prohibitively slow and expensive. This project presents **MetaForge**, an end-to-end computational pipeline that combines data harvesting from the Materials Project, physics-informed machine learning, genetic algorithm-driven inverse design, and graph neural network (GNN) structural relaxation to accelerate HEA discovery.

To date, the pipeline has achieved:

- **Density prediction RMSE of 0.073 g/cm³** and **strength prediction RMSE of 0.539 GPa** using Random Forest models trained on 132 Matminer Magpie descriptors across 5,000 synthetic alloy compositions.
- Successful identification of a refractory HEA candidate (**W₀.₁₀Mo₀.₄₀Ta₀.₀₅Nb₀.₀₅V₀.₄₀**) with a predicted specific strength of **9.37 GPa·cm³/g**.
- CHGNet-relaxed 3×3×3 BCC supercells (54 atoms) for structural validation.
- A live, publicly accessible web interface deployed at [metaforge-web.onrender.com](https://metaforge-web.onrender.com/).

This proposal outlines the next phase of research: DFT validation, expanded training data, additional property prediction targets, and experimental synthesis of top candidates.

---

## 1. Background & Motivation

### 1.1 The Promise of High Entropy Alloys

Conventional alloys are designed around one or two principal elements (e.g., Fe in steel, Al in aerospace alloys). High Entropy Alloys, first reported independently by Yeh et al. (2004) and Cantor et al. (2004), break this paradigm by combining five or more elements in near-equiatomic proportions. The high configurational entropy stabilizes simple solid-solution phases (BCC, FCC, HCP), producing materials with remarkable properties:

- **Refractory HEAs** (e.g., W-Mo-Ta-Nb-V): Retain strength above 1000°C, making them candidates for next-generation turbine blades and nuclear reactor components.
- **Corrosion-resistant HEAs** (e.g., Co-Cr-Fe-Ni-Cu): Outperform conventional stainless steels in marine and chemical processing environments.
- **Lightweight HEAs** (e.g., Al-Mg-Li-Ti-Zn): Target aerospace applications where specific strength (strength-to-weight ratio) is the critical metric.

### 1.2 The Combinatorial Explosion Problem

With ~70 metallic elements in the periodic table, the number of possible 5-element equiatomic combinations exceeds **12 million**. Even restricting to non-equiatomic compositions at 5% increments yields millions more candidates. Experimental synthesis and characterization of even a fraction of this space is economically infeasible.

### 1.3 Machine Learning as an Accelerator

Machine learning models trained on physics-derived descriptors (atomic radii, electronegativity, valence electron concentration) can screen millions of candidates in seconds. The key insight is that **material properties are encoded in composition**, and composition can be mathematically represented using established descriptor frameworks like Matminer's Magpie preset (132 features derived from elemental properties).

---

## 2. Work Completed

### 2.1 Data Pipeline

- Harvested elemental properties (atomic radii, density, VEC, electronegativity) for 17 elements across 4 HEA families from the **Materials Project API**.
- Built a combinatorial engine using the itertools library that generates thousands of candidate compositions and filters them using physics-based stability criteria:
  - Lattice strain δ < 6.6% (Hume-Rothery rules)
  - VEC thresholds for phase prediction (BCC: 5.0–6.8, FCC: ≥8.0)

### 2.2 Machine Learning Models

- Featurized all candidate compositions into 132-dimensional Magpie descriptor vectors using **Matminer**.
- Trained **Random Forest** regression models on 5,000 synthetic alloy compositions spanning all 17 elements.

| Model | Target | RMSE | R² |
|-------|--------|------|----|
| Density | Bulk density (g/cm³) | 0.073 | ~0.99 |
| Strength | Shear strength (GPa) | 0.539 | ~0.95 |

### 2.3 Genetic Algorithm Inverse Design

- Implemented a genetic algorithm that evolves alloy compositions over 20 generations to **maximize specific strength** (strength/density).
- The GA discovered an optimal refractory candidate:

| Property | Value |
|----------|-------|
| Composition | W₀.₁₀ Mo₀.₄₀ Ta₀.₀₅ Nb₀.₀₅ V₀.₄₀ |
| Predicted Density | 9.68 g/cm³ |
| Predicted Strength | 90.71 GPa |
| Specific Strength | 9.37 GPa·cm³/g |

### 2.4 Structural Relaxation (The Compute Breakthrough)

A critical milestone was achieved by scaling the crystal simulation to **3×3×3 BCC supercells (54 atoms)**. Initial attempts on a local 8GB laptop and Google Colab (12GB) failed due to insufficient memory. By optimizing the cell size and leveraging **CHGNet** (a graph neural network interatomic potential), I successfully:

- Built 54-atom Special Quasirandom Structures (SQS) for top candidates.
- Relaxed atomic positions using the FIRE optimizer with CHGNet as the energy/force calculator.
- Exported optimized structures as CIF files for further analysis.
- These .cif files are available to view in crystal structure web viewers

### 2.5 Web Deployment

- Built a full-stack web application (**Flask** backend + **React** frontend) that allows users to interactively adjust alloy compositions and receive real-time ML predictions.
- Deployed the application at: **[metaforge-web.onrender.com](https://metaforge-web.onrender.com/)**
- Open-sourced the full codebase: **[github.com/SA-FIND/High-Entropy-Alloy-Discovery](https://github.com/SA-FIND/High-Entropy-Alloy-Discovery)**

---

## 3. Proposed Next Steps

### 3.1 DFT Validation of Top Candidates

**Objective:** Validate ML predictions using first-principles Density Functional Theory (DFT) calculations. 

(As an undergraduate student, I am still learning computational materials with a focus on material informatics)

- Perform full structural relaxation and total energy calculations using **VASP** or **Quantum ESPRESSO** on the top 5 GA-discovered candidates.
- Calculate elastic constants (C₁₁, C₁₂, C₄₄) to derive bulk modulus, shear modulus, and Young's modulus from first principles.
- Compare DFT-predicted properties against ML predictions to quantify model accuracy and identify systematic biases.

### 3.2 Expanded Training Data

**Objective:** Improve model generalization by incorporating experimental and CALPHAD data.

- Integrate experimental HEA property data from published literature (Senkov et al., Miracle et al.).
- Incorporate **CALPHAD** (Calculation of Phase Diagrams) data for thermodynamic validation of predicted stable phases.
- Expand the training set from 5,000 to 50,000+ compositions using active learning, where the model identifies compositions it is least confident about and requests targeted DFT calculations.

### 3.3 Additional Property Prediction Targets

**Objective:** Expand the ML pipeline beyond density and strength.

- **Corrosion resistance:** Train models to predict pitting potential and passivation behavior.
- **High-temperature creep resistance:** Predict creep rate at elevated temperatures (800–1200°C).
- **Thermal conductivity:** Critical for thermal management in aerospace applications.
- **Hardness (Vickers):** Direct experimental validation metric.

### 3.4 Experimental Synthesis & Characterization

**Objective:** Physically validate the top 3 ML-discovered candidates.

- **Arc melting:** Synthesize top candidates using vacuum arc melting with high-purity elemental feedstocks.
- **XRD analysis:** Confirm predicted crystal structure (BCC/FCC) and detect any secondary phases.
- **Microhardness testing:** Vickers hardness measurements to validate strength predictions.
- **SEM/EDS:** Characterize microstructure and confirm elemental homogeneity.
- **Corrosion testing:** Potentiodynamic polarization in simulated seawater (for corrosion-resistant candidates).

### 3.5 Publication

**Target journals (in order of preference):**

1. *Computational Materials Science* (Elsevier) — directly aligned with the ML+materials theme.
2. *Journal of Alloys and Compounds* (Elsevier) — strong HEA readership.
3. *Acta Materialia* (Elsevier) — high-impact, if experimental validation is included.

**Target conferences:**

- TMS Annual Meeting — Computational Materials Science symposium
- MRS Spring Meeting — Machine Learning for Materials Discovery

( I am willing to learn under good guildance in the exploration of computational materials)
This project was primary inspired by Prof. Emmanuel Kwesi Arthur at KNUST (https://webapps.knust.edu.gh/staff/dirsearch/profile/summary/2d5b0c584928.html) on his teaching of non-ferrous alloys, Prof. Kwadwo Mensah-Darkwa at KNUST on his teaching of MatLab and software related program at the materials and metallurgical department (https://webapps.knust.edu.gh/staff/dirsearch/profile/summary/0719f2655b8c.html) and Prof. Taylor Sparks on his teaching of materials informatics at the University of Utah. (https://profiles.faculty.utah.edu/u0203991)


---

## 4. Compute & Resource Requirements

| Resource | Requirement | Justification |
|----------|-------------|---------------|
| **HPC Access** | 500–1,000 CPU-hours | DFT calculations (VASP/QE) for 5 candidate structures |
| **GPU Access** | 50–100 GPU-hours | CHGNet relaxation of larger supercells (5×5x5) |
| **Software Licenses** | VASP license | DFT calculations (available through most university HPC centers) |
| **Lab Access** | Arc melting furnace, XRD, SEM | Experimental synthesis and characterization |
| **Storage** | ~50 GB | DFT output files, expanded training datasets |

---

## 5. Proposed Timeline

| Month | Milestone |
|-------|-----------|
| **Month 1–2** | DFT validation of top 5 candidates; expand training data with literature values |
| **Month 3** | Train expanded ML models with new property targets (corrosion, creep, hardness) |
| **Month 4** | Experimental synthesis of top 3 candidates (arc melting) |
| **Month 5** | Characterization (XRD, SEM/EDS, microhardness, corrosion testing) |
| **Month 6** | Manuscript preparation and submission |

---

## 6. Collaboration Value

This project brings a **complete, functional ML pipeline** to the research group. The existing codebase includes:

- A fully automated data harvesting → featurization → training → optimization pipeline.
- Pre-trained universal models covering 17 elements across 4 HEA families.
- A live web deployment for interactive exploration.
- Open-source code ready for extension.

The primary areas where **collaboration with the research groups, Professors and PhD students** would be invaluable:

1. **DFT expertise:** Running and interpreting VASP/QE calculations.
2. **Experimental synthesis:** Access to arc melting equipment and characterization facilities.
3. **Domain knowledge:** Deep understanding of HEA thermodynamics, phase stability, material selection and quality assurance, computational materials, characterisation, mechanical behavior, etc.
4. **Publication mentorship:** Guidance on manuscript preparation for high-impact journals.

In return, this project contributes ML infrastructure, web deployment capabilities, and a novel screening methodology that can be applied to a group's broader research portfolio.

---

## 7. References

1. Yeh, J.W. et al. (2004). "Nanostructured high-entropy alloys with multiple principal elements." *Advanced Engineering Materials*, 6(5), 299–303.
2. Cantor, B. et al. (2004). "Microstructural development in equiatomic multicomponent alloys." *Materials Science and Engineering A*, 375–377, 213–218.
3. Senkov, O.N. et al. (2018). "Development and exploration of refractory high entropy alloys." *Journal of Materials Research*, 33(19), 3092–3128.
4. Miracle, D.B. & Senkov, O.N. (2017). "A critical review of high entropy alloys and related concepts." *Acta Materialia*, 122, 448–511.
5. Jain, A. et al. (2013). "Commentary: The Materials Project." *APL Materials*, 1(1), 011002.
6. Ward, L. et al. (2018). "Matminer: An open source toolkit for materials data mining." *Computational Materials Science*, 152, 60–69.
7. Deng, B. et al. (2023). "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling." *Nature Machine Intelligence*, 5, 1031–1041.
8. Ong, S.P. et al. (2013). "Python Materials Genomics (pymatgen)." *Computational Materials Science*, 68, 314–319.
9. Zunger, A. et al. (1990). "Special quasirandom structures." *Physical Review Letters*, 65(3), 353.

---

*This proposal accompanies the MetaForge codebase at [github.com/SA-FIND/High-Entropy-Alloy-Discovery](https://github.com/SA-FIND/High-Entropy-Alloy-Discovery)*
