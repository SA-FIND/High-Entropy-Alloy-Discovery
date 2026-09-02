# Research Proposal: Hierarchical Multi-Fidelity Inverse Design and Atomistic Validation of Multi-Principal Element Alloys for Extreme Environments

**Principal Investigator:** Solomon Ahedor  
**Affiliation:** Department of Materials & Metallurgical Engineering, Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana  
**Project:** MetaForge (Computational Materials Informatics Platform)  
**Date:** Revised Academic Edition — September 2026  

---

## Abstract

Exploring the vast multi-component compositional space of Multi-Principal Element Alloys (MPEAs) and High-Entropy Alloys (HEAs) presents a formidable combinatorial challenge, rendering conventional empirical trial-and-error synthesis economically intractable. This proposal presents **MetaForge**, an Integrated Computational Materials Engineering (ICME) framework that couples physics-informed compositional screening with atomistic graph neural network (GNN) relaxations and a first-principles validation roadmap. 

The framework implements a hierarchical multi-fidelity screening architecture:
1. **Tier 1 (High-Throughput Compositional Screening):** Enforces thermodynamic Hume-Rothery size mismatch (delta <= 6.6%), Guo valence electron concentration (VEC) phase boundaries, Miedema binary mixing enthalpies (-15.0 to +5.0 kJ/mol), and Yang-Zhang thermodynamic parameters (Omega >= 1.1). Surrogates trained on 132-dimensional Matminer Magpie descriptors predict continuum density and dislocation-calibrated yield strengths based on the Taylor-Varvenne dislocation model: `Sigma_y = Sigma_0 + (M * Tau_ss)`.
2. **Tier 2 (Atomistic SQS Annealing & GNN Potentials):** Special Quasirandom Structures (SQS, 54-atom BCC / 48-atom FCC) are synthesized by minimizing Warren-Cowley Short-Range Order (SRO) parameters via Monte Carlo simulated annealing, followed by structural relaxation using the universal CHGNet interatomic neural potential.
3. **Tier 3 (Quantum & Synthesis Roadmap):** Outlines ab-initio Density Functional Theory (DFT) calculations of ground-state elastic stiffness tensors (C11, C12, C44) and vacuum arc remelting synthesis.

Genetic algorithm optimization adhering strictly to the Yeh & Cantor multi-principal element criterion (5 to 35 atomic % per constituent) isolated a standout refractory HEA candidate: **W:18.5% - Mo:24.7% - Ta:26.1% - Nb:16.8% - V:13.9%**. This alloy exhibits a predicted yield strength of **1.96 GPa (1960 MPa)**, relaxed density of **14.29 g/cm³**, and a near-zero formation enthalpy (`Delta-E_f = +0.0286 eV/atom = +2.76 kJ/mol`) that is completely stabilized by high configurational entropy (`Delta-S_mix = 13.4 J/mol·K`). The interactive computational engine is deployed live at [metaforge-web.onrender.com](https://metaforge-web.onrender.com/).

---

## 1. Background & Theoretical Framework

### 1.1 Multi-Principal Element Metallurgy
Unlike classical physical metallurgy—which optimizes properties by perturbing a single principal solvent lattice (e.g., Fe in steels, Ni in superalloys)—High-Entropy Alloys rely on four core metallurgical effects:
* **High Entropy Effect:** High configurational entropy of mixing (`Delta-S_config >= 1.5 * R`) lowers total Gibbs free energy (`Delta-G = Delta-H - T * Delta-S`) at elevated temperatures, suppressing complex brittle intermetallics in favor of disordered solid solutions.
* **Sluggish Diffusion:** Fluctuating local atomic potential wells increase activation energies for vacancy migration, conferring exceptional high-temperature creep resistance.
* **Severe Lattice Distortion:** Size and modulus misfits among distinct constituents generate non-uniform internal stress fields, elevating intrinsic Peierls-Nabarro friction stress.
* **Cocktail Effect:** Non-linear synergistic interactions among constituents yield multi-functional mechanical and chemical enhancements.

### 1.2 The Combinatorial Bottleneck
Selecting 5 principal components from a candidate pool of 70 metallic elements yields over 12 million equiatomic alloy systems. Discretizing compositional variations by 5 atomic % increments expands the search hyper-volume to hundreds of millions of distinct candidates. High-throughput physical synthesis remains cost-prohibitive for broad sweeps. Informatics-guided inverse design is mandatory to down-select optimal candidates prior to laboratory synthesis.

---

## 2. Research Methodology & Multi-Fidelity Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   METAFORGE HIERARCHICAL WORKFLOW                      │
├────────────────────────────────────────────────────────────────────────┤
│  [Tier 1] Combinatorial Screening (Simplex Discretization)             │
│   • Hume-Rothery Lattice Distortion Filter: delta <= 6.6%              │
│   • Guo Valence Electron Concentration (VEC) Phase Stability           │
│   • Miedema Enthalpy of Mixing: -15.0 <= Delta-H_mix <= +5.0 kJ/mol    │
│   • Yang & Zhang Thermodynamic Ratio: Omega >= 1.1                     │
│   • Featurization: 132 Magpie Descriptors (Matminer)                   │
│   • Surrogate Property Modeling:                                       │
│       - Continuum Density (g/cm³)                                      │
│       - Dislocation Yield Strength: Sigma_y = Sigma_0 + M * Tau_ss     │
│                                                                        │
│  [Tier 2] Atomistic Generation & Neural Potential Relaxation           │
│   • SQS Synthesis: Warren-Cowley SRO Minimization (Monte Carlo)        │
│   • 54-atom BCC / 48-atom FCC Supercell Geometry                       │
│   • Universal Interatomic Potential Relaxation (CHGNet)                │
│   • Formation Energy: Delta-E_f = E_relaxed/N - Sum(c_i * E_ref)       │
│                                                                        │
│  [Tier 3] Validation & Laboratory Synthesis Roadmap                    │
│   • Ab-Initio Quantum DFT (VASP / Quantum ESPRESSO)                    │
│   • Elastic Stiffness Tensors (C11, C12, C44) & Moduli                 │
│   • Vacuum Arc Remelting, XRD Phase Analysis, Vickers Microhardness    │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Thermodynamic Phase Stability & Solid-Solution Criteria
To ensure that predicted alloys stabilize as disordered single-phase solid solutions, candidate compositions must satisfy four coupled thermodynamic criteria:

```
1. ATOMIC SIZE MISMATCH (delta):
   delta = 100 * SquareRoot( Sum of [ c_i * (1 - r_i / r_average)^2 ] ) <= 6.6%
   where r_i is atomic radius and r_average is average radius.

2. MIEDEMA ENTHALPY OF MIXING (Delta-H_mix):
   Delta-H_mix = Sum of [ 4 * H_ij * c_i * c_j ]
   Enforced Range: -15.0 kJ/mol <= Delta-H_mix <= +5.0 kJ/mol.

3. YANG-ZHANG THERMODYNAMIC RATIO (Omega):
   Omega = ( Average Melting Point * Delta-S_config ) / | Delta-H_mix | >= 1.1

4. VALENCE ELECTRON CONCENTRATION (VEC):
   VEC = Sum of [ c_i * VEC_i ]
   - VEC < 6.87 ==> Pure Body-Centered Cubic (BCC)
   - VEC >= 8.0 ==> Pure Face-Centered Cubic (FCC)
```

### 2.2 Yield Strength Formulation (Dislocation Theory)
Rather than treating yield strength as an empirical proxy, MetaForge implements a physical dislocation model based on the Taylor factor and Varvenne-Luque-Curtin solid-solution strengthening:

```
EQUATION: Dislocation Yield Strength
Sigma_y = Sigma_0 + [ M * Tau_ss ]

Where:
• Sigma_0 = Intrinsic Peierls-Nabarro friction stress of pure lattice 
            (approx. G_rom / 150 for BCC; G_rom / 350 for FCC).
• M       = Polycrystalline Taylor factor (2.73 for BCC; 3.06 for FCC).
• Tau_ss  = Solid-solution strengthening increment:
            Tau_ss = 0.05 * G_rom * (delta)^(2/3)
• G_rom   = Rule-of-mixtures average shear modulus.
```

---

## 3. Results & Discovered Candidate Alloys

Genetic algorithm inverse design adhering to the formal HEA constraint (5% to 35 atomic % per element) identified the following optimal alloys:

| Category | Crystal System | Discovered Composition (at.%) | Density (g/cm³) | Yield Strength (Sigma_y) | Formation Energy (Delta-E_f) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Refractory** | BCC (54-atom) | W:18.5% - Mo:24.7% - Ta:26.1% - Nb:16.8% - V:13.9% | 14.29 | **1.96 GPa (1960 MPa)** | +0.0286 eV/atom (+2.76 kJ/mol) |
| **Corrosion** | FCC (48-atom) | Co:8.6% - Cr:35.4% - Fe:33.1% - Ni:11.2% - Cu:11.6% | 8.00 | **0.74 GPa (740 MPa)** | +0.1325 eV/atom |
| **Lightweight** | FCC (48-atom) | Al:35.7% - Mg:11.5% - Li:5.1% - Ti:35.7% - Zn:11.9% | 3.54 | **0.63 GPa (630 MPa)** | +0.0966 eV/atom |
| **Aerospace** | FCC (48-atom) | Al:5.9% - Ti:32.2% - Sc:6.9% - Zr:35.0% - V:20.1% | 5.01 | **0.83 GPa (830 MPa)** | +0.2126 eV/atom |

---

## 4. Proposed Investigation: DFT Validation & Experimental Synthesis

### 4.1 Ab-Initio Density Functional Theory (Tier 3)
* **Objective:** Establish quantum ground-truth total energies and elastic constants for relaxed SQS supercells.
* **Methodology:** Plane-wave DFT calculations via VASP or Quantum ESPRESSO using Projector Augmented-Wave (PAW) pseudopotentials under Generalized Gradient Approximation (GGA-PBEsol). A plane-wave kinetic energy cutoff of 520 eV and Monkhorst-Pack k-point meshes with a density of 0.03 1/Å will be enforced.
* **Elastic Tensor Derivation:** Strains of +/- 1% and +/- 2% will be applied to compute the elastic stiffness matrix (C11, C12, C44), from which the Voigt-Reuss-Hill polycrystalline bulk modulus (B), shear modulus (G), Young's modulus (E), and Pugh's ductility ratio (B/G) will be evaluated.

### 4.2 Vacuum Arc Remelting & Experimental Characterization
* **Synthesis:** Ingot synthesis of the top candidate (W-Mo-Ta-Nb-V) via vacuum non-consumable arc melting under ultra-pure Argon (99.999%). Ingots will be flipped and remelted a minimum of seven times to ensure chemical homogeneity, followed by homogenization annealing in a high-vacuum tube furnace (1200°C for 24 h).
* **X-Ray Diffraction (XRD):** Rigaku SmartLab diffractometer (Cu-K_alpha radiation) to verify single-phase BCC solid-solution crystallinity and confirm the absence of secondary intermetallic reflections.
* **Electron Microscopy (SEM-EDS):** JEOL field-emission scanning electron microscope with energy-dispersive X-ray spectroscopy mapping to analyze microstructural dendrite segregation and partition coefficients.
* **Mechanical Testing:** Microhardness testing (Vickers diamond indenter, 500 gf load, 15 s dwell) across a 10 x 10 indentation matrix to validate predicted yield strengths via the empirical Tabor relation (`Yield Strength ≈ Hardness / 3`).

---

## 5. Compute Resources, Timeline, and Strategic Value

### 5.1 Resource Allocation

| Resource | Requirement | Technical Justification |
| :--- | :--- | :--- |
| **High-Performance Computing (HPC)** | 100,000 core-hours | High-accuracy ab-initio DFT relaxations and elastic tensor runs across 54-atom supercells. |
| **Software Infrastructure** | VASP 6.x / QE 7.x | Validated PAW pseudopotentials for transition metals and refractory species. |
| **Arc Melting & Metrology** | KNUST / Partner Lab | Non-consumable tungsten arc furnace, inert atmosphere box, Rigaku XRD, FE-SEM. |

### 5.2 Project Timeline (6 Months)

```
Month 1: Ab-Initio Quantum DFT on SQS Supercells (C11, C12, C44, E_0)
Month 2: Active Learning Retraining (Interfacing DFT outputs with MetaForge models)
Month 3: Vacuum Arc Melting Synthesis of W-Mo-Ta-Nb-V Refractory Ingot
Month 4: XRD Phase Identification & SEM-EDS Chemical Homogeneity Mapping
Month 5: Vickers Microhardness & High-Temperature Oxidation Assessment
Month 6: Manuscript Preparation & Submission to Computational Materials Science
```

---

## 6. References

1. Yeh, J. W., et al. (2004). "Nanostructured high-entropy alloys with multiple principal elements: novel alloy design concepts and outcomes." *Advanced Engineering Materials*, 6(5), 299-303.
2. Cantor, B., et al. (2004). "Microstructural development in equiatomic multicomponent alloys." *Materials Science and Engineering: A*, 375, 213-218.
3. Senkov, O. N., et al. (2011). "Mechanical properties of Nb25Mo25Ta25W25 and V20Nb20Mo20Ta20W20 refractory high entropy alloys." *Intermetallics*, 19(5), 698-706.
4. Miracle, D. B., & Senkov, O. N. (2017). "A critical review of high entropy alloys and related concepts." *Acta Materialia*, 122, 448-511.
5. Varvenne, C., Luque, A., & Curtin, W. A. (2016). "Theory of strengthening in dilute and concentrated solid-solution alloys." *Acta Materialia*, 118, 164-176.
6. Takeuchi, A., & Inoue, A. (2005). "Classification of bulk metallic glasses by atomic size difference, heat of mixing and period of constituents." *Materials Transactions*, 46(12), 2817-2829.
7. Deng, B., et al. (2023). "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling." *Nature Machine Intelligence*, 5(9), 1031-1041.
8. Ward, L., et al. (2018). "Matminer: An open-source toolkit for materials data mining." *Computational Materials Science*, 152, 60-69.
9. Zunger, A., et al. (1990). "Special quasirandom structures." *Physical Review Letters*, 65(3), 353.
