# Research Proposal: Machine Learning-Accelerated Discovery of Multi-Principal Element Alloys for Extreme Environments

**Principal Investigator:** Solomon Ahedor  
**Affiliation:** Department of Metallurgical & Materials Engineering (Year 4)  
**Project:** MetaForge — ML-Driven High Entropy Alloy Discovery Platform  
**Date:** May 2026

---

## Abstract

Finding the right high-entropy alloy (HEA) used to mean years of expensive trial and error in the lab. While these materials offer amazing thermal stability and strength, their design space is simply too massive to explore manually. This project introduces MetaForge. It’s an end-to-end computational pipeline that leverages physics-informed machine learning and data harvested from the Materials Project to dramatically speed up HEA discovery.

So far, the pipeline has achieved some strong baseline results:

- Using Random Forest models trained on 132 Matminer Magpie descriptors across 5,000 synthetic alloy compositions, we hit a density prediction RMSE of 0.073 g/cm³ and a strength prediction RMSE of 0.539 GPa.
- We successfully pinned down a promising refractory HEA candidate (W₀.₁₀Mo₀.₄₀Ta₀.₀₅Nb₀.₀₅V₀.₄₀) with a predicted specific strength of 9.37 GPa·cm³/g.
- The pipeline handles structural validation through CHGNet-relaxed 3×3×3 BCC supercells (54 atoms).
- A live version of the tool is up and running at [metaforge-web.onrender.com](https://metaforge-web.onrender.com/).

This proposal maps out what comes next. The immediate focus will be on DFT validation, feeding more data into the training set, adding new prediction targets, and finally synthesizing the top candidates in the lab.

---

## 1. Background & Motivation

### 1.1 The Promise of High Entropy Alloys

Most conventional alloys rely on one or two base elements—like iron in steel or aluminum in aerospace parts. High Entropy Alloys (first brought to light around 2004 by Yeh and Cantor) completely change the rulebook. They mix five or more elements in roughly equal amounts. Because of the high configurational entropy, they tend to stabilize into simple solid-solution phases like BCC, FCC, or HCP rather than brittle intermetallics.

This leads to some performance benefits:

- **Refractory HEAs** (think W-Mo-Ta-Nb-V) can hold their strength well past 1000°C. That makes them highly attractive for nuclear reactor internals or next-gen turbine blades.
- **Corrosion-resistant HEAs** (like Co-Cr-Fe-Ni-Cu) tend to hold up much better than standard stainless steels when exposed to harsh marine or chemical environments.
- **Lightweight HEAs** (Al-Mg-Li-Ti-Zn) are mostly aimed at aerospace, where shaving off weight without losing strength is everything.

### 1.2 The Combinatorial Explosion Problem

There are roughly 70 metallic elements on the periodic table. If you want to make a 5-element equiatomic alloy, you're looking at over 12 million possible combinations. And if you start tweaking the percentages by just 5% increments, that number explodes into the hundreds of millions. Testing even a tiny fraction of these in a physical lab is economically impossible.

### 1.3 Machine Learning as an Accelerator

This is where machine learning comes in. Models trained on basic physics descriptors—atomic radii, electronegativity, valence electron concentration—can churn through millions of candidate alloys in seconds. The core idea here is that an alloy's properties are deeply tied to its composition. By converting that composition into mathematical features using frameworks like Matminer's Magpie preset (which generates 132 distinct elemental features), we can teach an algorithm to spot the winners.

---

## 2. Work Completed

### 2.1 Data Pipeline

- I started by pulling elemental data (atomic radii, density, VEC, electronegativity) directly from the Materials Project API. This covered 17 different elements across 4 main HEA families.
- From there, I built a combinatorial engine to generate candidate mixtures and filter out the bad ones using standard physical stability rules. Specifically, I set the lattice strain cutoff at δ < 6.6% to satisfy Hume-Rothery rules and used VEC thresholds to predict the phase (5.0–6.8 for BCC, ≥8.0 for FCC).

### 2.2 Machine Learning Models

- Every candidate composition was featurized into a 132-dimensional Magpie vector.
- I then trained Random Forest regression models on a synthetic dataset of 5,000 alloys.

| Model | Target | RMSE | R² |
|-------|--------|------|----|
| Density | Bulk density (g/cm³) | 0.073 | ~0.99 |
| Strength | Shear strength (GPa) | 0.539 | ~0.95 |

### 2.3 Genetic Algorithm Inverse Design

- To push beyond random screening, I wrote a genetic algorithm that intentionally evolves compositions over 20 generations to maximize the strength-to-weight ratio (specific strength).
- The algorithm eventually isolated a standout refractory candidate:

| Property | Value |
|----------|-------|
| Composition | W₀.₁₀ Mo₀.₄₀ Ta₀.₀₅ Nb₀.₀₅ V₀.₄₀ |
| Predicted Density | 9.68 g/cm³ |
| Predicted Strength | 90.71 GPa |
| Specific Strength | 9.37 GPa·cm³/g |

### 2.4 Structural Relaxation (The Compute Breakthrough)

One of the biggest hurdles was scaling the crystal simulations to 3×3×3 BCC supercells (54 atoms). Standard local hardware (like my 8GB laptop) and free cloud tiers completely choked on the memory requirements. The workaround was leveraging CHGNet—a graph neural network interatomic potential. This allowed us to:

- Construct 54-atom Special Quasirandom Structures (SQS) for the best candidates.
- Relax the atomic positions using the FIRE optimizer, with CHGNet handling the energy and force calculations.
- Export the optimized blueprints as CIF files so they can be analyzed further.

### 2.5 Web Deployment

- I wrapped the whole prediction engine into a full-stack web app, pairing a Flask backend with a React frontend. Users can drag sliders to adjust the alloy composition and instantly see how the ML model reacts.
- It’s live right now at: **[metaforge-web.onrender.com](https://metaforge-web.onrender.com/)**
- The repository is fully open-source and hosted at **[github.com/SA-FIND/High-Entropy-Alloy-Discovery](https://github.com/SA-FIND/High-Entropy-Alloy-Discovery)**.

---

## 3. Proposed Next Steps

### 3.1 DFT Validation of Top Candidates

**Objective:** Check the ML predictions against first-principles Density Functional Theory (DFT) calculations.

- I plan to run full structural relaxations and total energy calculations on the top 5 candidates using VASP or Quantum ESPRESSO.
- By calculating the elastic constants (C₁₁, C₁₂, C₄₄), I can derive the bulk, shear, and Young's moduli from the ground up.
- This will let me compare the DFT results directly against the ML outputs to see where the model is biased.

### 3.2 Expanded Training Data

**Objective:** Feed the model more realistic data (like experimental and CALPHAD results) so it generalizes better.

- The plan is to scrape published experimental property data from major HEA studies (e.g., Senkov, Miracle).
- I also want to pull in CALPHAD (Calculation of Phase Diagrams) data to verify that the predicted phases actually match thermodynamic reality.
- Long term, the goal is to grow the training set from 5,000 up to 50,000+ compositions via active learning. The model will essentially flag the compositions it’s least sure about and request targeted DFT runs.

### 3.3 Additional Property Prediction Targets

**Objective:** Make the pipeline predict more than just density and strength.

- **Corrosion resistance:** I want to predict pitting potentials and how the alloy might passivate.
- **High-temperature creep:** Specifically looking at creep rates when things get hot (800–1200°C).
- **Thermal conductivity:** This is a huge deal for aerospace applications.
- **Hardness (Vickers):** A practical metric we can easily test in the lab.

### 3.4 Experimental Synthesis & Characterization

**Objective:** Actually make the top 3 discovered alloys in the real world.

- **Arc melting:** We’ll synthesize the best candidates using vacuum arc melting with high-purity metals.
- **XRD analysis:** This will confirm if we actually got the predicted BCC/FCC structure or if nasty secondary phases popped up.
- **Microhardness testing:** Taking Vickers hardness readings to see if the strength predictions hold any weight.
- **SEM/EDS:** Looking at the microstructure to make sure the elements mixed evenly.
- **Corrosion testing:** Running potentiodynamic polarization in simulated seawater for the corrosion-focused alloys.

### 3.5 Publication

**Target journals (in order of preference):**

1. *Computational Materials Science* (Elsevier) — Fits perfectly with the ML and materials overlap.
2. *Journal of Alloys and Compounds* (Elsevier) — They have a huge HEA audience.
3. *Acta Materialia* (Elsevier) — A bit ambitious, but possible if the experimental validation turns out well.

**Target conferences:**

- TMS Annual Meeting (specifically the Computational Materials Science symposium)
- MRS Spring Meeting (Machine Learning for Materials Discovery track)

---

## 4. Compute & Resource Requirements

| Resource | Requirement | Justification |
|----------|-------------|---------------|
| **HPC Access** | 500–1,000 CPU-hours | Needed for heavy DFT runs (VASP/QE) on the 5 candidate structures. |
| **GPU Access** | 50–100 GPU-hours | For running CHGNet relaxations on bigger supercells (like 4×4×4, 128 atoms). |
| **Software Licenses** | VASP license | Most university HPCs already have this, but it’s required for the DFT work. |
| **Lab Access** | Arc melting furnace, XRD, SEM | Obviously required to physically make and test the samples. |
| **Storage** | ~50 GB | DFT outputs and expanded datasets take up a decent amount of space. |

---

## 5. Proposed Timeline

| Month | Milestone |
|-------|-----------|
| **Month 1–2** | Get the DFT validation done for the top 5 candidates. Start pulling literature values to expand the training data. |
| **Month 3** | Retrain the models with the new targets (hardness, creep, etc.). |
| **Month 4** | Head to the lab and arc melt the top 3 candidates. |
| **Month 5** | Run the characterization tests (XRD, SEM/EDS, corrosion checks). |
| **Month 6** | Write up the results and submit the manuscript. |

---

## 6. Collaboration Value

This project provides a fully functional, end-to-end ML pipeline that the research group can start using immediately. The current codebase already handles:

- Automated data harvesting, featurization, and optimization.
- Universal models that work across 17 elements and 4 HEA categories.
- A live web interface for quick checks.
- Clean, open-source code that is easy to build upon.

To really push this forward, working with the group's current PhD students would be a massive help in a few key areas:

1. **DFT expertise:** Getting VASP/QE jobs to converge properly can be tricky, and hands-on help would save a lot of time.
2. **Experimental synthesis:** I’ll need guidance on the arc melting and characterization equipment.
3. **Domain knowledge:** Deep insights into HEA thermodynamics and phase stability.
4. **Publication mentorship:** Getting advice on how to structure the paper for a high-impact journal.

In exchange, this project brings a modern machine learning infrastructure and web deployment toolkit to the lab, which could easily be adapted to screen other types of materials the group is studying.

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
