**Research Proposal: Machine Learning-Accelerated Discovery of Multi-Principal Element Alloys for Extreme Environments**

**Principal Investigator:** Solomon Ahedor

**Affiliation:** Department of Materials & Metallurgical Engineering (Year 4)

**Project:** MetaForge (ML-Driven High Entropy Alloy Discovery Platform)

**Date:** May 2026

# Abstract

Finding the right high-entropy alloy (HEA) used to mean years of expensive trial and error in the lab. While these materials offer amazing thermal stability and strength, their design space is simply too massive to explore manually. This project introduces MetaForge. It’s an end-to-end computational pipeline that leverages physics-informed machine learning and data harvested from the Materials Project to speed up HEA discovery.

So far, the pipeline has achieved some strong baseline results:

Using Random Forest models trained on 132 Matminer Magpie descriptors (chemical properties of materials transformed into mathematical vector matrices for machine learning model) across 5,000 synthetic alloy compositions, hitting a density prediction RMSE of 0.073 g/cm³ and a strength prediction RMSE of 0.539 GPa.

I successfully pinned down a promising refractory HEA candidate

(W₀.₁₀Mo₀.₄₀Ta₀.₀₅Nb₀.₀₅V₀.₄₀) with a predicted specific strength of 9.37 GPa·cm³/g.

The pipeline handles structural validation through CHGNet-relaxed 3×3×3 BCC supercells (54 atoms).

A live version of the tool is running at [metaforge-web.onrender.com](https://metaforge-web.onrender.com/).

This proposal maps out what comes next, which is the immediate focus on DFT validation, feeding more data into the training set, adding new prediction targets and finally synthesizing the top candidates in the lab.

# Background & Motivation

## The Promise of High Entropy Alloys

Most conventional alloys rely on one or two base elements like iron in steel or aluminum in aerospace parts. High Entropy Alloys (first brought to light around 2004 by Yeh and Cantor) completely change the rulebook. They mix five or more elements in roughly equal amounts. Because of the high configurational entropy, they tend to stabilize into simple solid-solution phases like BCC, FCC, or HCP rather than brittle intermetallics.

This leads to some performance benefits:

**Refractory HEAs** (think W-Mo-Ta-Nb-V) can hold their strength well past 1000°C. That makes them highly attractive for nuclear reactor internals or next-gen turbine blades.

**Corrosion-resistant HEAs** (like Co-Cr-Fe-Ni-Cu) tend to hold up much better than standard stainless steels when exposed to harsh marine or chemical environments.

**Lightweight HEAs** (Al-Mg-Li-Ti-Zn) are mostly aimed at aerospace, where shaving off weight without losing strength is everything.

## The Combinatorial Problem

There are roughly 70 metallic elements on the periodic table. If you want to make a 5-element equiatomic alloy, you're looking at over 12 million possible combinations. And if you start tweaking the percentages by just 5% increments, that number explodes into the hundreds of millions. Testing even a tiny fraction of these in a physical lab is economically impossible.

## Machine Learning as an Accelerator

This is where machine learning comes in. Models trained on basic physics descriptors such as atomic radii, electronegativity, valence electron concentration can churn through millions of candidate alloys in seconds. The core idea here is that an alloy's properties are deeply tied to its composition. By converting that composition into mathematical features using frameworks like Matminer's Magpie preset (which generates 132 distinct elemental features), I can teach an algorithm to spot the winners.

# Work Completed

## Data Pipeline

I started by pulling elemental data (atomic radii, density, VEC, electronegativity) directly from the Materials Project API. This covered 17 different elements across 4 main HEA families.

From there, I built a combinatorial engine using the itertools library to generate candidate mixtures and filter out the bad ones using standard physical stability rules. Specifically, I set the lattice strain cutoff at δ < 6.6% to satisfy Hume-Rothery rules and used VEC thresholds to predict the phase (5.0–6.8 for BCC, ≥8.0 for FCC).

## Machine Learning Models

Every candidate composition was featurized into a 132-dimensional Magpie vector, this is because machine learning models do not truly understand chemicals unless transferred into a matrix mathematical formula.

I then trained Random Forest regression models on the filtered compositions.

> **Transparency Note:** The current training targets are **rule-of-mixtures analytical proxies** — density is a composition-weighted average of elemental densities, and strength uses a simplified Varvenne-Luque-Curtin (2016) solid-solution strengthening estimate. Because Magpie descriptors contain the same atomic properties used to compute these proxies, the high R² scores reflect successful function approximation rather than material property discovery. The DFT validation step (Section 4.1) is specifically designed to replace these proxies with physically grounded data.

|  |  |  |  |
| --- | --- | --- | --- |
| **Model** | **Target** | **RMSE** | **R²** |
| Density | RoM density proxy (g/cm³) | 0.073 | 0.99 |
| Strength | SS strengthening proxy (GPa) | 0.539 | 0.95 |

## Genetic Algorithm Inverse Design

To push beyond random screening, I wrote a genetic algorithm that intentionally evolves compositions over 20 generations to maximize the strength-to-weight ratio (specific strength). The algorithm was applied independently to all four alloy categories, isolating a standout optimal candidate for each:

| **Category** | **Composition (at.%)** | **Specific Strength (GPa·cm³/g)** |
| --- | --- | --- |
| **Refractory** | W₉.₄Mo₇₀.₂Ta₁.₆Nb₇.₆V₁₁.₃ | 11.35 |
| **Corrosion** | Co₅.₃Cr₄₀.₁Fe₄₁.₃Ni₁.₈Cu₁₁.₆ | 11.40 |
| **Lightweight** | Al₃₆.₄Mg₁₂.₇Li₁₆.₂Ti₃₃.₄Zn₁.₂ | 9.84 |
| **Aerospace** | Al₂₉.₉Ti₃₅.₁Sc₃₁.₁Zr₀.₇V₃.₂ | 9.35 |

## Structural Relaxation (The Compute Breakthrough)

One of the biggest hurdles was scaling the crystal simulations to 3×3×3 BCC supercells (54 atoms). Non-High-performance hardware (8GB RAM) and free cloud tiers (google colab) completely choked on the memory requirements. The workaround was leveraging CHGNet (a graph neural network). This allowed me to:

Construct 54-atom Special Quasirandom Structures (SQS) for the best candidates.

Relax the atomic positions using the FIRE optimizer with CHGNet handling the energy and force calculations.

Export the optimized blueprints as CIF files so they can be analyzed further.

## Web Deployment

I wrapped the whole prediction engine into a full-stack web app, pairing a Flask backend with a React frontend. Users can drag sliders to adjust the alloy composition and see how the ML model reacts in real-time.

It’s live at: [**metaforge-web.onrender.com**](https://metaforge-web.onrender.com/)

The repository is fully open-source and hosted at [**github.com/SA-FIND/High-EntropyAlloy-Discovery**](https://github.com/SA-FIND/High-Entropy-Alloy-Discovery).

# Proposed Next Steps

## DFT Validation of Top Candidates

**Objective:** To check the ML predictions against first-principles Density Functional Theory (DFT) calculations.

I plan to run full structural relaxations and total energy calculations on the top 5 candidates using VASP or Quantum ESPRESSO.

By calculating the elastic constants (C₁₁, C₁₂, C₄₄), I can derive the bulk, shear, and Young's moduli from the ground up.

This will let me compare the DFT results directly against the ML outputs to see how best the model predicted.

## Expanded Training Data

**Objective:** To feed the model more realistic data (like experimental and CALPHAD results) so it generalizes better.

The plan is to scrape published experimental property data from major HEA studies (e.g., Senkov, Miracle).

I also want to pull in CALPHAD (Calculation of Phase Diagrams) data to verify that the predicted phases actually match thermodynamic reality.

The long-term goal is to grow the training set from 5,000 up to 50,000+ compositions via active learning. The model will essentially flag the compositions it’s least sure about and request targeted DFT runs.

## Additional Property Prediction Targets

**Objective:** To make the pipeline predict more than just density and strength.

**Corrosion resistance:** I want to predict pitting potentials and how the alloy might passivate.

**High-temperature creep:** Specifically looking at creep rates at elevated temperatures (800– 1200°C).

**Thermal conductivity:** For aerospace applications.

**Hardness (Vickers):** A practical metric I can test in the lab.

## Experimental Synthesis & Characterization

**Objective:** To make the discovered alloys in the real world.

**Arc melting:** I will synthesize the best candidates using arc melting with high purity metals.

**XRD analysis:** This will confirm if I actually got the predicted BCC/FCC structure or if secondary phases came up.

**Microhardness testing:** Taking Vickers hardness readings to see if the strength predictions hold any weight.

**SEM/EDS:** Looking at the microstructure to make sure the elements mixed evenly and evaluate other morphological features.

**Corrosion testing:** Running potentio-dynamic polarization in simulated seawater for the corrosion-focused alloys.

## Publication

**Target journals:**

1. Computational Materials Science
2. Journal of Alloys and Compounds

**Target conferences:**

* TMS Annual Meeting (The Computational Materials Science symposium)
* MRS Spring Meeting (Machine Learning for Materials Discovery track)

# Compute & Resource Requirements

|  |  |  |
| --- | --- | --- |
| **Resource** | **Requirement** | **Justification** |
| **High Performance Computing Access** | 1 month | Needed for heavy DFT runs (VASP/QE) on the candidate structures. |
| **GPU Access** | 2 weeks | For running CHGNet relaxations on bigger supercells. |
| **Software Licenses** | VASP license |  |
| **Lab Access** | Arc melting furnace, XRD,  SEM |  |
| **Storage** | ~50 GB | DFT outputs and expanded datasets take up a decent amount of space. |

# Proposed Timeline

|  |  |
| --- | --- |
| **Month** | **Milestone** |
| **Month 1–2** | Get the DFT validation done for the top candidates. Start pulling literature values to expand the training data. |
| **Month 3** | Retrain the models with the new targets (hardness, creep, etc.). |
| **Month 4** | Head to the lab and arc melt the top candidates. |
| **Month 5** | Run the characterization tests (XRD, SEM/EDS, corrosion checks). |
| **Month 6** | Write up the results and submit the manuscript. |

# Collaboration Value

This project provides a fully functional, end-to-end ML pipeline that the research group can start using. The current codebase handles:

Automated data harvesting, featurization and optimization.

Universal models that work across 17 elements and 4 HEA categories.

A live web interface for quick checks.

An open-source code that is easy to build upon.

To really push this forward, working with the mentors or research group's current PhD students would be a massive help in a few key areas:

1. **DFT expertise:** VASP/QE technical understanding from the multiscale modelling, hands-on help would save a lot of time and also as an avenue to learn from experts in the field.
2. **Experimental synthesis:** I’ll need further guidance on the arc melting and characterization equipment and techniques.
3. **Domain knowledge:** Deep insights into HEA thermodynamics, phase stability, multiscale modelling and a deep-dive into material informatics.
4. **Publication mentorship:** Getting advice on how to structure the paper for a high impact journal.

In exchange, this project brings a modern machine learning infrastructure and web deployment toolkit to the lab, which could easily be adapted to screen other types of materials the group/ mentor is studying.

# 7. References

1. Yeh, J.W. et al. (2004). "Nanostructured high-entropy alloys with multiple principal elements." Advanced Engineering Materials.
2. Cantor, B. et al. (2004). "Microstructural development in equiatomic multicomponent alloys." Materials Science and Engineering A.
3. Senkov, O.N. et al. (2018). "Development and exploration of refractory high entropy alloys." Journal of Materials Research.
4. Miracle, D.B. & Senkov, O.N. (2017). "A critical review of high entropy alloys and related concepts." Acta Materialia.
5. Jain, A. et al. (2013). "Commentary: The Materials Project." APL Materials.
6. Ward, L. et al. (2018). "Matminer: An open-source toolkit for materials data mining." Computational Materials Science.
7. Deng, B. et al. (2023). "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling." Nature Machine Intelligence.
8. Ong, S.P. et al. (2013). "Python Materials Genomics (pymatgen)." Computational Materials Science.
9. Zunger, A. et al. (1990). "Special quasirandom structures." Physical Review Letters.
