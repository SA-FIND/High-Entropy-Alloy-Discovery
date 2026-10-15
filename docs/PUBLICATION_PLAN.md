# MetaForge: fix-and-publish plan

Written 2026-10-15. Approved by Solomon the same day: refractory-only scope; free (subscription-route) journal he can submit to himself; experimental collaboration (WP8) after the paper; cleaned v1 replaces the old paper now, with an archive for retired material.

## 0. Verification pass (2026-10-15), before any work

**Journal (no fee, independent submission).**
- *Modelling and Simulation in Materials Science and Engineering* (IOP, hybrid): IOP's own page says "Publication on a subscription-access basis is free of charge." Scope: new modelling methods / processing-structure-property links. Confirmed.
- *Computational Materials Science* (Elsevier, hybrid): subscription route is normally free at Elsevier; ScienceDirect blocked automated checking, so confirm on the Guide for Authors before submitting. Best scope fit.
- Dropped: *Materials & Design* (gold OA, APC about USD 3,150-3,750) and *Digital Discovery* (gold OA, APC up to GBP 2,100; launch waiver ended mid-2024).
- Decision: primary target Computational Materials Science, fallback MSMSE. Both accept single-author submissions; affiliation can be KNUST (where the work started) plus "independent researcher". A preprint on arXiv (cond-mat.mtrl-sci) before submission is allowed by both and costs nothing.

**Maresca-Curtin reduced edge model (from arXiv 1901.02100, Acta Mater. 182 (2020) 235-249).**
- tau_y0 = 0.040 alpha^(-1/3) mu [(1+nu)/(1-nu)]^(4/3) [sum c_n dV_n^2 / b^6]^(2/3)
- dE_b = 2.00 alpha^(1/3) mu b^3 [(1+nu)/(1-nu)]^(2/3) [sum c_n dV_n^2 / b^6]^(1/3)
- tau_y(T) = tau_y0 [1 - (kT/dE_b ln(eps0/eps))^(2/3)] for tau/tau_y0 > 0.5; tau_y0 exp[-(1/0.55)(kT/dE_b) ln(eps0/eps)] below.
- alpha = 1/12, eps0 = 1e4 1/s, Taylor factor M = 3.067; C_ij by rule of mixtures; mu = sqrt(C44 (C11-C12)/2); nu from B = (C11+2C12)/3 (the extracted text dropped the 1/3; with it nu(MoNbTaW) = 0.31, without it 0.43, so the 1/3 is right).
- Validation inputs (paper Table 2, MoNbTaW, Vegard/ROM): a = 3.228 A; dV = Mo -1.293, Nb 1.135, Ta 1.168, W -1.010 A^3; C11 375.5, C12 167.3, C44 101.6 GPa.
- The Baruffi et al. Materials Cloud dataset could not be located; validate against the published MoNbTaW / MoNbTaVW curves and against MPEA measurements instead.

**Data.**
- Couzinie 2018 is only available as image tables + PDF supplement (not machine-readable). Not needed: the MPEA dataset already holds 538 yield-strength rows on 157 refractory-type alloys (276 at room temperature, 193 at >= 600 C), years 2011-2019. Temporal holdout = 2019 papers (187 rows).
- Senkov 2010 values in benchmark_chgnet.py (3.2134 A / 13.75 g/cm3; 3.1832 A / 12.36 g/cm3) confirmed against the AFRL preprint (DTIC ADA550824).

**Software.** MACE (`pip install mace-torch`, `mace_mp`) and SevenNet (`pip install sevenn`, torch first, `SevenNetCalculator(model='7net-0', device='cpu')`) run on CPU. MatterSim on Windows is unconfirmed (source build needs conda); try `pip install mattersim`, drop it if it fails. **Outcome (2026-10-15):** CHGNet 0.4.2, MACE (mace-torch) and SevenNet 0.10.4 installed in `venvs/mlip`; MatterSim failed to build its wheel on Windows and is dropped (state this in the paper).

**Reference audit of the current paper (Crossref, 56 references).** Likely non-existent: [5] Lu & Chen JNM 2020, [29] DeCost et al. Acta Mater 2019. Wrong: [16] (no such Miracle-Senkov-Couzinie JMR 2020 review; real one is Senkov et al., JMR 33 (2018) 3092), [18] Rao et al. title and authors, [20] Rickman et al. (real: Acta Mater. 168 (2019) 473), [26] King et al. volume/pages (real: 104 (2016) 172), [30] Varvenne et al. title (real: "Theory of strengthening in fcc high entropy alloys"), [36] Jensen et al. (paper is on a different alloy), [42]/[43] Maresca-Curtin edge/screw swapped with wrong pages; wrong titles in [9], [25], [33], [35], [40]. All plan citations (Hu 2021, Lee 2021, Borg 2020, Giles 2022, Meredig 2018, Ding 2019, Deb 2002 NSGA-II) verified by DOI.

## 1. Audit of the current manuscript

`project_files/Solomon_Ahedor_MetaForge_Research_Paper.docx` (22 pages). Problems ranked by how badly a reviewer would react.

### Must remove or correct (integrity)

| # | Where | Problem |
|---|---|---|
| A1 | Fig. 10 (VAR roadmap), panel (d) | A "10 x 10 Vickers grid, 588 +/- 14 kgf/mm2, validating sigma_y = 1.92 GPa" is shown as if measured. No alloy was ever made. This is fabricated experimental data and must go. Panel (c) XRD must be labelled "simulated". |
| A2 | Sec. 2.3, Fig. 5, Sec. 4.2 | The strength model is attributed to Varvenne-Luque-Curtin [30], but the code uses `sigma_0 = G/150`, `tau = 0.05 G delta^(2/3)`. That is an ad hoc formula with invented constants, not the Varvenne theory (which uses misfit volumes, elastic constants and an energy barrier with temperature and strain-rate dependence). Fig. 5 compares this formula with experiment as if it were the theory. |
| A3 | Table 1, Fig. 8, Sec. 3.1 | R2 = 0.994 for "yield strength" is a Random Forest re-learning the same formula (linear regression gets R2 = 0.99). The text says this shows "no overfitting" and "physical trends". It shows nothing about real strength. |
| A4 | Sec. 2.5 | The fitness function in the paper (w1 sigma/rho + w2 Omega - w3 delta) is not the one in the code (sigma/rho - 0.5 E). The paper calls it multi-objective and Pareto-optimal; the code optimises one scalar and never computes a Pareto front. |
| A5 | Sec. 2.5, Table 2 | "Strict 5-35 at.% bounds" are not enforced: clip-then-renormalise lets elements exceed 35% (Al 35.7% in the paper; V 53.7%, Cu 55.5%, Al 45.9% in the Oct 13 rerun). |
| A6 | Table 2, Sec. 3.3 | The refractory composition in the paper does not match the CIF in the repo (W18Ta17Mo11V5Nb3). Formation energies come from an RF trained on 50 CHGNet labels, not from relaxing the candidate. |
| A7 | Abstract, Sec. 2.1, Sec. 5.1 | "Tier 3 benchmarks formation energies / quantum DFT verification": no DFT was run. Must be phrased as a protocol only, or removed. |
| A8 | Sec. 3.4-3.6 | Corrosion ("self-healing Cr2O3", "antifouling"), ultralight (Li-Mg-Zn "single-phase FCC") and aerospace claims have no supporting calculation. The FCC template for Al-Ti-Sc-Zr-V and Al-Mg-Li-Ti-Zn is unjustified (mostly hcp/bcc elements); the +27.8% CHGNet volume change in the rerun shows the template lattice is wrong. |
| A9 | References | At least: [18] Rao et al. author list is wrong (real: Rao, Tung, Xie, Wei, Zhang, Ponge, Li, Raabe... Science 378, 78, 2022, doi 10.1126/science.abo4940). [5] "F. J. Lu, S. Y. Chen" and [29] look unreliable. Every reference must be checked against DOI. [17] Fieser et al., Materials 19 (2026) 3075 exists. |
| A10 | Para after Sec. 3.7 benchmark | Formatting bug from my Oct 12 edit: the "For both refractory alloys..." sentence landed inside an equation object and renders in math italics. Fix in Word. |
| A11 | Keywords | "Model Context Protocol" has nothing to do with the paper. |

### Weak but fixable (scientific quality)

- Four unrelated alloy families with no measured data for three of them dilutes the paper. Reviewers will ask why Li-Mg-Zn and a "marine" alloy appear in a refractory paper.
- No uncertainty on any prediction, no test of extrapolation, no comparison with existing models or with known alloys.
- "Tc = dH/dS = 210 K" mixes Miedema enthalpy, CHGNet formation energy and ideal entropy; physically not meaningful as a stability temperature.
- No ductility consideration, although room-temperature brittleness is the main barrier for W-Mo-rich RHEAs.

## 2. What the literature says reviewers will expect

- **Physics-based strength for BCC HEAs.** Maresca & Curtin, Acta Mater. 182 (2020) 144 and 235: edge dislocations control high-temperature strength in random BCC alloys; a parameter-free reduced model needs only composition, lattice parameter, elastic constants and element misfit volumes, and was used to screen >600,000 Mo-Nb-Ta-V-W compositions. Baruffi, Maresca & Curtin, MRS Commun. 12 (2022) 1111 (equations corrected Feb 2024) compares screw vs edge theory against many BCC alloys; data on Materials Cloud (doi 10.24435/materialscloud:p6-za). Lee et al., Nat. Commun. (2021) is the experimental counterpart.
- **Measured data.** MPEA dataset (Borg et al., Sci. Data 2020; already in use). Couzinie, Senkov, Miracle, Dirras, Data in Brief 21 (2018) 1622: 122 RHEAs with yield strength vs temperature. Gorsse et al., Data in Brief 21 (2018) 2664: 370 HEAs.
- **ML for RHEA strength.** Giles et al., npj Comput. Mater. (2022): ML forward model + optimisation for high-temperature yield strength with UQ from repeated k-fold CV. Several 2023-2025 papers on yield strength-plasticity trade-off with UQ (e.g. arXiv 2304.13932).
- **Design loops.** Rao et al., Science 378 (2022) 78 (active learning + DFT + CALPHAD + experiment, 17 alloys made). BIRDSHOT batch Bayesian optimisation for refractory alloys (arXiv 2405.07130, 2405.08900): true multi-objective Pareto design with hypervolume reporting.
- **Ductility proxies.** Hu et al., Acta Mater. 210 (2021) 116800: surface energy / unstable stacking fault energy (D parameter) for 106 bcc alloys with surrogate models; later work questions D but it remains the standard screening proxy. VEC-based ductility criteria are a cheaper alternative.
- **Universal ML potentials.** Elastic-property benchmark of MatterSim, MACE, SevenNet, CHGNet on ~11,000 MP structures (arXiv 2510.22999): CHGNet is the weakest for elastic constants, SevenNet the best. No benchmark on disordered HEAs was found: that is an open gap.
- **Honest evaluation.** Meredig et al., MSDE 2018 (leave-one-cluster-out CV); Durdy et al., Digital Discovery 2022 (kernelised LOCO-CV); Sutton et al. (domains of applicability).
- **In your folder.** Xiao et al. 2022 (Nb-Mo-Ta-W hardness thin-film library, HEA&M) gives composition-resolved hardness for exactly this system and is a ready validation set. Raabe, Mianroodi, Neugebauer, Nat. Comput. Sci. 3 (2023) 198 (physics-informed AI) is the framing reference. Han et al., Sci. Adv. 2024 (sustainability of HEMs) supports adding cost/criticality as an objective.

## 3. Proposed paper

**Scope:** refractory BCC MPEAs only (Mo-Nb-Ta-V-W, optionally + Ti, Cr, Hf, Zr). This is where measured data, physics theory and the UTC/UTK hypersonics pitch all line up. The other three families stay in the web app with coverage warnings, not in the paper.

**Working title:** "Physics-informed, uncertainty-aware design of refractory multi-principal element alloys: universal ML potentials feed a parameter-free strength theory, checked against measured data."

**Research questions (each gives a figure):**
1. How accurate are universal ML potentials (CHGNet, MACE-MP-0, MatterSim, SevenNet) for lattice parameters and elastic constants of disordered RHEAs? (Gap in the literature.)
2. Fed with those inputs, how well does the parameter-free Maresca-Curtin edge model predict measured yield strength from 25 to 1600 C, compared with a pure ML model and a hybrid?
3. Does the hybrid extrapolate better to unseen alloy families (LOCO-CV) and to alloys published after the training cut-off (temporal holdout)?
4. Under these validated models, what does the true Pareto front of specific high-temperature strength vs density vs a ductility proxy look like, and which 3-5 compositions should be made first, with prediction intervals?

**Why this is interesting:** it answers a question nobody has closed (do universal potentials make physics-based strength screening work without DFT?), it is fully reproducible on a laptop, and it turns the retraction story into a strength: every number is tested against experiment.

## 4. Work packages

Effort is in Mondays (one Monday = about 6 focused hours). Laptop CPU is enough for all of it.

### WP0. Clean the record (1 Monday)
- Remove A1 hardness panel; label XRD as simulated. Remove A2 misattribution (call the old formula what it is, or drop it). Remove Table 1 strength rows, Fig. 8(b-c). Fix A10, A11.
- Verify every reference by DOI; replace unverifiable ones.
- Keep this cleaned version as v1 so nothing false remains anywhere while v2 is built.
- **Done when:** a reviewer reading v1 finds no claim without a script and data file behind it.

### WP1. Fix the code (1 Monday)
- GA bounds: replace clip-and-renormalise with exact projection onto the bounded simplex (or rejection sampling); add a unit test that 10,000 children all satisfy 0.05 <= c_i <= 0.35 and sum to 1.
- Blueprint lattice from the rule-of-mixtures volume (not a fixed per-category a), so "volume change on relaxation" means something; structure type from VEC (BCC for VEC < 6.87) instead of a hard-coded template.
- Coverage penalty: GA may not leave the region where the strength model has data (distance to nearest training alloy in Magpie space, or element-count rule already in the web app).
- Replace the scalar fitness with NSGA-II (pymoo) and report the Pareto set and hypervolume.
- **Done when:** a rerun gives bounded compositions, sensible CHGNet volume changes (|dV| < 3%) and a Pareto front plot.

### WP2. Data (1 Monday)
- Merge MPEA (Borg 2020) + Couzinie 2018 RHEA compilation; keep test temperature, test type, processing state; deduplicate by composition + temperature + reference.
- Add the Xiao et al. 2022 Nb-Mo-Ta-W hardness library as an independent validation set (hardness, not yield strength: use as rank correlation only).
- Freeze a temporal holdout: all alloys from papers published after a cut-off year (e.g. 2021) are never used for training or model selection.
- **Done when:** `data/processed/rhea_strength.csv` with a data card (sources, counts, licence, holdout definition).

### WP3. Physics model (2 Mondays)
- Implement the Maresca-Curtin reduced edge model (zero-temperature strength and energy barrier from mu, nu, b and misfit volumes; thermal activation for T and strain rate; Taylor factor 3.06). Take equations from the papers, including the 2024 correction to Baruffi et al.
- First reproduce published numbers (MoNbTaW, MoNbTaVW curves; Baruffi Materials Cloud data) as unit tests before using it for anything new.
- **Done when:** the implementation matches published predictions within a few percent.

### WP4. Universal potential benchmark (2 Mondays)
- For the elements and about 15-20 RHEAs with measured lattice parameters / moduli: relax SQS (3 seeds) and compute C11, C12, C44 by finite strain with CHGNet, MACE-MP-0, MatterSim, SevenNet. Compare with experiment and published DFT.
- Misfit volumes per element from the relaxed SQS (Vegard slope), as the theory needs.
- Report the 3d-magnet artefact already found for CHGNet as a limitation, with the moment-collapse evidence.
- **Done when:** table + parity plots; one potential chosen for WP5 with a stated error.

### WP5. Strength models and honest validation (2 Mondays)
- Three models: (a) ML on measured data (current HistGradientBoosting, now with temperature); (b) physics only (WP3 fed by WP4); (c) hybrid: ML on the residual of the physics model, or physics prediction as a feature.
- Validation: grouped CV by alloy; LOCO-CV by alloy family; temporal holdout scored once at the end. Report MAE, R2, and coverage of 90% prediction intervals (quantile or ensemble UQ, calibration plot).
- **Done when:** one table and one figure that answer research questions 2 and 3, whichever model wins.

### WP6. Design (1-2 Mondays)
- NSGA-II over Mo-Nb-Ta-V-W(+Ti, Cr): maximise yield strength at 1000-1200 C / density, minimise density, constrain a ductility proxy (VEC-based, or a D-parameter surrogate from Hu et al. 2021 if time), penalise uncertainty (lower confidence bound) and stay inside the domain of applicability. Optional fourth objective: raw-material cost / criticality (Han et al. 2024).
- Retrospective test: does the pipeline, trained before the cut-off year, rank the post-cut-off strong alloys highly?
- Output 3-5 recommended compositions with predicted strength +/- interval at RT and 1000 C, density, phase prediction.
- **Done when:** Pareto plot, retrospective-test plot, recommendation table.

### WP7. Write v2 (2 Mondays)
- Structure: Introduction (gap: physics vs ML, uMLIP accuracy unknown for HEAs) / Data / Methods (potentials, theory, ML, UQ, design) / Results (RQ1-RQ4) / Discussion (limits: magnetism, SRO, oxidation, ductility proxy) / Data and code availability.
- About 8 figures, every one generated by a script in the repo from committed data.
- Target journals, realistic order: Computational Materials Science; Materials & Design; Integrating Materials and Manufacturing Innovation. Digital Discovery if the validation/benchmark story is the strongest part.
- Add an "Author contributions / AI use" statement as journals now require.

### WP8. Experimental validation (optional, not in the computational paper)
- Ask for collaboration, do not claim results: arc-melting + hardness of 2-3 recommended alloys through a lab (UTK Liaw group or MTU Sanders are natural fits; KNUST if arc melting and a microhardness tester are available).

## 5. Timeline

About 12 Mondays (mid-October 2026 to mid-January 2027), or faster on non-work evenings.

| Mondays | Work |
|---|---|
| 1 | WP0 + WP1 |
| 2 | WP2 |
| 3-4 | WP3 |
| 5-6 | WP4 |
| 7-8 | WP5 |
| 9-10 | WP6 |
| 11-12 | WP7 |

WP0 alone already makes the paper and pitch emails safe to share; it should happen before any more emails mention MetaForge.

## 6. Risks

- The Maresca-Curtin model may need inputs (misfit volumes, elastic constants) that the potentials get wrong for V- or Cr-rich alloys. Mitigation: WP4 measures that error first; fall back to experimental/DFT element data where needed.
- Couzinie data may be too small at high temperature for ML; the physics model is what makes high-temperature predictions credible, ML only corrects it.
- A hybrid that does not beat the physics model is still a publishable, honest result.

## 7. Decisions needed from Solomon

1. Scope: refractory only in the paper (recommended), or keep four families?
2. Target journal tier: Computational Materials Science (safer) or Digital Discovery (more visible, harder)?
3. Pursue WP8 collaboration now (helps the UTK/MTU applications) or after v2 exists?
4. Whether v1 (cleaned, WP0) should replace the current paper on the CV and GitHub immediately.
