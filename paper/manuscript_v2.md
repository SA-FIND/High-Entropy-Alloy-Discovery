# Physics-informed, uncertainty-aware prediction and design of refractory multi-principal element alloys: a parameter-free strength theory corrected by machine learning, and the limits of universal interatomic potentials as its inputs

Solomon Ahedor

Department of Materials Engineering, Kwame Nkrumah University of Science and Technology, Kumasi, Ghana. Correspondence: ahedorsolomon@gmail.com

<!-- Draft v2, 2026-10-17. Every number comes from committed files in results/. Remaining for the author: AI-use statement, co-authorship decision, abstract to be cut to the journal's word limit. -->

## Abstract

Refractory multi-principal element alloys (RMPEAs) retain high strength at temperatures where nickel superalloys soften, but their composition space is too large to explore experimentally. Parameter-free theories of solid-solution strengthening and machine-learning (ML) models trained on measured data each offer a route to screening, yet their accuracy outside the alloys used to build them is rarely tested. We compiled 277 yield-strength measurements from 25 to 1600 °C on 72 single-phase body-centred cubic (BCC) alloys of Ti, Zr, Hf, V, Nb, Ta, Cr, Mo and W from the open MPEA dataset, and compared three predictors under validation designed for discovery: cross-validation grouped by alloy, by alloy system (unseen chemistries), and a temporal holdout of 14 alloys first reported in 2019. The reduced edge-dislocation theory of Maresca and Curtin, with no fitted parameters, explains 45% of the variance (R² = 0.45, mean absolute error 261 MPa) but under-predicts strength systematically, most at room temperature. A gradient-boosted model trained only on composition descriptors reaches R² = 0.51. A hybrid that uses the theory as a baseline and learns only its residual is best on every split: R² = 0.67 for unseen alloys, 0.64 for unseen alloy systems and 0.65 (MAE 175 MPa) for the 2019 alloys, where the theory alone gives 0.10. Conformalized quantile regression gives 90% prediction intervals with 84–88% empirical coverage. We then asked whether universal machine-learning interatomic potentials (CHGNet, MACE-MP-0, SevenNet-0) can replace the experimental elemental inputs of the theory. All three reproduce bcc lattice parameters within 1.7%, but underestimate shear elastic constants of the group V–VI elements by factors of 1.5 to 60; used as inputs, they degrade the strength predictions (R² from 0.45 to between −0.6 and −4.0). The softening persists in alloy supercells (R² = −1.05 with MACE-MP-0 alloy constants). Their atomic volumes alone are usable but do not improve on experimental values. Finally, a multi-objective search traded the lower 90% bound of specific strength at 1000 °C against valence electron concentration, a ductility proxy. Without a phase constraint, every Pareto alloy combined Zr with W, a Laves-forming pair never reported in a single-phase alloy. A data-derived pairwise phase rule (82% of single-phase and 77% of multiphase alloys correctly classified in leave-one-out tests) moves the front to Ti–Zr–Hf–V–Mo alloys predicted to match the best measured specific strength at 1000 °C at 24% lower density. Because Ti–Zr–V alloys soften abruptly above 600 °C, a behaviour the model over-predicts six- to seven-fold when such alloys are unseen, a high-temperature test of one candidate is the most informative next experiment. All data, code and models are openly available.

**Keywords:** refractory high-entropy alloys; solid-solution strengthening; edge dislocations; machine learning; uncertainty quantification; universal interatomic potentials

## 1. Introduction

Multi-principal element alloys (MPEAs), introduced by Yeh et al. [1] and Cantor et al. [2], replace a single solvent with several elements in comparable amounts. The refractory subset based on Ti, Zr, Hf, V, Nb, Ta, Cr, Mo and W includes single-phase BCC alloys such as Nb25Mo25Ta25W25 and V20Nb20Mo20Ta20W20 that keep yield strengths of 400–500 MPa at 1600 °C [3,4], well above the useful range of nickel superalloys [5,6]. Because the number of possible compositions grows combinatorially, computational screening is attractive, and two families of approaches have emerged.

The first is mechanistic. Maresca and Curtin showed that in random BCC alloys with sufficient solute misfit, edge dislocations rather than screw dislocations can control strength at high temperature, and derived a reduced, parameter-free model that needs only composition, lattice parameter, elastic constants and solute misfit volumes [7,8]. Lee et al. confirmed edge-dislocation activity experimentally and used the model to screen more than ten million compositions [9]; Baruffi et al. compared screw and edge theories across 63 BCC alloys [10]. The second family is data-driven: ML models trained on compiled mechanical-property data [11–13], sometimes embedded in active-learning or optimisation loops [14,15].

Both approaches leave open questions that matter for alloy design. How well does the parameter-free theory predict measured strengths across the full refractory composition space, including Ti-, Zr- and Hf-rich alloys? Do ML models extrapolate to chemistries and to alloys not yet made, which is what design requires, rather than only interpolate under random cross-validation [16,17]? And since the theory's inputs (elastic constants and atomic volumes of each element in the bcc structure) are unavailable or ill-defined for elements that are not bcc at room temperature, can universal machine-learning interatomic potentials (uMLIPs) [18–20], trained on hundreds of thousands of density functional theory calculations, supply them?

This work addresses these questions with one dataset and one validation protocol. We (i) benchmark the parameter-free edge theory against 277 measurements; (ii) compare it with a purely data-driven model and with a hybrid that learns only the theory's residual, under validation splits that test extrapolation to new alloys, new alloy systems and alloys published later than the training data; (iii) attach calibrated prediction intervals; (iv) test three uMLIPs as sources of the theory's inputs; and (v) use the best predictor in a constrained multi-objective search. We report negative as well as positive results, since both are needed to decide how far such tools can be trusted.

## 2. Data and methods

### 2.1 Data

Yield strengths were taken from the open MPEA dataset of Borg et al. [11] (Apache 2.0 licence). We kept entries whose alloys contain only Ti, Zr, Hf, V, Nb, Ta, Cr, Mo and W and whose microstructure is reported as single-phase BCC, giving 277 measurements on 72 alloys between 20 and 1600 °C (246 in compression, 31 in tension; Fig. 1). Duplicate rows (same composition, temperature, value, test and source) were removed. Alloys whose first appearance in the dataset is 2019, the most recent year, form a temporal holdout of 14 alloys and 60 measurements; they were not used for model training or selection and were scored once, after all modelling choices were fixed. The remaining 58 alloys (217 measurements) form the training set. The processed file and the script that builds it are provided.

### 2.2 Parameter-free edge-dislocation model

We use the reduced model of Maresca and Curtin [7] in the form given by Lee et al. [9]. The zero-temperature yield stress and the zero-stress energy barrier are

τ_y0 = A_σ α^(−1/3) μ [(1+ν)/(1−ν)]^(4/3) [Σ_n c_n ΔV_n² / b⁶]^(2/3),
ΔE_b = A_E α^(1/3) μ b³ [(1+ν)/(1−ν)]^(2/3) [Σ_n c_n ΔV_n² / b⁶]^(1/3),

with A_σ = 0.040, A_E = 2.00, line-tension coefficient α = 1/12, Burgers vector b = (√3/2) a, isotropic shear modulus μ = [C44 (C11 − C12)/2]^(1/2) and Poisson ratio ν = (3B − 2μ)/[2(3B + μ)] with B = (C11 + 2C12)/3. The uniaxial yield strength at temperature T and strain rate ε̇ is σ_y = 3.067 τ_y, with τ_y = τ_y0 [1 − (kT/ΔE_b · ln(ε̇0/ε̇))^(2/3)] when τ_y/τ_y0 ≥ 0.5 and τ_y = τ_y0 exp[−(1/0.55)(kT/ΔE_b) ln(ε̇0/ε̇)] otherwise, ε̇0 = 10⁴ s⁻¹ [7,9]. We take ε̇ = 10⁻³ s⁻¹ since strain rates are not recorded in the dataset. Alloy inputs follow the screening route of Lee et al. [9]: Vegard's law for the mean atomic volume, misfit volumes ΔV_n = V_n − V̄, and rule-of-mixtures elastic constants, with the elemental bcc values of their Supplementary Table 2 (Ti, Zr and Hf from high-temperature bcc data). The implementation reproduces published values: energy barriers of 2.21 and 2.33 eV for MoNbTaW and MoNbTaVW with the DFT inputs of ref. [7], against 2.2 and 2.5 eV from the full theory, and μ = 47.8 GPa, ν = 0.365 for NbTaTiV as in ref. [9]. We note that the two-regime form as published is discontinuous: at the switch, strength steps up by about 5%.

### 2.3 Machine-learning and hybrid models

Each measurement is described by the 132 Magpie composition descriptors [21,22], the test temperature and the test mode. The ML model is a histogram gradient-boosted regressor (scikit-learn [23]; depth 3, learning rate 0.05, 300 iterations, L2 regularisation 1.0, at least 10 samples per leaf). The hybrid model is the edge-theory prediction plus the same regressor trained on the residual (measured − theory). A two-parameter linear rescaling of the theory serves as a minimal calibration baseline. Hyperparameters were fixed before any evaluation and not tuned.

### 2.4 Validation and uncertainty

Three splits were used on the training alloys: five-fold cross-validation grouped by alloy, so that no alloy appears in both training and test folds; five-fold cross-validation grouped by alloy system (the set of elements present), which tests extrapolation to unseen chemistries [16,17]; and the temporal holdout described above. Ninety-percent prediction intervals were obtained by conformalized quantile regression [24]: 5% and 95% quantile regressors on the hybrid residual, widened by the 90% quantile of out-of-fold conformity scores computed inside the training data (grouped by alloy).

### 2.5 Universal interatomic potentials

CHGNet 0.3.0 [18], MACE-MP-0 "medium" [19] and SevenNet-0 [20] were used through ASE [25] on CPU. MatterSim could not be installed on the Windows workstation used and was omitted. For the nine elements in the bcc structure, the lattice parameter was obtained from a quadratic fit of the energy around its minimum, and cubic elastic constants from the curvature of the energy under hydrostatic, volume-conserving tetragonal and shear strains of up to ±1% (five points each). An energy route was used because CHGNet stresses were inconsistent with its energies for these cells (bulk modulus of bcc W 125 GPa from stress against 271 GPa from energy), whereas MACE agreed between the two routes. Values were identical in 2-atom and 54-atom cells. For alloy-level inputs, 54-atom special quasirandom structures (SQS) [26] were generated by Monte Carlo minimisation of first- and second-shell Warren–Cowley parameters, relaxed in positions and cell (FIRE, forces below 0.02 eV/Å), and their relaxed-ion elastic constants computed by the same energy route. Because SQS calculations are costly on CPU, they were run with MACE-MP-0, the most accurate potential at the element level, for 15 training alloys: MoNbTaW and MoNbTaVW, whose lattice parameters are known [3], and the most frequently measured alloy of each of the other training alloy systems. Misfit volumes for this route were taken from the potential's own elemental bcc volumes.

### 2.6 Multi-objective design

The search space is Ti–Zr–Hf–V–Nb–Ta–Cr–Mo–W with four to six elements, each between 5 and 35 at.%, enforced exactly by Euclidean projection onto the bounded simplex. NSGA-II [27] (pymoo [28]; population 80, 60 generations) maximised the lower 90% conformal bound of the hybrid yield strength at 1000 °C divided by rule-of-mixtures density, and minimised valence electron concentration (VEC), a ductility proxy: Sheikh et al. found ductile refractory MPEAs at VEC ≤ 4.4 and brittle ones at VEC ≥ 4.6 [29]. Constraints were a volume misfit δ = (Σ c_n ΔV_n²)^(1/2)/(3V̄) ≤ 6.6% and a domain-of-applicability limit: the standardised Magpie distance to the nearest measured alloy could not exceed the 95th percentile of nearest-neighbour distances among the measured alloys. For design, the hybrid model was refitted on all 72 alloys after validation.

The strength model assumes a single BCC phase, so the search must also stay where that assumption is plausible. Simple descriptors do not predict phase reliably in these data: for the 97 refractory alloys of the nine elements with a reported microstructure (67 single-phase BCC, 30 multiphase, counting an alloy as single-phase only if every report of it says so), logistic regression on VEC, δ and a Laves-pair index reached a cross-validated ROC AUC of only 0.56–0.63, as reported phases depend on processing. We therefore used a transparent pairwise rule: a candidate may not contain any element pair at a concentration product c_i c_j larger than in some alloy reported as single-phase BCC. Pairs never observed together in a single-phase alloy (Zr–Cr, Zr–W, Hf–Cr, Hf–W, V–Cr, Ta–Cr and Cr–W) are therefore excluded. In leave-one-alloy-out validation the rule accepted 55 of the 67 single-phase alloys (82%) and rejected 23 of the 30 multiphase ones (77%). It is a data-domain constraint rather than a thermodynamic model, and it cannot detect phase separation in chemistries that are within the caps, such as HfNbTaTiZr, reported in the dataset both as single-phase BCC and as BCC + HCP. A third constraint concerns temperature. Ti–Zr–V alloys soften abruptly above about 600 °C (NbTiVZr: 834 MPa at 600 °C, 187 MPa at 800 °C and 58 MPa at 1000 °C), presumably as diffusion-controlled mechanisms that the glide-based theory does not describe take over. When these alloys are held out by alloy system, the hybrid over-predicts their 1000 °C strength six- to seven-fold (339–496 MPa). The temperature at which a chemistry softens can only be learned from alloys tested near it, and only 20 of the 72 alloys have data at or above 900 °C. We therefore required the standardised Magpie distance from a candidate to the nearest alloy tested at ≥ 900 °C to satisfy the same 95th-percentile limit. This constraint is a coverage argument; it could not be validated retrospectively, because almost all high-temperature test rows belong to the hot-tested alloys themselves. Finally, candidates whose predicted strength at 1000 °C exceeds the highest strength measured at or above 900 °C in the dataset (927 MPa) are flagged as extrapolating in the target as well as in composition.

## 3. Results

### 3.1 CHGNet relaxations reproduce refractory lattice parameters

Relaxing atoms and cell of three SQS realisations each, CHGNet gives a = 3.240 Å for NbMoTaW and 3.196 Å for NbMoTaWV, 0.83% and 0.41% above the measured 3.2134 and 3.1832 Å [3], with seed-to-seed scatter below 0.004 Å (Fig. 2). The relaxed volumes lie 0.5–1.0% above Vegard's law, so these refractory alloys show no volume contraction relative to the rule of mixtures. For the magnetic 3d alloys CoCrFeNi and CoCrFeMnNi, CHGNet gives volumes 5–6% below Vegard but lattice parameters 1.5–2.4% below experiment, while the predicted Fe, Co and Mn moments fall during relaxation; this is consistent with a magnetovolume artefact of the potential rather than a physical effect.

### 3.2 The parameter-free theory captures trends but under-predicts

Without any fitting, the edge theory explains 45% of the variance in the 217 training measurements (R² = 0.447, MAE 261 MPa; Fig. 3a). It under-predicts on average by 109 MPa; the bias is largest at room temperature (−142 MPa over 131 measurements) and decreases at high temperature (−38 MPa between 900 and 1200 °C; Fig. 5). Room-temperature strength in BCC alloys also has screw-dislocation and interstitial contributions [8,9,10] that the edge model does not include; Lee et al. attributed a similar shortfall for NbTaTiV to 1.6 at.% O + N [9]. On the 2019 alloys the bias grows to −212 MPa and R² falls to 0.10.

### 3.3 Learning the theory's residual is better than learning strength directly

Table 1 and Fig. 4 compare the four predictors. The purely data-driven model is slightly better than the theory for unseen alloys (R² 0.51) and unseen systems (0.52), but worse than a two-parameter rescaling of the theory on the 2019 alloys (0.46 against 0.59). The hybrid is best on all three splits: R² = 0.666 (MAE 186 MPa) for unseen alloys, 0.638 (217 MPa) for unseen alloy systems and 0.649 (175 MPa) for the 2019 alloys. Its 90% conformal intervals contain 84%, 88% and 88% of the measurements on the three splits, close to the nominal level; exact coverage is not guaranteed because measurements of the same alloy at different temperatures are not exchangeable.

Table 1. Yield-strength prediction for single-phase BCC refractory alloys.

| Model | Unseen alloys R² / MAE (MPa) | Unseen systems R² / MAE | 2019 alloys R² / MAE |
|---|---|---|---|
| Edge theory (no fitting) | 0.447 / 261 | 0.447 / 261 | 0.098 / 297 |
| Edge theory, linear rescaling | 0.538 / 225 | 0.426 / 254 | 0.591 / 194 |
| ML (Magpie + T) | 0.506 / 224 | 0.519 / 245 | 0.455 / 225 |
| Hybrid (theory + ML residual) | **0.666 / 186** | **0.638 / 217** | **0.649 / 175** |

### 3.4 Universal potentials give good volumes but soft shear moduli

All three potentials reproduce the bcc lattice parameters of the nine elements to within 1.7% (Fig. 6a). Their elastic constants are much less accurate (Fig. 6b). For W, the reference C′ = (C11 − C12)/2 is 164 GPa; CHGNet, MACE and SevenNet give 37, 53 and 51 GPa. C44 of Mo is 125 GPa in the reference and 41, 38 and 2.5 GPa in the three potentials. Several values are negative (C44 of Cr in CHGNet; C′ of Ti in CHGNet and of Hf in SevenNet), i.e. the potentials predict a mechanically unstable bcc lattice. This systematic softening is consistent with the analysis of Deng et al. [30]. Used as inputs to the edge theory, the potentials' elemental values worsen predictions from R² = 0.45 to −0.60 (MACE), −2.05 (CHGNet) and −4.03 (SevenNet), with strengths under-predicted by 520–970 MPa on average (Table 2). Combining experimental elastic constants with each potential's atomic volumes gives R² = 0.32–0.36: usable, but no better than experimental volumes.

The softening is not an artefact of unstable elemental bcc structures. For the 15 alloys computed as relaxed SQS supercells with MACE-MP-0, lattice parameters agree with Vegard's law to within 0.6% (MoNbTaW: 3.247 Å, 1.0% above the measured 3.2134 Å [3]), but C′ is 38–79% (mean 52%) and C44 49–83% (mean 63%) of the rule-of-mixtures values from experimental elemental constants. The deficit is largest for the Mo- and W-rich alloys (C′ of MoNbTaW: 39 GPa against 104 GPa). Used as inputs to the theory, these alloy-level constants give R² = −1.05 on the 146 measurements of the 15 alloys, against 0.55 for experimental elemental inputs on the same measurements (Table 2).

Table 2. Edge-theory predictions with different inputs (217 training measurements).

| Inputs | R² | MAE (MPa) | Bias (MPa) |
|---|---|---|---|
| Experimental elemental values [9] | 0.447 | 261 | −109 |
| CHGNet elemental a, C_ij | −2.05 | 746 | −738 |
| MACE-MP-0 elemental a, C_ij | −0.60 | 527 | −524 |
| SevenNet-0 elemental a, C_ij | −4.03 | 967 | −967 |
| CHGNet volumes + experimental C_ij | 0.318 | 299 | −107 |
| MACE-MP-0 volumes + experimental C_ij | 0.364 | 279 | −67 |
| SevenNet-0 volumes + experimental C_ij | 0.364 | 289 | −116 |
| MACE-MP-0 alloy SQS a, C_ij (15 alloys, 146 measurements) | −1.05 | 540 | −534 |
| Experimental elemental values, same 146 measurements | 0.545 | 211 | −119 |

### 3.5 Design: each constraint changes the answer

Table 3 shows how the Pareto set changes as constraints are added, and Fig. 7 the resulting fronts. With only the composition domain and the size-misfit limit, the search returned Ti–Zr alloys with W, Mo and V. They are predicted to reach up to 1550 MPa at 1000 °C, and their best lower bound of specific strength is 0.150 GPa cm³ g⁻¹. All 80 violate the phase rule, 73 through the Zr–W pair, and 30 are predicted to be stronger at 1000 °C than any alloy in the dataset at or above 900 °C (927 MPa). In other words, the optimiser found the combination of large misfit and high shear modulus that the edge theory rewards, in a part of composition space where single-phase BCC alloys have not been reported.

With the phase rule, the front moves to Ti–Zr–Hf alloys with Mo and V, or with Ta and Nb at the low-VEC end, and no prediction exceeds the measured range. The high-temperature domain constraint changed the front only slightly: the Ti–Zr–Hf–V–Mo candidates lie at Magpie distances of 6.5–7 from HfMo0.5NbTiV0.5 and HfMoNbTiZr (both tested at 1000 °C; median nearest-neighbour distance between hot-tested alloys 5.3), so they satisfy it.

Table 3. Pareto sets (80 alloys each) under increasing constraints.

| Constraints | Elements (count in 80) | Phase-rule violations | Alloys with V and Zr | Max predicted σy at 1000 °C (MPa) | Best lower-bound σy/ρ at 1000 °C (GPa cm³ g⁻¹) |
|---|---|---|---|---|---|
| Composition domain, δ ≤ 6.6% | Ti 80, Zr 80, W 73, Hf 52, Mo 51, V 48, Ta 7 | 80 | 48 | 1550 (30 above 927) | 0.150 |
| + pairwise phase rule | Ti 80, Zr 80, Hf 80, Mo 62, V 61, Ta 18 | 0 | 61 | 865 | 0.098 |
| + high-temperature domain (final) | Ti 80, Zr 80, Hf 80, Mo 64, V 56, Ta 17, Nb 16 | 0 | 56 | 934 (1 above 927) | 0.098 |

The final front has three regimes. Below VEC 4.25, Ti–Zr–Hf alloys with 5–25% Nb or Ta have negative lower bounds at 1000 °C: the model cannot exclude near-zero strength. This is consistent with the weakness of HfNbTiZr (154 MPa) and HfNbTaTiZr (295 MPa) at that temperature, and these compositions lie close to HfTa0.4–0.6TiZr, which is reported as BCC + HCP. At VEC ≤ 4.4, the ductile range of Sheikh et al. [29], the best candidate is Ti30.7Zr26.6Hf14.9V15.6Mo12.2: predicted 1555 MPa at 25 °C and 488 MPa at 1000 °C (90% interval 232–989 MPa), with density 7.49 g cm⁻³. Its lower bound of specific strength, 0.031 GPa cm³ g⁻¹, is no better than the measured values of NbTa0.3TiZr (0.037) and HfNbTaTiZr (0.030), so the model cannot promise a gain in the ductile range. Above VEC 4.5, Ti–Zr–Hf–V–Mo alloys reach lower bounds of 0.08–0.10 GPa cm³ g⁻¹. For example, Ti22.6Zr22.4Hf13.1V21.3Mo20.6 (VEC 4.62, 7.77 g cm⁻³) is predicted at 791 MPa at 1000 °C (670–1423 MPa). That would match the measured specific strength of HfMoTaTiZr (855 MPa, 0.084 GPa cm³ g⁻¹) at 24% lower density.

These high-strength candidates carry the risk identified in Section 2.6: 56 of the 80 alloys contain both V and Zr, the chemistry in which measured strength collapses between 600 and 1000 °C and in which the hybrid over-predicted unseen alloys six- to seven-fold. Their phase stability also rests on a data-domain rule, not on thermodynamics. We therefore regard them as hypotheses that a single experiment can test, not as discoveries. Compression tests between 600 and 1000 °C on one Ti–Zr–Hf–V–Mo candidate, after homogenisation and with phase analysis, would show whether Mo and Hf suppress the softening of Ti–Zr–V alloys. Either outcome would be informative, because the test probes the model where its training data are thinnest.

## 4. Discussion

**Why the hybrid works.** The edge theory supplies the temperature dependence of strength from physical inputs, which a composition-only regressor must otherwise learn from sparse high-temperature data. The ML part then corrects a composition-dependent offset, mainly the room-temperature shortfall that comes from screw-dislocation, interstitial and processing contributions the theory omits [8–10]. This division of labour explains why the hybrid extrapolates to unseen alloy systems (R² 0.64) and to alloys published later (0.65) better than either component alone: neither the theory (0.45, 0.10) nor the ML model (0.52, 0.46) handles both the temperature and the composition dependence well. The gain is not universal, however. Where the theory's mechanism stops applying, as in the high-temperature collapse of Ti–Zr–V alloys, the residual is large and chemistry-specific, and a model that has not seen the chemistry at temperature cannot learn it. Physics-informed ML inherits the domain of its physics as well as that of its data.

**Universal potentials as inputs.** The potentials tested give accurate volumes and inaccurate shear moduli, at both the element and the alloy level. Because the edge theory scales linearly with μ (and through ΔE_b, temperature dependence also depends on μ), a factor-of-two softening of C′ and C44 propagates directly into the strength. The softening is systematic and consistent with the analysis of Deng et al. [30], who traced it to the under-sampling of high-energy configurations in the training data and showed that fine-tuning on a small number of additional calculations can correct it. Until such corrections are standard, uMLIPs can supply misfit volumes but not elastic constants for strength theories. A practical route is therefore to combine experimental or DFT elastic constants with uMLIP volumes and to reserve the potentials for quantities, such as lattice parameters and relaxed structures, that they predict well.

**Limitations.** The dataset is small (72 alloys) and concentrated in a few systems: 20 alloys have been tested at or above 900 °C. Strain rates, interstitial contents and heat treatments are not recorded consistently, which limits both the theory and the ML. Phase stability is handled by a data-domain rule that classifies about 80% of alloys correctly in leave-one-out tests but is no substitute for CALPHAD or diffusion-couple data, and it cannot detect phase separation inside the domain (HfNbTaTiZr). VEC is a coarse ductility proxy. The conformal intervals are calibrated on alloys like the training data; their nominal coverage need not hold for compositions far from it, and the strength-at-temperature collapse described above is an example. Finally, the design objective uses a lower bound at a single temperature; creep and oxidation, which often decide the usefulness of refractory alloys, are not considered.

## 5. Conclusions

1. On 277 measurements of 72 single-phase BCC refractory alloys, the parameter-free edge-dislocation theory of Maresca and Curtin explains 45% of the variance in yield strength without fitting, but under-predicts by about 110 MPa on average and fails for alloys published later (R² 0.10).
2. A hybrid model that learns only the theory's residual from composition descriptors is the best predictor under every validation scheme designed for discovery: R² 0.67 for unseen alloys, 0.64 for unseen alloy systems and 0.65 for a temporal holdout, with 90% conformal intervals covering 84–88% of measurements.
3. Universal interatomic potentials (CHGNet, MACE-MP-0, SevenNet-0) reproduce bcc lattice parameters within 1.7% but underestimate shear elastic constants by factors of 1.5–60 for elements and 1.3–2.6 for alloys. Used as inputs, they make strength predictions much worse than experimental inputs; only their atomic volumes are usable.
4. In design, each guard against extrapolation changes the answer. Without a phase rule, all candidates combined Laves-forming pairs never reported in single-phase alloys; with it, the predicted gains shrink to a 24% density reduction at the specific strength of the best measured alloys. These Ti–Zr–Hf–V–Mo alloys lie in a chemistry known to soften above 600 °C, and testing one of them at temperature is the most informative next experiment.

## Data and code availability

All data, code, trained models and the scripts that generate every number and figure are available at https://github.com/SA-FIND/High-Entropy-Alloy-Discovery (MIT licence). The processed dataset is derived from the MPEA dataset [11] under its Apache 2.0 licence.

## Use of AI tools

[State plainly which AI assistance was used for code and text, and that all results were checked by the author, as the target journal requires.]

## References

[1] J.-W. Yeh, S.-K. Chen, S.-J. Lin, J.-Y. Gan, T.-S. Chin, T.-T. Shun, C.-H. Tsau, S.-Y. Chang, Nanostructured high-entropy alloys with multiple principal elements: novel alloy design concepts and outcomes, Adv. Eng. Mater. 6 (2004) 299–303. https://doi.org/10.1002/adem.200300567
[2] B. Cantor, I.T.H. Chang, P. Knight, A.J.B. Vincent, Microstructural development in equiatomic multicomponent alloys, Mater. Sci. Eng. A 375–377 (2004) 213–218. https://doi.org/10.1016/j.msea.2003.10.257
[3] O.N. Senkov, G.B. Wilks, D.B. Miracle, C.P. Chuang, P.K. Liaw, Refractory high-entropy alloys, Intermetallics 18 (2010) 1758–1765. https://doi.org/10.1016/j.intermet.2010.05.014
[4] O.N. Senkov, G.B. Wilks, J.M. Scott, D.B. Miracle, Mechanical properties of Nb25Mo25Ta25W25 and V20Nb20Mo20Ta20W20 refractory high entropy alloys, Intermetallics 19 (2011) 698–706. https://doi.org/10.1016/j.intermet.2011.01.004
[5] O.N. Senkov, D.B. Miracle, K.J. Chaput, J.-P. Couzinie, Development and exploration of refractory high entropy alloys—a review, J. Mater. Res. 33 (2018) 3092–3128. https://doi.org/10.1557/jmr.2018.153
[6] E.P. George, D. Raabe, R.O. Ritchie, High-entropy alloys, Nat. Rev. Mater. 4 (2019) 515–534. https://doi.org/10.1038/s41578-019-0121-4
[7] F. Maresca, W.A. Curtin, Mechanistic origin of high strength in refractory BCC high entropy alloys up to 1900K, Acta Mater. 182 (2020) 235–249. https://doi.org/10.1016/j.actamat.2019.10.015
[8] F. Maresca, W.A. Curtin, Theory of screw dislocation strengthening in random BCC alloys from dilute to "High-Entropy" alloys, Acta Mater. 182 (2020) 144–162. https://doi.org/10.1016/j.actamat.2019.10.007
[9] C. Lee, F. Maresca, R. Feng, Y. Chou, T. Ungar, et al., Strength can be controlled by edge dislocations in refractory high-entropy alloys, Nat. Commun. 12 (2021) 5474. https://doi.org/10.1038/s41467-021-25807-w
[10] C. Baruffi, F. Maresca, W.A. Curtin, Screw vs. edge dislocation strengthening in body-centered-cubic high entropy alloys and implications for guided alloy design, MRS Commun. 12 (2022) 1111–1118. https://doi.org/10.1557/s43579-022-00278-2 ; correction: MRS Commun. 14 (2024) 130. https://doi.org/10.1557/s43579-023-00507-2
[11] C.K.H. Borg, C. Frey, J. Moh, T.M. Pollock, S. Gorsse, et al., Expanded dataset of mechanical properties and observed phases of multi-principal element alloys, Sci. Data 7 (2020) 430. https://doi.org/10.1038/s41597-020-00768-9
[12] J.-P. Couzinié, O.N. Senkov, D.B. Miracle, G. Dirras, Comprehensive data compilation on the mechanical properties of refractory high-entropy alloys, Data Brief 21 (2018) 1622–1641. https://doi.org/10.1016/j.dib.2018.10.071
[13] B. Giles, D. Sengupta, S. Broderick, K. Rajan, Machine-learning-based intelligent framework for discovering refractory high-entropy alloys with improved high-temperature yield strength, npj Comput. Mater. 8 (2022) 235. https://doi.org/10.1038/s41524-022-00926-0
[14] Z. Rao, P.-Y. Tung, R. Xie, Y. Wei, H. Zhang, et al., Machine learning–enabled high-entropy alloy discovery, Science 378 (2022) 78–85. https://doi.org/10.1126/science.abo4940
[15] C. Wen, Y. Zhang, C. Wang, D. Xue, Y. Bai, S. Antonov, L. Dai, T. Lookman, Y. Su, Machine learning assisted design of high entropy alloys with desired property, Acta Mater. 170 (2019) 109–117. https://doi.org/10.1016/j.actamat.2019.03.010
[16] B. Meredig, E. Antono, C. Church, M. Hutchinson, J. Ling, et al., Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery, Mol. Syst. Des. Eng. 3 (2018) 819–825. https://doi.org/10.1039/c8me00012c
[17] S. Durdy, M.W. Gaultois, V.V. Gusev, D. Bollegala, M.J. Rosseinsky, Random projections and kernelised leave one cluster out cross validation: universal baselines and evaluation tools for supervised machine learning of material properties, Digital Discovery 1 (2022) 763–778. https://doi.org/10.1039/d2dd00039c
[18] B. Deng, P. Zhong, K. Jun, J. Riebesell, K. Han, C.J. Bartel, G. Ceder, CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling, Nat. Mach. Intell. 5 (2023) 1031–1041. https://doi.org/10.1038/s42256-023-00716-3
[19] I. Batatia, P. Benner, Y. Chiang, et al., A foundation model for atomistic materials chemistry, arXiv:2401.00096 (2023).
[20] Y. Park, J. Kim, S. Hwang, S. Han, Scalable parallel algorithm for graph neural network interatomic potentials in molecular dynamics simulations, J. Chem. Theory Comput. 20 (2024) 4857–4868. https://doi.org/10.1021/acs.jctc.4c00190
[21] L. Ward, A. Agrawal, A. Choudhary, C. Wolverton, A general-purpose machine learning framework for predicting properties of inorganic materials, npj Comput. Mater. 2 (2016) 16028. https://doi.org/10.1038/npjcompumats.2016.28
[22] L. Ward, A. Dunn, A. Faghaninia, N.E.R. Zimmermann, S. Bajaj, et al., Matminer: an open source toolkit for materials data mining, Comput. Mater. Sci. 152 (2018) 60–69. https://doi.org/10.1016/j.commatsci.2018.05.018
[23] F. Pedregosa, G. Varoquaux, A. Gramfort, et al., Scikit-learn: machine learning in Python, J. Mach. Learn. Res. 12 (2011) 2825–2830.
[24] Y. Romano, E. Patterson, E.J. Candès, Conformalized quantile regression, Adv. Neural Inf. Process. Syst. 32 (2019); arXiv:1905.03222.
[25] A. Hjorth Larsen, J. Jørgen Mortensen, J. Blomqvist, I.E. Castelli, R. Christensen, et al., The atomic simulation environment—a Python library for working with atoms, J. Phys.: Condens. Matter 29 (2017) 273002. https://doi.org/10.1088/1361-648X/aa680e
[26] A. Zunger, S.-H. Wei, L.G. Ferreira, J.E. Bernard, Special quasirandom structures, Phys. Rev. Lett. 65 (1990) 353–356. https://doi.org/10.1103/PhysRevLett.65.353
[27] K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, A fast and elitist multiobjective genetic algorithm: NSGA-II, IEEE Trans. Evol. Comput. 6 (2002) 182–197. https://doi.org/10.1109/4235.996017
[28] J. Blank, K. Deb, Pymoo: multi-objective optimization in Python, IEEE Access 8 (2020) 89497–89509. https://doi.org/10.1109/ACCESS.2020.2990567
[29] S. Sheikh, S. Shafeie, Q. Hu, J. Ahlström, C. Persson, J. Veselý, et al., Alloy design for intrinsically ductile refractory high-entropy alloys, J. Appl. Phys. 120 (2016) 164902. https://doi.org/10.1063/1.4966659
[30] B. Deng, Y. Choi, P. Zhong, J. Riebesell, S. Anand, Z. Li, et al., Systematic softening in universal machine learning interatomic potentials, npj Comput. Mater. 11 (2025) 9. https://doi.org/10.1038/s41524-024-01500-6
