"""
MetaForge: ML-Accelerated High Entropy Alloy Discovery

Stages:
  1. Combinatorial generation + stability filtering (Zhang delta, Guo VEC)
  2. Magpie featurization (132 descriptors via Matminer)
  3. Random Forest surrogate training with 5-fold CV
  4. GA inverse design maximizing specific strength
  5. SQS supercell generation + CHGNet relaxation

Property model note:
  Density and strength targets are RoM analytical proxies, not experimental
  or DFT values. LinearRegression sanity check confirms these are recoverable
  from Magpie features. See RESEARCH_PROPOSAL.md for DFT validation roadmap.
"""

# Imports
import os
import warnings
warnings.filterwarnings('ignore')
import itertools
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pymatgen.core import Lattice, Structure, Species
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

# HEA categories with constituent elements and VEC values
alloy_categories = {
    "Refractory":  {'W': 6, 'Mo': 6, 'Ta': 5, 'Nb': 5, 'V': 5},
    "Corrosion":   {'Co': 9, 'Cr': 6, 'Fe': 8, 'Ni': 10, 'Cu': 11},
    "Lightweight": {'Al': 3, 'Mg': 2, 'Li': 1, 'Ti': 4, 'Zn': 12},
    "Aerospace":   {'Al': 3, 'Ti': 4, 'Sc': 3, 'Zr': 4, 'V': 5}
}

# Elemental shear moduli (GPa) for RoM baseline
shear_moduli = {
    'Al': 26, 'Ti': 44, 'Sc': 29, 'Zr': 33, 'V': 47,
    'Mg': 17, 'Li': 4.2, 'Zn': 43, 'W': 161, 'Mo': 126,
    'Ta': 69, 'Nb': 38, 'Co': 82, 'Cr': 115, 'Fe': 82,
    'Ni': 76, 'Cu': 48
}

# Elemental properties — harvested via MP API in HEA.ipynb, stored as constants
master_props = {
    "Refractory": {
        "W": {"r": 1.37, "vec": 6, "density": 19.25},
        "Mo": {"r": 1.39, "vec": 6, "density": 10.28},
        "Ta": {"r": 1.43, "vec": 5, "density": 16.69},
        "Nb": {"r": 1.43, "vec": 5, "density": 8.57},
        "V": {"r": 1.31, "vec": 5, "density": 6.11}
    },
    "Corrosion": {
        "Co": {"r": 1.25, "vec": 9, "density": 8.9},
        "Cr": {"r": 1.25, "vec": 6, "density": 7.19},
        "Fe": {"r": 1.24, "vec": 8, "density": 7.87},
        "Ni": {"r": 1.25, "vec": 10, "density": 8.9},
        "Cu": {"r": 1.28, "vec": 11, "density": 8.96}
    },
    "Lightweight": {
        "Al": {"r": 1.43, "vec": 3, "density": 2.7},
        "Mg": {"r": 1.60, "vec": 2, "density": 1.74},
        "Li": {"r": 1.52, "vec": 1, "density": 0.534},
        "Ti": {"r": 1.47, "vec": 4, "density": 4.5},
        "Zn": {"r": 1.33, "vec": 12, "density": 7.14}
    },
    "Aerospace": {
        "Al": {"r": 1.43, "vec": 3, "density": 2.7},
        "Ti": {"r": 1.47, "vec": 4, "density": 4.5},
        "Sc": {"r": 1.62, "vec": 3, "density": 2.98},
        "Zr": {"r": 1.60, "vec": 4, "density": 6.52},
        "V": {"r": 1.31, "vec": 5, "density": 6.11}
    }
}

# Crystal structure templates: Refractory → BCC (2-atom), others → FCC (4-atom)
structure_templates = {
    "Refractory": {
        "type": "BCC",
        "basis_species": ["H"] * 2,
        "basis_coords": [
            [0.0, 0.0, 0.0],    # corner
            [0.5, 0.5, 0.5],    # body center
        ],
        "supercell": [3, 3, 3],  # 2 * 27 = 54 atoms
        "lattice_param": 3.25,
    },
    "Corrosion": {
        "type": "FCC",
        "basis_species": ["H"] * 4,
        "basis_coords": [
            [0.0, 0.0, 0.0],    # corner
            [0.5, 0.5, 0.0],    # face center (xy)
            [0.5, 0.0, 0.5],    # face center (xz)
            [0.0, 0.5, 0.5],    # face center (yz)
        ],
        "supercell": [2, 2, 3],  # 4 * 12 = 48 atoms
        "lattice_param": 3.56,
    },
    "Lightweight": {
        "type": "FCC",
        "basis_species": ["H"] * 4,
        "basis_coords": [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        "supercell": [2, 2, 3],  # 4 * 12 = 48 atoms
        "lattice_param": 4.05,
    },
    "Aerospace": {
        "type": "FCC",
        "basis_species": ["H"] * 4,
        "basis_coords": [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        "supercell": [2, 2, 3],  # 4 * 12 = 48 atoms
        "lattice_param": 3.90,
    },
}


def compute_ss_strength(comp, master_props_cat):
    """Simplified VLC (2016) solid-solution strengthening: RoM shear modulus
    + atomic size misfit correction. Full VLC needs C11/C12/C44 and line
    tension parameters not available here.
    Ref: Varvenne, Luque & Curtin, Acta Mater. 118 (2016) 164-176
    """
    # RoM shear modulus
    G_rom = sum(frac * shear_moduli[el] for el, frac in comp.items())

    # Mean atomic radius for misfit calculation
    r_avg = sum(frac * master_props_cat[el]['r'] for el, frac in comp.items())

    # Atomic size misfit (delta)
    misfit_sq = sum(
        frac * ((master_props_cat[el]['r'] - r_avg) / r_avg) ** 2
        for el, frac in comp.items()
    )
    delta = np.sqrt(misfit_sq)

    # VLC contribution: tau_ss ~ alpha * G * delta^(2/3)
    alpha = 0.04  # Empirical prefactor for FCC HEAs (Varvenne et al. 2016)
    tau_ss = alpha * G_rom * delta ** (2 / 3) if delta > 0 else 0.0

    return G_rom + tau_ss


def build_sqs_structure(comp, template, elements, seed=42):
    """Builds a randomized SQS-like supercell structure based on a composition.
    """
    total_atoms = len(template["basis_species"]) * np.prod(template["supercell"])
    
    # Distributing atoms across elements
    atom_counts = {el: int(round(comp.get(el, 0.0) * total_atoms)) for el in elements}
    difference = total_atoms - sum(atom_counts.values())
    if difference != 0:
        max_el = max(atom_counts, key=atom_counts.get)
        atom_counts[max_el] += difference
        
    # Building and shuffling atom list for randomization
    atom_list = []
    for el, count in atom_counts.items():
        atom_list.extend([Species(el, 0)] * count)
    
    # Use localized random generator to ensure reproducibility without affecting global state
    rng = random.Random(seed)
    rng.shuffle(atom_list)
    
    # Building supercell from category-specific basis
    lattice = Lattice.cubic(template["lattice_param"])
    dummy_struct = Structure(
        lattice,
        template["basis_species"],
        template["basis_coords"],
    )
    dummy_struct.make_supercell(template["supercell"])
    
    # Replacing dummy atoms with alloy species
    for i in range(total_atoms):
        dummy_struct.replace(i, atom_list[i])
        
    return dummy_struct


print(">>> Starting Comprehensive HEA Discovery <<\n")

# Matminer Magpie featurizer
ep_feat = ElementProperty.from_preset("magpie")

print("\n[1/2] Running Master Engine (Combinatorial Filter -> ML Training -> GA Inverse Design)...")

# Storing best alloy per category
best_alloys_discovered = {}

# Main loop over alloy categories
for cat, elements_dict in alloy_categories.items():
    print(f"\n{'='*60}")
    print(f"Processing Category: {cat.upper()}")
    print(f"{'='*60}")
    
    elements = list(elements_dict.keys())
    allowed_percentages = range(5, 40, 5)
    valid_compositions = []
    
    # Combinatorial generation (5% increments summing to 100%)
    for combo in itertools.product(allowed_percentages, repeat=len(elements)):
        if sum(combo) == 100:
            comp = dict(zip(elements, [x/100.0 for x in combo]))
            valid_compositions.append(comp)
            
    # Physical filtering: delta, VEC, and property proxy calculations
    results = []
    for comp in valid_compositions:
        vec_total = sum(frac * master_props[cat][el]['vec'] for el, frac in comp.items())
        r_avg = sum(frac * master_props[cat][el]['r'] for el, frac in comp.items())
        variance_sum = sum(frac * (1 - master_props[cat][el]['r'] / r_avg)**2 for el, frac in comp.items())
        delta = 100 * np.sqrt(variance_sum)
        
        # RoM density
        rom_density = sum(frac * master_props[cat][el]['density'] for el, frac in comp.items())
        
        # SS strengthening proxy (RoM + VLC misfit)
        rom_strength = compute_ss_strength(comp, master_props[cat])
        
        results.append({
            **comp,
            'VEC': round(vec_total, 3),
            'Delta': round(delta, 3),
            # Analytical proxies, not experimental/DFT data
            'RoM_Density': round(rom_density, 2),
            'RoM_Strength': round(rom_strength, 2),
        })
                        

    df_results = pd.DataFrame(results)
    
    # Stability filtering (Delta < 6.6, VEC thresholds per category)
    if cat == "Refractory":
        df_stable = df_results[(df_results['Delta'] < 6.6) & (df_results['VEC'] >= 5.0) & (df_results['VEC'] <= 6.8)]
    elif cat == "Corrosion":
        df_stable = df_results[(df_results['Delta'] < 6.6) & (df_results['VEC'] >= 8.0)]
    else: 
        df_stable = df_results[(df_results['Delta'] < 6.6)]
        
    df_stable = df_stable.copy()
    print(f"  -> Generated {len(df_results)} combinations.")
    print(f"  -> {len(df_stable)} survived the {cat} physical stability filter.")
    
    if df_stable.empty:
        print("  -> Skipping ML: No stable alloys found.")
        continue
        
    print("\n  [ML] Generating Magpie descriptors...")
    
    # Featurizing stable compositions into 132-dim Magpie vectors
    features_list = []
    for _, row in df_stable.iterrows():
        comp_dict = {el: row[el] for el in elements}
        comp_obj = Composition(comp_dict)
        features_list.append(ep_feat.featurize(comp_obj))
        
    X = np.array(features_list)
    y_density = df_stable['RoM_Density'].values
    y_strength = df_stable['RoM_Strength'].values
    
    # 5-fold cross-validation
    print("  [ML] Running 5-fold cross-validation...")
    
    rf_density_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_strength_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    cv_r2_density = cross_val_score(rf_density_cv, X, y_density, cv=5, scoring='r2')
    cv_r2_strength = cross_val_score(rf_strength_cv, X, y_strength, cv=5, scoring='r2')
    
    print(f"  [ML] Density  5-fold CV R2: {cv_r2_density.mean():.4f} +/- {cv_r2_density.std():.4f}")
    print(f"  [ML] Strength 5-fold CV R2: {cv_r2_strength.mean():.4f} +/- {cv_r2_strength.std():.4f}")
    
    # Training final RF models (80/20 split) for GA inference
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_density, test_size=0.2, random_state=42)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_strength, test_size=0.2, random_state=42)
    
    ml_density = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_d, y_train_d)
    ml_strength = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_s, y_train_s)
    
    # RF holdout evaluation
    rmse_d = np.sqrt(mean_squared_error(y_test_d, ml_density.predict(X_test_d)))
    r2_d = r2_score(y_test_d, ml_density.predict(X_test_d))
    rmse_s = np.sqrt(mean_squared_error(y_test_s, ml_strength.predict(X_test_s)))
    r2_s = r2_score(y_test_s, ml_strength.predict(X_test_s))
    
    print(f"\n  [ML] Random Forest Results (80/20 holdout):")
    print(f"       Density  | RMSE: {rmse_d:.4f} g/cm3 | R2: {r2_d:.4f}")
    print(f"       Strength | RMSE: {rmse_s:.4f} GPa    | R2: {r2_s:.4f}")
    
    # LinearRegression sanity check — comparable R² confirms leakage
    lr_density = LinearRegression().fit(X_train_d, y_train_d)
    lr_strength = LinearRegression().fit(X_train_s, y_train_s)
    
    lr_r2_d = r2_score(y_test_d, lr_density.predict(X_test_d))
    lr_r2_s = r2_score(y_test_s, lr_strength.predict(X_test_s))
    
    print(f"\n  [ML] LinearRegression Sanity Check (same split):")
    print(f"       Density  R2: {lr_r2_d:.4f}  (RF: {r2_d:.4f})")
    print(f"       Strength R2: {lr_r2_s:.4f}  (RF: {r2_s:.4f})")
    if lr_r2_d > 0.99 and lr_r2_s > 0.99:
        print("       [!] Both LR R2 > 0.99 -- confirms targets are linear in Magpie features.")
        print("         These are rule-of-mixtures proxies, not learned material properties.")
    
    # Sampling stable compositions and running CHGNet relaxations
    print(f"\n  [CHGNet] Sampling compositions and running relaxations for energy surrogate training...")
    sample_size = min(50, len(df_stable))
    df_sample = df_stable.sample(n=sample_size, random_state=42).copy()
    
    print(f"  [CHGNet] Running {sample_size} CHGNet relaxations for {cat}...")
    from chgnet.model.dynamics import StructOptimizer
    relaxer = StructOptimizer()
    
    sampled_energies = []
    sampled_features = []
    
    template = structure_templates[cat]
    total_atoms = len(template["basis_species"]) * np.prod(template["supercell"])
    
    for idx, row in df_sample.iterrows():
        comp_dict = {el: row[el] for el in elements}
        struct = build_sqs_structure(comp_dict, template, elements, seed=42)
        
        comp_obj = Composition(comp_dict)
        feats = ep_feat.featurize(comp_obj)
        
        try:
            # StructOptimizer relax call with steps=100
            relax_result = relaxer.relax(struct, steps=100)
            final_energy = relax_result["trajectory"].energies[-1]
            energy_per_atom = final_energy / total_atoms
        except Exception as e:
            print(f"    [!] CHGNet relaxation failed for composition {comp_dict}: {e}")
            energy_per_atom = 0.0 # fallback
            
        sampled_energies.append(energy_per_atom)
        sampled_features.append(feats)
        
    X_energy = np.array(sampled_features)
    y_energy = np.array(sampled_energies)
    
    # Train the energy surrogate RF model (80/20 split)
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_energy, y_energy, test_size=0.2, random_state=42)
    ml_energy = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_e, y_train_e)
    
    rmse_e = np.sqrt(mean_squared_error(y_test_e, ml_energy.predict(X_test_e)))
    r2_e = r2_score(y_test_e, ml_energy.predict(X_test_e))
    
    # LinearRegression sanity check on energy
    lr_energy = LinearRegression().fit(X_train_e, y_train_e)
    lr_r2_e = r2_score(y_test_e, lr_energy.predict(X_test_e))
    
    print(f"\n  [ML] Random Forest Energy Results (80/20 holdout):")
    print(f"       Energy   | RMSE: {rmse_e:.4f} eV/atom | R2: {r2_e:.4f}")
    print(f"  [ML] LinearRegression Sanity Check on Energy:")
    print(f"       Energy   R2: {lr_r2_e:.4f}  (RF: {r2_e:.4f})")
    
    if lr_r2_e < 0.99:
        print("       [!] LR R2 on Energy target is below 0.99 -- confirms energy landscape is non-linear.")
    else:
        print("       [!] LR R2 on Energy target is >= 0.99 -- energy target is surprisingly linear.")
        
    os.makedirs("MetaForge-Web", exist_ok=True)
    
    # Saving trained models
    joblib.dump(ml_density, f"MetaForge-Web/ml_density_{cat}.model")
    joblib.dump(ml_strength, f"MetaForge-Web/ml_strength_{cat}.model")
    joblib.dump(ml_energy, f"MetaForge-Web/ml_energy_{cat}.model")
    
    # GA inverse design - incorporating energy surrogate model
    print("\n  [GA] Initializing Genetic Algorithm to find optimum alloy composition...")
    
    # GA parameters
    POPULATION_SIZE = 50
    GENERATIONS = 20
    MUTATION_RATE = 0.1
    
    # Generating random alloy via Dirichlet distribution
    def generate_random_alloy():
        fractions = np.random.dirichlet(np.ones(len(elements)), size=1)[0]
        return {elements[i]: round(fractions[i], 3) for i in range(len(elements))}

    # Fitness: specific strength balanced with energy minimization stability
    def evaluate_fitness(alloy):
        comp_obj = Composition(alloy)
        feats = [ep_feat.featurize(comp_obj)]
        d = ml_density.predict(feats)[0]
        s = ml_strength.predict(feats)[0]
        e = ml_energy.predict(feats)[0]
        if d <= 0: return -9999.0, d, s, e
        beta = 2.0
        score = (s / d) - beta * e
        return score, d, s, e

    # Initializing population
    population = [generate_random_alloy() for _ in range(POPULATION_SIZE)]
    best_alloy_ever = None
    best_score_ever = -float('inf')
    best_d = 0
    best_s = 0
    best_e = 0

    # Running GA evolution
    for gen in range(GENERATIONS):
        fitness_scores = []
        for alloy in population:
            score, d, s, e = evaluate_fitness(alloy)
            fitness_scores.append(score)
            if score > best_score_ever:
                best_score_ever = score
                best_alloy_ever = alloy
                best_d = d
                best_s = s
                best_e = e
                
        # Selecting top half by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_half = [population[i] for i in sorted_indices[:POPULATION_SIZE//2]]
        
        # Breeding + mutation
        next_gen = top_half[:]
        while len(next_gen) < POPULATION_SIZE:
            parent1 = random.choice(top_half)
            parent2 = random.choice(top_half)
            child = {el: (parent1[el] + parent2[el]) / 2 for el in elements}
            if random.random() < MUTATION_RATE:
                el1, el2 = random.sample(elements, 2)
                shift = random.uniform(0, 0.05)
                if child[el1] > shift:
                    child[el1] -= shift
                    child[el2] += shift
            total = sum(child.values())
            child = {el: round(v / total, 3) for el, v in child.items()}
            next_gen.append(child)
        population = next_gen
        
    print(f"  [GA] Optimization Complete! Top Combined Fitness Score: {best_score_ever:.2f}")
    

    best_str = " - ".join([f"{el}:{v*100:.1f}%" for el, v in best_alloy_ever.items()])
    print(f"       Composition: {best_str}")
    print(f"       Density: {best_d:.2f} g/cm3 | Strength: {best_s:.2f} GPa | Predicted Energy: {best_e:.4f} eV/atom")
    
    best_alloys_discovered[cat] = best_alloy_ever
    
    # SQS blueprint generation
    template = structure_templates[cat]
    total_atoms = len(template["basis_species"]) * np.prod(template["supercell"])
    
    print(f"\n  [CIF] Generating {total_atoms}-atom {template['type']} SQS blueprint for top candidate...")
    
    # Reusing the modular SQS generator function
    dummy_struct = build_sqs_structure(best_alloy_ever, template, elements, seed=42)
        
    print(f"\n  [CIF] Relaxing {cat} {total_atoms}-atom {template['type']} SQS blueprint using CHGNet...")
    
    # CHGNet structural relaxation
    from chgnet.model.dynamics import StructOptimizer
    relaxer = StructOptimizer()
    

    relax_result = relaxer.relax(dummy_struct, steps=100)
    relaxed_struct = relax_result["final_structure"]
    
    # Verifying space group after relaxation
    try:
        sga = SpacegroupAnalyzer(relaxed_struct, symprec=0.1)
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
        print(f"  [CIF] Detected space group after relaxation: {sg_symbol} (#{sg_number})")
        print(f"         Expected: {'Im-3m (BCC)' if template['type'] == 'BCC' else 'Fm-3m (FCC)'}")
    except Exception as e:
        print(f"  [CIF] SpacegroupAnalyzer warning: {e}")
    
    # Real density from the relaxed structure, computed from actual cell
    # volume + composition mass, not the RoM proxy used during screening
    chgnet_density = relaxed_struct.density
    delta_pct = 100 * abs(chgnet_density - best_d) / best_d
    print(f"\n  [VALIDATION] CHGNet-relaxed density:  {chgnet_density:.2f} g/cm\u00b3")
    print(f"               RoM proxy density:      {best_d:.2f} g/cm\u00b3")
    print(f"               Difference: {abs(chgnet_density - best_d):.2f} g/cm\u00b3 ({delta_pct:.1f}%)")
    
    # Saving CIF outputs
    file_name_blueprint = f"Optimal_{cat}_Blueprint.cif"
    CifWriter(dummy_struct).write_file(file_name_blueprint)
    print(f"  [CIF] Saved unrelaxed blueprint to: {file_name_blueprint}")
    
    file_name_relaxed = f"Optimal_{cat}_Relaxed.cif"
    CifWriter(relaxed_struct).write_file(file_name_relaxed)
    print(f"  [CIF] Saved relaxed structure to: {file_name_relaxed}")

print("\n[2/2] Master Script Complete!")
