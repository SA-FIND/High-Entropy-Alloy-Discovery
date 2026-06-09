# Importing libraries
import os
import warnings
warnings.filterwarnings('ignore')
import itertools
import random
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from mp_api.client import MPRester
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pymatgen.core import Lattice, Structure, Species
from pymatgen.io.cif import CifWriter
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

warnings.filterwarnings("ignore")
load_dotenv()
api_key = os.getenv("MY_API_KEY")

# Define the 4 categories
alloy_categories = {
    "Refractory":  {'W': 6, 'Mo': 6, 'Ta': 5, 'Nb': 5, 'V': 5},
    "Corrosion":   {'Co': 9, 'Cr': 6, 'Fe': 8, 'Ni': 10, 'Cu': 11},
    "Lightweight": {'Al': 3, 'Mg': 2, 'Li': 1, 'Ti': 4, 'Zn': 12},
    "Aerospace":   {'Al': 3, 'Ti': 4, 'Sc': 3, 'Zr': 4, 'V': 5}
}

# Real-world shear moduli (GPa) for rule of mixtures
shear_moduli = {
    'Al': 26, 'Ti': 44, 'Sc': 29, 'Zr': 33, 'V': 47,
    'Mg': 17, 'Li': 4.2, 'Zn': 43, 'W': 161, 'Mo': 126,
    'Ta': 69, 'Nb': 38, 'Co': 82, 'Cr': 115, 'Fe': 82,
    'Ni': 76, 'Cu': 48
}

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
print(">>> Starting Comprehensive HEA Discovery <<\\n")

ep_feat = ElementProperty.from_preset("magpie")

print("\\n[2/3] Running Master Engine (Combinatorial Filter -> ML Training -> GA Inverse Design)...")
best_alloys_discovered = {}

for cat, elements_dict in alloy_categories.items():
    print(f"\\n{'='*50}")
    print(f"Processing Category: {cat.upper()}")
    print(f"{'='*50}")
    
    elements = list(elements_dict.keys())
    allowed_percentages = range(5, 40, 5)
    valid_compositions = []
    
    for combo in itertools.product(allowed_percentages, repeat=len(elements)):
        if sum(combo) == 100:
            comp = dict(zip(elements, [x/100.0 for x in combo]))
            valid_compositions.append(comp)
            
    # Physical Filtering
    results = []
    for comp in valid_compositions:
        vec_total = sum(frac * master_props[cat][el]['vec'] for el, frac in comp.items())
        r_avg = sum(frac * master_props[cat][el]['r'] for el, frac in comp.items())
        variance_sum = sum(frac * (1 - master_props[cat][el]['r'] / r_avg)**2 for el, frac in comp.items())
        delta = 100 * np.sqrt(variance_sum)
        
        density = sum(frac * master_props[cat][el]['density'] for el, frac in comp.items())
        strength = sum(frac * shear_moduli[el] for el, frac in comp.items())
        
        results.append({**comp, 'VEC': round(vec_total, 3), 'Delta': round(delta, 3), 
                        'Target_Density': round(density, 2), 'Target_Strength': round(strength, 2)})
                        
    df_results = pd.DataFrame(results)
    
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
        
    print("\\n  [ML] Generating Magpie descriptors...")
    features_list = []
    for _, row in df_stable.iterrows():
        comp_dict = {el: row[el] for el in elements}
        comp_obj = Composition(comp_dict)
        features_list.append(ep_feat.featurize(comp_obj))
        
    X = np.array(features_list)
    y_density = df_stable['Target_Density'].values
    y_strength = df_stable['Target_Strength'].values
    
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_density, test_size=0.2, random_state=42)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_strength, test_size=0.2, random_state=42)
    
    ml_density = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_d, y_train_d)
    ml_strength = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_s, y_train_s)
    
    rmse_d = np.sqrt(mean_squared_error(y_test_d, ml_density.predict(X_test_d)))
    r2_d = r2_score(y_test_d, ml_density.predict(X_test_d))
    rmse_s = np.sqrt(mean_squared_error(y_test_s, ml_strength.predict(X_test_s)))
    r2_s = r2_score(y_test_s, ml_strength.predict(X_test_s))
    
    print(f"  [ML] Density Model  | RMSE: {rmse_d:.4f} g/cm³ | R²: {r2_d:.4f}")
    print(f"  [ML] Strength Model | RMSE: {rmse_s:.4f} GPa    | R²: {r2_s:.4f}")
    
    os.makedirs("MetaForge-Web", exist_ok=True)
    joblib.dump(ml_density, f"MetaForge-Web/ml_density_{cat}.model")
    joblib.dump(ml_strength, f"MetaForge-Web/ml_strength_{cat}.model")
    
    print("\\n  [GA] Initializing Genetic Algorithm to find optimum specific strength...")
    POPULATION_SIZE = 50
    GENERATIONS = 20
    MUTATION_RATE = 0.1
    
    def generate_random_alloy():
        fractions = np.random.dirichlet(np.ones(len(elements)), size=1)[0]
        return {elements[i]: round(fractions[i], 3) for i in range(len(elements))}

    def evaluate_fitness(alloy):
        comp_obj = Composition(alloy)
        feats = [ep_feat.featurize(comp_obj)]
        d = ml_density.predict(feats)[0]
        s = ml_strength.predict(feats)[0]
        if d <= 0: return 0, d, s
        return s / d, d, s

    population = [generate_random_alloy() for _ in range(POPULATION_SIZE)]
    best_alloy_ever = None
    best_score_ever = -1
    best_d = 0
    best_s = 0

    for gen in range(GENERATIONS):
        fitness_scores = []
        for alloy in population:
            score, d, s = evaluate_fitness(alloy)
            fitness_scores.append(score)
            if score > best_score_ever:
                best_score_ever = score
                best_alloy_ever = alloy
                best_d = d
                best_s = s
                
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_half = [population[i] for i in sorted_indices[:POPULATION_SIZE//2]]
        
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
        
    print(f"  [GA] Optimization Complete! Top Specific Strength: {best_score_ever:.2f} GPa/(g/cm³)")
    best_str = " - ".join([f"{el}:{v*100:.1f}%" for el, v in best_alloy_ever.items()])
    print(f"       Composition: {best_str}")
    print(f"       Density: {best_d:.2f} g/cm³ | Strength: {best_s:.2f} GPa")
    
    best_alloys_discovered[cat] = best_alloy_ever
    
    print("\\n  [CIF] Generating 54-atom 3D SQS blueprint for top candidate...")
    TOTAL_ATOMS = 54
    atom_counts = {el: int(round(best_alloy_ever[el] * TOTAL_ATOMS)) for el in elements}
    difference = TOTAL_ATOMS - sum(atom_counts.values())
    if difference != 0:
        max_el = max(atom_counts, key=atom_counts.get)
        atom_counts[max_el] += difference
        
    atom_list = []
    for el, count in atom_counts.items():
        atom_list.extend([Species(el, 0)] * count)
    random.seed(42)
    random.shuffle(atom_list)
    
    lattice = Lattice.cubic(3.25)
    dummy_struct = Structure(lattice, ["H", "H"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    dummy_struct.make_supercell([3, 3, 3])
    
    for i in range(TOTAL_ATOMS):
        dummy_struct.replace(i, atom_list[i])
        
    file_name = f"Optimal_{cat}_Blueprint.cif"
    CifWriter(dummy_struct).write_file(file_name)
    print(f"  [CIF] Saved to: {file_name}")

print("\\n[3/3] Master Script Complete!")
