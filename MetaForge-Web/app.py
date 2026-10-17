import os
import logging
from flask import Flask, request, jsonify, render_template, make_response, send_from_directory
from flask_cors import CORS
import joblib
import json
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
import warnings
import traceback

warnings.filterwarnings("ignore")

app = Flask(__name__)
# Configuring cross-origin resource sharing
CORS(app, resources={r"/*": {"origins": "*"}})

# Disabling static file caching
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enforcing HTTP security headers
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdnjs.cloudflare.com https://3Dmol.org https://cdn.jsdelivr.net; connect-src 'self' *"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Loading category-specific regression models and Matminer Magpie features
logger.info("Loading MetaForge ML Models...")
models = {
    "Refractory Alloy": {},
    "Corrosion Resistance": {},
    "Lightweight Alloy": {},
    "Aerospace Alloy": {}
}

categories_map = {
    "Refractory Alloy": "Refractory",
    "Corrosion Resistance": "Corrosion",
    "Lightweight Alloy": "Lightweight",
    "Aerospace Alloy": "Aerospace"
}

for web_cat, file_cat in categories_map.items():
    # Look in models/ subdirectory first, fallback to BASE_DIR
    density_path = os.path.join(BASE_DIR, 'models', f'ml_density_{file_cat}.model')
    if not os.path.exists(density_path):
        density_path = os.path.join(BASE_DIR, f'ml_density_{file_cat}.model')
        
    strength_path = os.path.join(BASE_DIR, 'models', f'ml_strength_{file_cat}.model')
    if not os.path.exists(strength_path):
        strength_path = os.path.join(BASE_DIR, f'ml_strength_{file_cat}.model')
        
    energy_path = os.path.join(BASE_DIR, 'models', f'ml_energy_{file_cat}.model')
    if not os.path.exists(energy_path):
        energy_path = os.path.join(BASE_DIR, f'ml_energy_{file_cat}.model')
    
    try:
        if os.path.exists(density_path) and os.path.exists(strength_path):
            models[web_cat]['density'] = joblib.load(density_path)
            models[web_cat]['strength'] = joblib.load(strength_path)
            logger.info(f"Loaded density and strength models for {web_cat}.")
        
        if os.path.exists(energy_path):
            models[web_cat]['energy'] = joblib.load(energy_path)
            logger.info(f"Loaded energy model for {web_cat}.")
    except Exception as e:
        logger.warning(f"Could not load models for {web_cat}: {e}")

# Yield strength trained on MEASURED data (MPEA dataset, RT compression); see
# scripts/train_measured_strength.py. Falls back to the per-category proxy models.
measured_strength = None
measured_meta = {}
try:
    measured_strength = joblib.load(os.path.join(BASE_DIR, 'models', 'ml_strength_measured.model'))
    with open(os.path.join(BASE_DIR, 'models', 'ml_strength_measured.json'), encoding='utf-8') as f:
        measured_meta = json.load(f)
    logger.info("Loaded measured-data strength model.")
except Exception as e:
    logger.warning(f"Measured strength model not loaded, using proxy models: {e}")

# Refractory hybrid (edge-dislocation theory + ML residual, 90% intervals); see
# scripts/export_web_model.py and paper/manuscript_v2.md.
refractory_hybrid = None
try:
    from refractory_hybrid import RefractoryHybrid
    refractory_hybrid = RefractoryHybrid(os.path.join(BASE_DIR, 'models', 'refractory_hybrid.joblib'))
    logger.info("Loaded refractory hybrid strength model.")
except Exception as e:
    logger.warning(f"Refractory hybrid model not loaded: {e}")

# An element is "covered" if at least this many measured alloys contain it.
MIN_ALLOYS_PER_ELEMENT = 20

ep_feat = ElementProperty.from_preset("magpie")
logger.info("All components initialized successfully.")

@app.route('/')
def home():
    """Renders the single-page application."""
    response = make_response(render_template('index.html'))
    # Allow standard caching but require revalidation
    response.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return response

@app.route('/favicon.ico')
def favicon():
    """Serves the favicon.ico icon."""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/robots.txt')
def robots():
    """Returns robots.txt for search engine crawlers."""
    content = "User-agent: *\nAllow: /\nSitemap: https://metaforge-web.onrender.com/sitemap.xml\n"
    response = make_response(content)
    response.headers['Content-Type'] = 'text/plain'
    return response

@app.route('/sitemap.xml')
def sitemap():
    """Returns sitemap.xml for search engines."""
    content = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
   <url>
      <loc>https://metaforge-web.onrender.com/</loc>
      <changefreq>weekly</changefreq>
      <priority>1.0</priority>
   </url>
</urlset>'''
    response = make_response(content)
    response.headers['Content-Type'] = 'application/xml'
    return response

@app.route('/structures/<category>')
def get_structure(category):
    """Serves the relaxed CIF if one exists, else the unrelaxed blueprint."""
    cat_map = {
        "Refractory Alloy": "Refractory",
        "Corrosion Resistance": "Corrosion",
        "Lightweight Alloy": "Lightweight",
        "Aerospace Alloy": "Aerospace"
    }
    file_cat = cat_map.get(category, category)
    # Search data/structures, root, and local directories for relaxed CIF
    candidate_paths = [
        os.path.join(os.path.dirname(BASE_DIR), "data", "structures", f"Optimal_{file_cat}_Relaxed.cif"),
        os.path.join(os.path.dirname(BASE_DIR), f"Optimal_{file_cat}_Relaxed.cif"),
        os.path.join(BASE_DIR, "data", "structures", f"Optimal_{file_cat}_Relaxed.cif"),
        os.path.join(BASE_DIR, f"Optimal_{file_cat}_Relaxed.cif"),
        os.path.join(os.path.dirname(BASE_DIR), "data", "structures", f"Optimal_{file_cat}_Blueprint.cif"),
        os.path.join(os.path.dirname(BASE_DIR), f"Optimal_{file_cat}_Blueprint.cif"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                content = f.read()
            resp = make_response(content)
            resp.headers['Content-Type'] = 'text/plain; charset=utf-8'
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
            
    return jsonify({'error': f'Structure for {category} not found.'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON payload provided.'}), 400

        category = data.get("category")
        elements_data = data.get("elements", {})
        
        if not category or not elements_data:
            return jsonify({'error': 'Missing category or elements.'}), 400
            
        if category not in models or 'density' not in models[category]:
            return jsonify({'error': f'Models not found for category: {category}.'}), 400

        ml_density = models[category]['density']
        ml_strength = models[category]['strength']
        ml_energy = models[category].get('energy')

        elements = list(elements_data.keys())
        raw_values = []
        
        # Extracting elemental weights
        for el in elements:
            val = elements_data.get(el, 0)
            try:
                raw_values.append(float(val))
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid type for element {el}. Expected numeric value.'}), 400

        total = sum(raw_values)
        
        # Preventing division by zero for null compositions
        if total == 0:
            res = {
                'density': 0,
                'strength': 0,
                'score': 0,
                'composition': {el: 0 for el in elements}
            }
            if ml_energy is not None:
                res['energy'] = 0
            return jsonify(res)
        
        # Normalizing atomic weights into molar fractions
        fractions = [v / total for v in raw_values]
        comp_dict = {el: frac for el, frac in zip(elements, fractions)}
        
        # Generating Magpie feature vectors from composition
        comp = Composition(comp_dict)
        features = ep_feat.featurize(comp)
        
        # Predicting bulk density and strength metrics
        density = ml_density.predict([features])[0]
        hybrid = None
        if (category == "Refractory Alloy" and refractory_hybrid is not None
                and refractory_hybrid.covers(comp_dict)):
            rt, rt_lo, rt_hi = refractory_hybrid.predict(comp_dict, features, 25.0)
            ht, ht_lo, ht_hi = refractory_hybrid.predict(comp_dict, features, 1000.0)
            strength = rt / 1000.0
            strength_source = (f"Hybrid model: edge-dislocation theory + ML correction, 72 measured BCC alloys. "
                               f"90% interval {rt_lo / 1000:.2f}-{rt_hi / 1000:.2f} GPa at 25 C; "
                               f"at 1000 C {ht / 1000:.2f} GPa ({ht_lo / 1000:.2f}-{ht_hi / 1000:.2f})")
            notes = refractory_hybrid.warnings(comp_dict, features)
            if ht > refractory_hybrid.ys_max_hot_MPa:
                notes.append(f"1000 C value exceeds the strongest alloy measured at >= 900 C "
                             f"({refractory_hybrid.ys_max_hot_MPa:.0f} MPa)")
            coverage_warning = "; ".join(notes) + "." if notes else None
            hybrid = {'strength_25C_interval': [round(rt_lo / 1000, 3), round(rt_hi / 1000, 3)],
                      'strength_1000C': round(ht / 1000, 3),
                      'strength_1000C_interval': [round(ht_lo / 1000, 3), round(ht_hi / 1000, 3)]}
        elif measured_strength is not None:
            strength = measured_strength.predict([features])[0]
            counts = measured_meta.get('element_counts', {})
            sparse = sorted(el for el, frac in comp_dict.items()
                            if frac >= 0.05 and counts.get(el, 0) < MIN_ALLOYS_PER_ELEMENT)
            strength_source = (f"Measured-data model (MPEA, RT compression, "
                               f"CV MAE {measured_meta.get('cv_mae_gpa', 0):.2f} GPa)")
            coverage_warning = (f"Few measured alloys contain {', '.join(sparse)}; treat this strength as an extrapolation."
                                if sparse else None)
        else:
            strength = ml_strength.predict([features])[0]
            strength_source = "Analytical proxy (solid-solution model), not measured data"
            coverage_warning = None
        score = strength / density if density > 0 else 0
        
        res = {
            'density': round(float(density), 2),
            'strength': round(float(strength), 2),
            'score': round(float(score), 2),
            'composition': {el: round(frac * 100, 1) for el, frac in comp_dict.items()},
            'strength_source': strength_source,
            'coverage_warning': coverage_warning,
        }
        if hybrid:
            res.update(hybrid)

        if ml_energy is not None:
            energy = ml_energy.predict([features])[0]
            res['energy'] = round(float(energy), 3)
            
        return jsonify(res)
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.run(debug=debug_mode, port=5000)