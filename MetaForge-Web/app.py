import os
import logging
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import joblib
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
import warnings
import traceback

warnings.filterwarnings("ignore")

app = Flask(__name__)
# Configuring cross-origin resource sharing
CORS(app, resources={r"/*": {"origins": "*"}})

# Disabling static file caching
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
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; script-src 'self' 'unsafe-inline' https://unpkg.com; connect-src 'self' *"
    return response

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
    try:
        models[web_cat]['density'] = joblib.load(f'ml_density_{file_cat}.model')
        models[web_cat]['strength'] = joblib.load(f'ml_strength_{file_cat}.model')
        logger.info(f"Loaded models for {web_cat}.")
    except Exception as e:
        logger.warning(f"Could not load models for {web_cat}: {e}")

ep_feat = ElementProperty.from_preset("magpie")
logger.info("All components initialized successfully.")

@app.route('/')
def home():
    """Renders the single-page application."""
    response = make_response(render_template('index.html'))
    # Allow standard caching but require revalidation
    response.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return response

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
            return jsonify({'density': 0, 'strength': 0, 'score': 0, 'composition': {el: 0 for el in elements}})
        
        # Normalizing atomic weights into molar fractions
        fractions = [v / total for v in raw_values]
        comp_dict = {el: frac for el, frac in zip(elements, fractions)}
        
        # Generating Magpie feature vectors from composition
        comp = Composition(comp_dict)
        features = ep_feat.featurize(comp)
        
        # Predicting bulk density and strength metrics
        density = ml_density.predict([features])[0]
        strength = ml_strength.predict([features])[0]
        score = strength / density if density > 0 else 0
        
        return jsonify({
            'density': round(float(density), 2),
            'strength': round(float(strength), 2),
            'score': round(float(score), 2),
            'composition': {el: round(frac * 100, 1) for el, frac in comp_dict.items()}
        })
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.run(debug=debug_mode, port=5000)