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

# Disable static file caching in Flask config
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
    return response

# Loading regression models and Matminer Magpie features
logger.info("Loading MetaForge ML Models...")
ml_density = joblib.load('ml_density.model')
ml_strength = joblib.load('ml_strength.model')
ep_feat = ElementProperty.from_preset("magpie")
logger.info("Models loaded successfully.")

@app.route('/')
def home():
    """Renders the single-page application with strict cache-busting headers."""
    response = make_response(render_template('index.html'))
    # Ensure browser always fetches the latest UI updates
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON payload provided.'}), 400

        # DYNAMIC EXTRACTION: Read whichever elements the frontend sends
        elements = list(data.keys())
        raw_values = []
        
        # Extracting elemental weights
        for el in elements:
            val = data.get(el, 0)
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