import pytest
import json
import sys
import os

# Add parent directory to path to find app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test that the frontend serves correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Computational Materials Intelligence" in response.data

def test_predict_endpoint_valid_aerospace(client):
    """Test the /predict API with valid, standard composition for Aerospace."""
    payload = {
        "category": "Aerospace Alloy",
        "elements": {'Al': 20, 'Ti': 20, 'Sc': 20, 'Zr': 20, 'V': 20}
    }
    response = client.post('/predict', json=payload)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Verify the structure of the response
    assert 'density' in data
    assert 'strength' in data
    assert 'score' in data
    assert 'composition' in data
    assert data['density'] > 0
    assert data['strength'] > 0
    
    # Verify mathematical normalization (they should all equal exactly 20.0%)
    assert data['composition']['Al'] == 20.0
    assert data['composition']['Ti'] == 20.0

def test_predict_endpoint_all_categories(client):
    """Test the /predict API across all 4 supported metallurgical categories."""
    categories = [
        ("Aerospace Alloy", {'Al': 20, 'Ti': 20, 'Sc': 20, 'Zr': 20, 'V': 20}),
        ("Refractory Alloy", {'W': 20, 'Mo': 20, 'Ta': 20, 'Nb': 20, 'V': 20}),
        ("Corrosion Resistance", {'Co': 20, 'Cr': 20, 'Fe': 20, 'Ni': 20, 'Cu': 20}),
        ("Lightweight Alloy", {'Al': 20, 'Mg': 20, 'Li': 20, 'Ti': 20, 'Zn': 20})
    ]
    
    for cat_name, elements in categories:
        payload = {"category": cat_name, "elements": elements}
        response = client.post('/predict', json=payload)
        assert response.status_code == 200, f"Failed for category: {cat_name}"
        data = json.loads(response.data)
        assert 'density' in data
        assert 'strength' in data
        assert 'score' in data

def test_predict_endpoint_zero_input(client):
    """Test the /predict API with all zeros to prevent division by zero."""
    payload = {
        "category": "Aerospace Alloy",
        "elements": {'Al': 0, 'Ti': 0, 'Sc': 0, 'Zr': 0, 'V': 0}
    }
    response = client.post('/predict', json=payload)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # It should cleanly return 0 instead of crashing
    assert data['density'] == 0
    assert data['strength'] == 0
    assert data['score'] == 0
    assert data['composition']['Al'] == 0

def test_predict_endpoint_missing_element_fields(client):
    """Test the /predict API when some element fields are missing."""
    payload = {
        "category": "Aerospace Alloy",
        "elements": {'Al': 100} # Only sending Al
    }
    response = client.post('/predict', json=payload)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['composition']['Al'] == 100.0

def test_predict_endpoint_missing_category(client):
    """Test the /predict API when category is omitted (should return 400)."""
    payload = {'elements': {'Al': 20, 'Ti': 20}}
    response = client.post('/predict', json=payload)
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_endpoint_invalid_category(client):
    """Test the /predict API with unknown category (should return 400)."""
    payload = {
        "category": "NonExistent Alloy",
        "elements": {'Al': 20, 'Ti': 20}
    }
    response = client.post('/predict', json=payload)
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_endpoint_invalid_payload_type(client):
    """Test the /predict API with invalid data type for element (should return 400)."""
    payload = {
        "category": "Aerospace Alloy",
        "elements": {'Al': 'not_a_number', 'Ti': 20}
    }
    response = client.post('/predict', json=payload)
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Invalid type' in data['error']

def test_predict_endpoint_none_payload(client):
    """Test the /predict API with no payload (should return 400)."""
    response = client.post('/predict', json=None)
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Invalid JSON payload' in data['error']

def test_security_headers(client):
    """Test that security headers are present in the response."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.headers.get('X-Frame-Options') == 'DENY'
    assert response.headers.get('X-Content-Type-Options') == 'nosniff'
    assert 'Content-Security-Policy' in response.headers

def test_structures_endpoint(client):
    """Test the /structures/<category> endpoint for serving CIF crystal structures."""
    # Test valid category
    response = client.get('/structures/Refractory%20Alloy')
    assert response.status_code == 200
    assert 'text/plain' in response.headers.get('Content-Type', '')
    assert b"data_Optimal" in response.data or b"_atom_site" in response.data

    # Test unknown category
    response_404 = client.get('/structures/UnknownAlloy')
    assert response_404.status_code == 404
