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

def test_predict_endpoint_valid_input(client):
    """Test the /predict API with valid, standard composition."""
    payload = {'Al': 20, 'Ti': 20, 'Sc': 20, 'Zr': 20, 'V': 20}
    response = client.post('/predict', json=payload)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Verify the structure of the response
    assert 'density' in data
    assert 'strength' in data
    assert 'score' in data
    assert 'composition' in data
    
    # Verify mathematical normalization (they should all equal exactly 20.0%)
    assert data['composition']['Al'] == 20.0
    assert data['composition']['Ti'] == 20.0

def test_predict_endpoint_zero_input(client):
    """Test the /predict API with all zeros to prevent division by zero."""
    payload = {'Al': 0, 'Ti': 0, 'Sc': 0, 'Zr': 0, 'V': 0}
    response = client.post('/predict', json=payload)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # It should cleanly return 0 instead of crashing
    assert data['density'] == 0
    assert data['strength'] == 0
    assert data['score'] == 0
    assert data['composition']['Al'] == 0

def test_predict_endpoint_missing_fields(client):
    """Test the /predict API when some fields are missing (should default to 0)."""
    payload = {'Al': 100} # Only sending Al
    response = client.post('/predict', json=payload)
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Al should be 100%, everything else 0%
    assert data['composition']['Al'] == 100.0
    assert data['composition']['Ti'] == 0.0

def test_predict_endpoint_invalid_payload(client):
    """Test the /predict API with invalid data type (should return 400)."""
    payload = {'Al': 'not_a_number', 'Ti': 20}
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
