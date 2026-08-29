import urllib.request
import json

data = json.dumps({"Al": 20, "Ti": 20, "Sc": 20, "Zr": 20, "V": 20}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:5000/predict', data=data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
