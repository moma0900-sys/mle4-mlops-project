'''Sending example request via .json file to check, if the http-request is working'''
import json
import requests
from pathlib import Path

payload = Path("sample-request.json").read_text(encoding="utf-8")
payload = json.loads(payload)
print(payload)
print(type(payload))

response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)

print(response.status_code)
print(response.json())