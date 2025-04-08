import pickle
import numpy as np
import time
import requests
import subprocess
import pytest
from score import score
import os

def test_docker():
    """Tests the Flask app inside the Docker container."""
    # os.system("cd 'Assignment 4'")
    os.system("docker build -t flask-spam-detector:0.0.1 .")
    container = subprocess.Popen(["docker", "run", "-p", "5000:5000", "--name", "flask_container", "flask-spam-detector:0.0.1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(20)

    url = "http://127.0.0.1:5000/score"
    data = {"text": "This is a test message."}
    response = requests.post(url, json=data)
    
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "propensity" in json_data
    assert isinstance(json_data["prediction"], int)
    assert isinstance(json_data["propensity"], float)
    assert 0.0 <= json_data["propensity"] <= 1.0

    print("Stopping Docker container...")
    subprocess.run(["docker", "stop", "flask_container"], check=True)
    print("Removing Docker container...")
    subprocess.run(["docker", "rm", "flask_container"], check=True)
    print("Removing Docker image...")
    subprocess.run(["docker", "rmi", "flask-spam-detector:0.0.1"], check=True)