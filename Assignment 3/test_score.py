import pickle
import numpy as np
import time
import requests
import subprocess
import pytest
import unittest
from score import score

# Load the best saved model and vectorizer
MODEL_PATH = "logistic_model.pkl"

# Load model and vectorizer once for all tests
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

def test_smoke():
    # """Test if the function runs without crashing."""
    text = "This is a test message."
    prediction, propensity = score(text, model, .5)
    assert prediction is not None
    assert propensity is not None

def test_output_format():
    # """Test if the output types are as expected."""
    text = "Test email content."
    prediction, propensity = score(text, model, .5)
    assert isinstance(prediction, int)
    assert isinstance(propensity, float)

def test_prediction_values():
    # """Test if prediction value is either 0 or 1."""
    text = "Sample spam text."
    prediction, _ = score(text, model, .5)
    assert prediction in [0, 1]

def test_propensity_range():
    # """Test if propensity score is between 0 and 1."""
    text = "Sample ham text."
    _, propensity = score(text, model, .5)
    assert 0.0 <= propensity <= 1.0

@pytest.mark.parametrize("threshold, expected", [(0, 1), (1, 0)])
def test_threshold(threshold, expected):
    # """Test if setting threshold to 0 results in all predictions being 1, and 1 results in all predictions being 0."""
    text = "Random email content."
    prediction, _ = score(text, model, threshold)
    assert prediction == expected

def test_obvious_spam():
    # """Test if an obvious spam text is classified as spam (1)."""
    spam_text = "Congratulations! You won a free iPhone. Click here to claim your prize."
    prediction, _ = score(spam_text, model, .5)
    assert prediction == 1

def test_obvious_non_spam():
    # """Test if an obvious non-spam text is classified as non-spam (0)."""
    ham_text = "Dear team, please find the attached report for your review."
    prediction, _ = score(ham_text, model, .5)
    assert prediction == 0

class TestFlaskIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # """Launches the Flask app using the command line."""
        cls.process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Give the server time to start

    @classmethod
    def tearDownClass(cls):
        # """Closes the Flask app using the command line."""
        cls.process.terminate()
        cls.process.wait()

    def test_flask(self):
        # """Tests the response from the localhost endpoint."""
        url = "http://127.0.0.1:5000/score"
        data = {"text": "This is a test message."}
        response = requests.post(url, json=data)
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("prediction", json_data)
        self.assertIn("propensity", json_data)
        self.assertIsInstance(json_data["prediction"], int)
        self.assertIsInstance(json_data["propensity"], float)
        self.assertGreaterEqual(json_data["propensity"], 0.0)
        self.assertLessEqual(json_data["propensity"], 1.0)

if __name__ == "__main__":
    unittest.main()
