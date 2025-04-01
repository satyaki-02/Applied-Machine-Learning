from flask import Flask, request, jsonify
import pickle
from score import score

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = "logistic_model.pkl"
# VECTORIZER_PATH = "joblib/vectorizer.pkl"
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    
    text = data["text"]
    prediction, propensity = score(text, model, .5)
    
    return jsonify({"prediction": prediction, "propensity": propensity})

if __name__ == "__main__":
    app.run(debug=True)
