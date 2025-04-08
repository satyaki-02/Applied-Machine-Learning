from flask import Flask, request, jsonify
import pickle
from score import score

app = Flask(__name__)

# Load model
MODEL_PATH = "logistic_model.pkl"
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def welcome():
    return jsonify({"message": "Welcome"})


@app.route("/score", methods=["GET", "POST"])
def score_endpoint():
    try:
        if request.method == "GET":
            text = request.args.get("text")
            if not text:
                return jsonify({"error": "Missing 'text' parameter"}), 400
        else:
            data = request.get_json()
            if not data or "text" not in data:
                return jsonify({"error": "Missing 'text' parameter"}), 400
            text = data["text"]

        prediction, propensity = score(text, model, 0.5)
        return jsonify({"prediction": prediction, "propensity": propensity})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
