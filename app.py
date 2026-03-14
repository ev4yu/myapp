from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline
import joblib
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# --- Load sentiment model (three‑class) ---
print("--- Loading Sentiment Model (three‑class) ... ---")
sentiment_task = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)
print("--- Sentiment model ready ---")

# --- Load XGBoost model (unchanged) ---
print("--- Loading XGBoost Risk Model... ---")
risk_model = joblib.load("ok.pkl")
print("--- All models loaded ---")

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        user_text = data.get('text', '')
        if not user_text:
            return jsonify({'status': 'empty'}), 400

        results = sentiment_task(user_text)[0]  # list of dicts
        print(f"Raw: '{user_text}' -> {results}")  # debug

        # Convert to dictionary: label -> score
        scores = {item['label'].lower(): item['score'] for item in results}

        # Get percentages
        neg = round(scores.get('negative', 0) * 100, 1)
        neu = round(scores.get('neutral', 0) * 100, 1)
        pos = round(scores.get('positive', 0) * 100, 1)

        # Determine overall category
        if pos > 60:
            category = "positive"
        elif neg > 60:
            category = "negative"
        elif neu > 60:
            category = "neutral"
        else:
            category = "mixed"

        return jsonify({
            'positive': pos,
            'neutral': neu,
            'negative': neg,
            'category': category,
            'status': 'success'
        })
    except Exception as e:
        print("ERROR in /analyze:", e)
        return jsonify({'error': str(e), 'status': 'failed'}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")
        if features is None:
            return jsonify({"error": "Missing 'features'"}), 400
        
        prediction = risk_model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        print("ERROR in /predict:", str(e))
        return jsonify({'error': str(e)}), 500

# --- PAGE ROUTES ---

@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")

@app.route("/model")
def serve_model():
    return send_from_directory("static", "model.html")

@app.route("/sentiment")
def sentiment():
    return send_from_directory("static", "sentiment.html")

@app.route("/trends")
def trends():
    return send_from_directory("static", "trends.html")

@app.route("/game")
def game():
    return send_from_directory("static", "game.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)  # debug=True for auto-reload and better error messages