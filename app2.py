from flask import Flask, request, jsonify
from flask_cors import CORS  # This handles cross-app communication
from transformers import pipeline
import torch

app = Flask(__name__)
CORS(app) # Allows your main website to talk to this specific server

# 1 BERT initialization 
# We use DistilBERT for faster inference during the live presentation
print("--- Loading BERT Sentiment Model... ---")
sentiment_task = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    return_all_scores=True
)
print("--- Model Loaded Successfully! ---")

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        user_text = data.get('text', '')

        if not user_text:
            return jsonify({'positive': 0, 'negative': 0, 'status': 'empty'}), 400

        # 2. Run Inference
        results = sentiment_task(user_text)[0]
        
        # 3. Extract and format percentages
        # LABEL_0 is Negative, LABEL_1 is Positive in this specific model
        neg_score = round(results[0]['score'] * 100, 1)
        pos_score = round(results[1]['score'] * 100, 1)
        
        return jsonify({
            'positive': pos_score,
            'negative': neg_score,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500

if __name__ == '__main__':
    # Running on 5001 to separate the AI logic from the main UI
    print("--- Starting Sentiment Inference Server on Port 5001 ---")