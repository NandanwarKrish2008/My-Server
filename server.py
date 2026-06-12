from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
# Allow requests from any origin (essential for local HTML files)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_FILENAME = 'spam_model.pkl'  # ✅ Correct filename from your structure

# Load model globally so it only loads ONCE when the server starts
try:
    model = joblib.load(MODEL_FILENAME)
    print(f"✅ Successfully loaded model: {MODEL_FILENAME}")
except FileNotFoundError:
    print(f"❌ ERROR: '{MODEL_FILENAME}' not found. Please ensure it's in the same folder as server.py")
    model = None
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    model = None

# ==========================================
# ROUTES
# ==========================================

# Root route - serves the HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load on the server. Check terminal."}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request."}), 400

    text = str(data['text']).strip()
    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400

    try:
        # 1. Make prediction (sklearn expects a list of strings)
        prediction = model.predict([text])[0]
        
        # 2. Get confidence score
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            # Find the probability index of the predicted class
            classes = model.classes_
            pred_idx = np.where(classes == prediction)[0][0]
            confidence = float(probs[pred_idx])
        else:
            # Fallback for models without predict_proba (e.g., some SVMs)
            confidence = 0.85 

        # 3. Normalize the output to "Spam" or "Ham" for the frontend
        pred_str = str(prediction).strip().lower()
        
        if pred_str in ['1', 'spam', 'true']:
            final_prediction = "Spam"
        elif pred_str in ['0', 'ham', 'false']:
            final_prediction = "Ham"
        else:
            final_prediction = str(prediction).title() # Fallback for custom labels

        return jsonify({
            "prediction": final_prediction,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({"error": "Model prediction failed. Check server console for details."}), 500

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == '__main__':
    print("🚀 Starting Flask Server on http://localhost:5000")
    print("📁 Serving index.html from current directory")
    print("🔗 API Endpoint: http://localhost:5000/predict")
    print("Press Ctrl+C to stop.")
    app.run(debug=True, host='0.0.0.0', port=5000)