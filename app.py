from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

# ----- SETUP -----
app = Flask(__name__)

# Logging for debugging
logging.basicConfig(level=logging.INFO)

# Load model once at startup
logging.info("Loading model...")
model = tf.keras.models.load_model("plant_disease_model.keras")
logging.info("Model loaded successfully!")

CLASS_NAMES = [
    'Pepper_bell___Bacterial_spot',
    'Pepper_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

CONFIDENCE_THRESHOLD = 0.5  # minimum confidence

# ----- HELPERS -----
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logging.error(f"Image preprocessing failed: {e}")
        return None

# ----- ROUTES -----
@app.route("/", methods=["GET"])
def home():
    return "Flask Plant Disease Detection Server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    file = request.files["file"]
    logging.info(f"Received file: {file.filename}")

    img = preprocess_image(file.read())
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Model prediction
    preds = model.predict(img)
    confidence = float(np.max(preds))
    idx = int(np.argmax(preds))
    disease = CLASS_NAMES[idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            "error": "Image not recognized as a leaf or too low confidence",
            "confidence": round(confidence, 4)
        }), 400

    status = "Healthy" if "healthy" in disease.lower() else "Infected"

    logging.info(f"Prediction: {disease}, Confidence: {confidence}")
    return jsonify({
        "disease": disease,
        "status": status,
        "confidence": round(confidence, 4)
    })

# ----- RUN -----
if __name__ == "__main__":
    # For local testing only
    app.run(host="0.0.0.0", port=5000)
