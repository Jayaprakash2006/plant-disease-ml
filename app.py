from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# 🔥 LOAD MODEL ONCE AT STARTUP
model = tf.keras.models.load_model("plant_disease_model.keras")

CLASS_NAMES = [...]

@app.route("/", methods=["GET"])
def health():
    return "ML service running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)  # ⚡ FAST NOW
    class_idx = np.argmax(preds[0])

    return jsonify({
        "class": CLASS_NAMES[class_idx],
        "confidence": float(preds[0][class_idx])
    })
