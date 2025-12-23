from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model("plant_disease_model.keras")

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

CONFIDENCE_THRESHOLD = 0.5

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except:
        return None

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    img = preprocess_image(request.files["file"].read())
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    preds = model.predict(img)
    confidence = float(np.max(preds))
    idx = int(np.argmax(preds))

    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            "error": "Low confidence",
            "confidence": round(confidence, 4)
        }), 400

    disease = CLASS_NAMES[idx]
    status = "Healthy" if "healthy" in disease.lower() else "Infected"

    return jsonify({
        "disease": disease,
        "status": status,
        "confidence": round(confidence, 4)
    })

@app.route("/")
def home():
    return "Flask ML Server Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
