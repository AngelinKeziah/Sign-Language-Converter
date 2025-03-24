import os
import cv2
import numpy as np
import tensorflow as tf
import base64
from gtts import gTTS
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# ✅ Load the trained model
model_path = "D:/SAK THIS PC DOWNLOADS/SignLanguageConverter/asl proj/models/model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
model = tf.keras.models.load_model(model_path)

# ✅ Class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# ✅ Prediction function
def predict_sign(img):
    img_arr = cv2.resize(img, (64, 64))  # Resize image to 64x64
    img_arr = img_arr / 255.0  # Normalize
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
    pred = model.predict(img_arr)[0]
    pred_index = np.argmax(pred)
    return class_labels[pred_index]

# ✅ Generate voice from prediction
def generate_voice(prediction):
    audio_path = os.path.join("static", "prediction.mp3")
    tts = gTTS(text=prediction, lang='en')
    tts.save(audio_path)
    return audio_path

# ✅ Home route
@app.route("/")
def index():
    return render_template("index.html")

# ✅ Portfolio route
@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")

# ✅ Route to predict uploaded image
@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file:
        img_arr = np.frombuffer(file.read(), np.uint8)
        img_arr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img_arr is None:
            return jsonify({"error": "Invalid image format"})

        result = predict_sign(img_arr)
        audio_path = generate_voice(result)
        return jsonify({"prediction": result, "audio": "/static/prediction.mp3"})

    return jsonify({"error": "Failed to process the file"})

# ✅ Serve audio for prediction
@app.route("/static/<path:filename>")
def serve_audio(filename):
    return send_from_directory("static", filename)

# ✅ Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
