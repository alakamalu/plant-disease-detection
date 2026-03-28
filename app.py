from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=123l4LVyt2jIC5Vx9sFgtlHZvLx_5viX6"
MODEL_PATH = "plant_disease_model.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

app = Flask(__name__)

# Load model safely
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

# Class names
class_names = [
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        os.makedirs("static", exist_ok=True)
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions)

        result = f"{class_names[class_idx]} ({confidence:.2f})"

        return render_template('index.html', prediction=result, img_path=filepath)

    return "Error"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)