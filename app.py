from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Load your trained model
model = load_model('mnist_cnn_model.h5')

# Homepage with form
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Load the image and preprocess
    img = Image.open(file).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))           # Resize to 28x28
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Predict
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    return f"Predicted Digit: {predicted_digit}"

if __name__ == '__main__':
    app.run(debug=True)
