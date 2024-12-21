from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Loaded a pre-trained model (mobilenet_v2) from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(MODEL_URL)

# This model (efficientnet) was less accurate and giving wrong output
# MODEL_URL = "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1"
# model = hub.load(MODEL_URL)

# Downloading and loading ImageNet labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", LABELS_URL)

with open(labels_path, "r") as f:
    imagenet_labels = np.array([line.strip() for line in f.readlines()])

# Directory to save uploaded images in uploads not in static
UPLOAD_FOLDER = "static"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to process the uploaded image
def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode JPEG
    image = tf.image.resize(image, [224, 224])  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Route for serving the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve images after prediction to show in website
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Process the image and predict
    processed_image = process_image(file_path)
    predictions = model(processed_image)  # Run prediction

    # Map prediction
    predicted_class = np.argmax(predictions[0], axis=-1)
    description = imagenet_labels[predicted_class]  

    # Generate URL for the uploaded image
    image_url = url_for('get_uploaded_file', filename=filename, _external=True)

    #return description and image_url
    return jsonify({"description": description, "image_url": image_url})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
