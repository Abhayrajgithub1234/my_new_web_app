from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = tf.keras.models.load_model("model/leaf_classifier.h5")

# Define the categories
categories = ["Dasana", "Kepula", "Parijata", "Surali"]

# Check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Classify uploaded image using the model
def classifyImage(filename):
    img = cv2.imread(f"uploads/{filename}")
    img_resized = cv2.resize(img, (64, 64))  # Resize to 64x64
    img_normalized = img_resized / 255.0  # Normalize the image
    img_reshaped = np.reshape(img_normalized, (1, 64, 64, 3))  # Reshape for model input

    # Predict category
    prediction = model.predict(img_reshaped)
    class_index = np.argmax(prediction)
    
    # Handle "No Match" condition
    confidence = np.max(prediction)
    if confidence < 0.5:  # Set a confidence threshold
        return "No Match Found"
    return categories[class_index]

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/edit", methods=["GET", "POST"])
def edit():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return "ERROR"
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return "ERROR - File not found"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Classify the uploaded image
            classification_result = classifyImage(filename)
            return render_template("index.html", result=classification_result)
        
    return render_template("index.html")

# Run the app
app.run(debug=True)
