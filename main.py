from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
from screctKey import key  # Ensure the import is correct
from skimage.feature import hog
from skimage import feature

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = key

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model
model = joblib.load("leaf_classifier_model.pkl")

# Define the categories
categories = ["Dasana", "Kepula", "Parijata", "Surali"]

# Check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Classify uploaded image using the model
# Feature extraction methods
def extract_hog_feature(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

def extract_lbp_feature(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= hist.sum()  # Normalize LBP histogram
    return hist

# Image classification function
def classifyIimage(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (64, 64))

    # Extract features
    hog_feature = extract_hog_feature(img_resized)
    lbp_feature = extract_lbp_feature(img_resized)

    # Combine HOG and LBP features
    combined_features = np.hstack((hog_feature, lbp_feature)).reshape(1, -1)

    # Predict the category
    prediction = model.predict(combined_features)
    print(prediction)
    if (prediction[0]<0 or prediction[0]>len(categories)-1):
        return "Match Not found"
    return categories[prediction[0]]

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/edit", methods=["POST"])
def edit():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('home'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('home'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Classify the uploaded image
            classification_result = classifyIimage(filename)
            return render_template("result.html", result=classification_result)
        
    return redirect(url_for('home'))

@app.route("/result")
def result():
    classification_result = request.args.get('result', "No result found.")
    return render_template("result.html", result=classification_result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True,port = 5001)
