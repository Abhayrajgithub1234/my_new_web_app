from flask import Flask, render_template, request, flash,redirect
from werkzeug.utils import secure_filename
import os
import cv2 as cv


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def classifyImage(filename):
    print(f"The {filename} belongs to category of Sunflowers")
    img = cv.imread(f"uploads/{filename}",0)
    


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")
@app.route("/edit", methods = ["GET","POST"])
def edit():
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "ERROR"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "ERROR - File not found"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            classifyImage(filename)
            return render_template("index.html")
        

    return render_template("index.html")

app.run(debug=True)