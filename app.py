from flask import Flask, redirect, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0:'Dasana',1:'Kepula',2:'Parijata',3:'Surali'}

model = load_model('model.h5')

model.makePriction()

def projected(image_path,target_size=(100,100)):

