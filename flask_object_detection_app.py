from flask import Flask, render_template, request, session, Response
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.io.json import json_normalize
import csv
import os
import cv2
import base64
import json
import pickle
from werkzeug.utils import secure_filename
 
import cv2
import numpy as np

import torch

 
#*** Backend operation
 
# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
 
# Accepted image for to upload for object detection model
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'You Will Never Guess'
 
# YOLO object detection function
def detect_object(uploaded_image_path):
    # # Loading image
    img = str(uploaded_image_path)
    model = torch.hub.load('ultralytics/yolov5:master', 'custom', 'D:\\myStudy\\AI\\Project\\Code\\Deploy\\train_results\\weights\\best.pt')
    output_image = model(img)

    return(output_image)
 
 
@app.route('/')
def index():
    return render_template('./index_upload_and_display_image.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
 
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 
        return render_template('./index_upload_and_display_image2.html')
 
@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('./show_image.html', user_image = img_file_path)
 
@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path = detect_object(uploaded_image_path)
    output_image_path.show()
    return 
 
# flask clear browser cache (disable cache)
# Solve flask cache images issue
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(debug = True)