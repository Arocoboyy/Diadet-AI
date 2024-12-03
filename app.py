from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
from inference_sdk import InferenceHTTPClient
import json

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/images_upload/<filename>')
def display_image(filename):
    return send_from_directory('images_upload', filename)

@app.route("/", methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template("index.html", error="Tidak ada foto yang Anda upload!")
    
    imagefile= request.files['imagefile']
    if imagefile.filename == '':
        return render_template("index.html", error="Tidak ada foto yang Anda upload!")
    image_path="images_upload/"+imagefile.filename
    imagefile.save(image_path)

    CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dAD9MajKfGlG3TYFTd9r"
    )

    result=CLIENT.infer(image_path, model_id="diadet-ai/3")
    pred_class = result['predictions'][0]['class']
    percent = str(float(result['predictions'][0]['confidence'])*100)[:6]
    return render_template("index.html", prediction=pred_class, percent=percent, image_file=imagefile.filename)

from inference_sdk import InferenceHTTPClient

if __name__ == "__main__":
    app.run(debug=True)