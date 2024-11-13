from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import numpy as np
import os
from inference_sdk import InferenceHTTPClient
import json

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path="images_upload/"+imagefile.filename
    imagefile.save(image_path)

    CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dAD9MajKfGlG3TYFTd9r"
    )

    result=CLIENT.infer(image_path, model_id="diadet-ai/2")
    return render_template("index.html", prediction=result)

from inference_sdk import InferenceHTTPClient

if __name__ == "__main__":
    app.run(debug=True)