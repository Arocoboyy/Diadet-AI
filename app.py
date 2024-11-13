from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from ultralytics import YOLO
from tensorflow import expand_dims
import numpy as np
import os
from inference_sdk import InferenceHTTPClient

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

    result=CLIENT.infer("diabet2.jpg", model_id="diadet-ai/1")

    return render_template("index.html", prediction=result)

from inference_sdk import InferenceHTTPClient

if __name__ == "__main__":
    app.run(debug=True)
