from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow import keras
from tensorflow import expand_dims
import numpy as np
import os


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)