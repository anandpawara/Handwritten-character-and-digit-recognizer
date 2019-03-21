# from load import *
import importlib.util
from flask import Flask, render_template, request
# from scipy.misc import imsave, imread, imresize
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
import base64
from model import *
from model.load import *
# sys.path.append('./model/load')
app = Flask(__name__)
global model, graph
model, graph = init()


def convertImage(imgData1):
    imgstr = re.search(b'base64,(*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    print("debug")
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    print("debug2")
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        print("debig3")
        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == "__main__":
    app.run(debug=True)
