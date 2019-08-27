import h5py
import keras
from keras.utils.io_utils import HDF5Matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from numpy import array
import base64
import cv2
import numpy as np
import math
import random
import Preprocess
import PossibleChar
import PossiblePlate
import os
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import load_model
import time
import tensorflow as tf
import ops as utils_ops
	
from flask import Flask
from flask import jsonify, request
app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
	msg = request.json['msg']

	img_data = msg.encode('utf-8')

	with open("test.jpg", "wb") as fh:
		fh.write(base64.decodebytes(img_data))

	stri = Main.main('test.jpg')
	
	return jsonify({"result":stri}),220

if __name__ == "__main__":
	app.run()
