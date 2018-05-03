import numpy as np
import keras
from keras.models import load_model
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.utils.io_utils import HDF5Matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from numpy import array
import base64
	
from flask import Flask
from flask import jsonify, request
app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
	msg = request.json['msg']

	img_data = msg.encode('utf-8')

	with open("test.jpg", "wb") as fh:
		fh.write(base64.decodebytes(img_data))

	model = load_model('C:\\Users/HP/Desktop/Flask_Example/venv/app/food_model-1.h5')
	train_h5_path = 'C:\\Users/HP/Desktop/Flask_Example/venv/app/food_c101_n10099_r32x32x1.h5'
	test_h5_path = 'C:\\Users/HP/Desktop/Flask_Example/venv/app/food_test_c101_n1000_r32x32x1.h5'
	X_train = HDF5Matrix(train_h5_path, 'images')
	y_train = HDF5Matrix(train_h5_path, 'category')
	sample_imgs = 25

	with h5py.File(train_h5_path, 'r') as n_file:
		total_imgs = n_file['images'].shape[0]
		read_idxs = slice(0,sample_imgs)
		im_data = n_file['images'][read_idxs]
		im_label = n_file['category'].value[read_idxs]
		label_names = [x.decode() for x in n_file['category_names'].value]

	img = Image.open("test.jpg")
	arr = array(img)
	import cv2
	arr = cv2.cvtColor(arr,cv2.cv2.COLOR_BGR2GRAY)
	arr = cv2.resize(arr,(32,32))
	nparray = np.asarray(arr)
	nparray = nparray.reshape((1,32,32,1));
	prediction = model.predict_classes(nparray)
	print(label_names[prediction[0]])
	return jsonify({"result":label_names[prediction[0]]}),220

if __name__ == "__main__":
	app.run()
