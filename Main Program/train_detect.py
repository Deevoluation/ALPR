#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:24:53 2017

@author: saquib
"""

"""
The kears implementation of character-digit recognition.
"""
# Part-1 : Building the CNN

# Importing the Keras libraries and packages

from keras.models import Sequential # To initialise the nn as a sequence of layers
from keras.layers import Convolution2D # To make the convolution layer for 2D images
from keras.layers import MaxPooling2D # 
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import CSVLogger
from keras.optimizers import RMSprop

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 1 - Convolution
classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))
# Step 3 - Flattening
classifier.add(Flatten())

classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dropout((0.07)))
classifier.add(Dense(36, activation = 'softmax'))

csv=CSVLogger("epochs2.log")

# Compiling the CNN
classifier.compile(optimizer = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.005), loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('Train',target_size=(64, 64),batch_size=32,class_mode='categorical')

test_set = test_datagen.flow_from_directory('Test',target_size=(64, 64),batch_size=32,class_mode='categorical')

classifier.fit_generator(train_set,steps_per_epoch=47605,epochs=5,validation_data=test_set,validation_steps=1292,callbacks=[csv])

import cv2 
import numpy as np
img=cv2.imread('temp1.jpg')
img=cv2.resize(img,(64,64))
img=np.reshape(img,[1,64,64,3])
classes=classifier.predict_classes(img)
print(classes)

