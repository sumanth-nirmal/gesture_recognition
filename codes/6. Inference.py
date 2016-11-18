from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.optimizers import SGD
import numpy as np
import os
import cv2

# load json and create model
json_file = open('VGG16_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("VGG16_Weights.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Code to load the image from webcam
#cam = cv2.VideoCapture(0)
#ret, im = cam.read()

# Code to load the image from local directory
im = cv2.imread('test_images/5.jpeg')
im = cv2.resize(im, (224, 224)).astype(np.float32)
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)

out = loaded_model.predict(im)
print 'The recognized gesture is Gesture ' + str(np.argmax(out)+1)
