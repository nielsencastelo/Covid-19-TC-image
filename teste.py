# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:28:11 2020

@author: Nielsen C. Damasceno Dantas
"""

import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np

import cv2 as cv

BS = 8
data = []

new_model = tf.keras.models.load_model('covid19.model')
#new_model.summary()

im= cv.imread('database/normal/1868.png')
#im = cv.imread('database/covid/ciaa199.pdf-001-b.png')



image = cv.cvtColor(im, cv.COLOR_BGR2RGB)
#image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
image = cv.resize(image, (224, 224))
data.append(image)
data = np.array(data) / 255.0


#loss, acc = new_model.evaluate(image,  test_labels, verbose=2)

predIdxs = new_model.predict(data)
prob_normal = predIdxs[0][1] * 100;
prob_cob    = predIdxs[0][0] * 100;
#predIdxs = np.argmax(predIdxs, axis=1)
print("Probabilidade Normal: %.2f" %  prob_normal)
print("Probabilidade Covid: %.2f" %  prob_cob)

#print(classification_report(testY.argmax(axis=1), predIdxs,
#	target_names=lb.classes_))
