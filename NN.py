# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:46:03 2020

@author: admin
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import input_shape
matrix = np.loadtxt('param.txt', dtype = 'f')
embd = np.zeros( (2048, 5) )
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
target = np.loadtxt('data.txt', dtype = 'f')
for i in range(2048):
    if target[i] > 0:
        target[i]=1
    else:
        target[i]=0    

X_train , X_test, y_train,  y_test = train_test_split(embd, target, random_state=44, train_size = 0.2)

model = keras.Sequential([keras.layers.Flatten(input_shape (28,28)),
                keras.layers.Dense(128,activation = tf.nn.sigmoid),                          
                keras.layers.Dense(10,activation = tf.nn.softmax)])
model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy',metrics =['accuracy'])