# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:46:03 2020

@author: admin
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
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

encoder = LabelEncoder()
encoder.fit(target)
encoded_Y = encoder.transform(target)

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=5, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, matrix, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
