# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:31:03 2020

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
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
matrix = np.loadtxt('param.txt', dtype = 'f')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
target = np.loadtxt('data.txt', dtype = 'f')


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, matrix, target, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))