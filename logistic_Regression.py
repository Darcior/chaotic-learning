# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:20:56 2020

@author: admin
"""
import numpy as np
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

X_train , X_test, y_train,  y_test = train_test_split(matrix, target, random_state=44, train_size = 0.8)

clf = LogisticRegression().fit(X_train, y_train)
clf2 = clf.predict(X_test)
from sklearn.metrics import f1_score
print(f1_score(y_test, clf2, average='micro'))
clf4 = LogisticRegression(max_iter = 2000).fit(X_train, y_train)
clf5 = clf4.predict(X_test)
print(clf4.score(X_test, y_test))
