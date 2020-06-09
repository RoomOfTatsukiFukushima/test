#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 02:32:12 2019

@author: fukushimatatsuki
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data=pd.read_csv('housing.csv',sep=',')
data['total_bedrooms'].fillna(0,inplace=True)

train, test=train_test_split(data, test_size=0.2,random_state=42)

train_x=train.drop('median_house_value',axis=1)
train_y=train['median_house_value'].copy()

test_x=test.drop('median_house_value',axis=1)
test_y=test['median_house_value'].copy()

train_x.drop('ocean_proximity',axis=1)
test_x.drop('ocean_proximity',axis=1)

from sklearn.datasets.base import get_data_home 
from sklean.datasets import fetch_mldata
print (get_data_home()) # これで表示されるパスに mnist-original.mat を置く
mnist=fetch_mldata('MNIST original')

X,y=mnist['data'],mnist['target']
print(X.shape,y.shape)

import matplotlib.pyplot as plt

s=X[20000].reshape(28,28)
plt.imshow(s,interpolation='nearest')

X_train,y_train=X[:60000],y[:60000]
X_test,y_test=X[60000:],y[60000:]

from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_train)
knn_clf.predict(X_test[1:20])
