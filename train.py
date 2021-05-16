#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20/04/21 18:04
# @Author  : Mo_Fadel
# @File    : train.py
# @Software: PyCharm

# import packages
from sklearn import datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import pickle as pkl
import numpy as np

np.random.seed(123)

# load the iris dataset
iris = datasets.load_iris()
X_input = iris.data
X_target =  iris.target

# # split data into train, validation and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_input, X_target, shuffle=True, test_size=.20)

# define our model
lr = LogisticRegressionCV(cv=5, max_iter=1000)
lr.fit(X_train, Y_train)

# Inference time
test_pred = lr.predict(X_test)
print('Test Accuracy = ', sum(test_pred == Y_test)/len(Y_test))

# Save the model
with open('log_reg.pkl','wb') as f:
    pkl.dump(lr, f)