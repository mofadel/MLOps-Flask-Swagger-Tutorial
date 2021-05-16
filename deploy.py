#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20/04/21 18:05
# @Author  : Mo_Fadel
# @File    : deploy.py
# @Software: PyCharm

import flask
import numpy as np
from flasgger import Swagger
import pickle as pkl

## 1- Create the app
app = flask.Flask(__name__)
swagger = Swagger(app)

## 2- Load the trained model
model = pkl.load(open('log_reg.pkl','rb'))
print('Model Loaded Successfully !')

## 3- define our function/service
@app.route('/predict', methods=['POST'])
def predict():
    """ Endpoint taking one input
    ---
    parameters:
        - name: Sepal Length
          in: query
          type: number
          required: true
        - name: Sepal Width
          in: query
          type: number
          required: true
        - name: Petal Length
          in: query
          type: number
          required: true
        - name: Petal Width
          in: query
          type: number
          required: true
    responses:
        200:
            description: "0: Setosa, 1: Versicolour, 2: Virginica"
    """

    s_length = flask.request.args.get("Sepal Length")
    s_width = flask.request.args.get("Sepal Width")
    p_length = flask.request.args.get("Petal Length")
    p_width = flask.request.args.get("Petal Width")

    input_features = np.array([[float(s_length), float(s_width), float(p_length), float(p_width)]])
    prediction = model.predict(input_features)

    return str(prediction[0])

## 4- run the app
if __name__== '__main__':
    app.run(host='127.0.0.1', port=7000)

