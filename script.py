#importing libraries
import os
import numpy as np
import flask
import category_encoders as ce
import pickle
import pandas as pd
from sklearn import tree
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = resultados_ordinal(to_predict_list)
        copia_predict_list = to_predict_list
        try:
            to_predict_list = list(map(float, to_predict_list))
            result = ValuePredictor(to_predict_list)
            if int(result)==0:
                prediction='El numero de casos positivos esta por debajo de la media :D '
            elif int(result)==1:
                prediction='El numero de casos positivos esta por encima de la media, estate alerta!'
            else:
                prediction=f'{int(result)} No-definida'
        except ValueError:
            prediction='Error en el formato de los datos'
        
        return render_template("result.html", prediction=prediction)


if __name__=="__main__":

    app.run(port=5001)