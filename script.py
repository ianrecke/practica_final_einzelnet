#importing libraries
import os
import numpy as np
import flask
import category_encoders as ce
import pickle
import pandas as pd
from sklearn import tree
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = to_predict_list
    loaded_model = pickle.load(open("random_forest_classifier_entropy.sav","rb"))
    to_predict = np.array(to_predict).reshape(1,-1)
    result = loaded_model.predict(to_predict)
    return result[0]

def resultados_ordinal(to_predict):
    dataset_owid = pd.read_csv('owid-covid-data.csv',na_values='?')
    dataset_owid_ordinal = pd.read_csv('dataset_ordinal.csv',na_values='?')
    continente_ordinal = dataset_owid_ordinal['continent'].iloc[np.where(dataset_owid['continent'] == to_predict[0])[0][0]]
    region_ordinal = dataset_owid_ordinal['location'].iloc[np.where(dataset_owid['location'] == to_predict[1])[0][1]]
    to_predict[0] = continente_ordinal
    to_predict[1] = region_ordinal
    return to_predict

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