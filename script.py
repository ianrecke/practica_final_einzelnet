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


    
@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        try:
            print("hello")
            
            else:
                prediction=f'{int(result)} No-definida'
        except ValueError:
            prediction='Error en el formato de los datos'
        
        return render_template("result.html", prediction=prediction)


if __name__=="__main__":

    app.run(port=5001)