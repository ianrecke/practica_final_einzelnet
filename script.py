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

def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 

def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_average.append(np.mean(data[i:i+window_size]))
        else:
            moving_average.append(np.mean(data[i:len(data)]))
    return moving_average

def entrenamientoModelo(X_train_confirmed,X_test_confirmed,y_train_confirmed,future_forcast):
    poly = PolynomialFeatures(degree=4)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    return linear_model

def obtencion_datos():
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    # recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/08-06-2021.csv')
    cols = confirmed_df.keys()
    confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()
    deaths = deaths_df.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()
    world_cases = []
    total_deaths = [] 
    mortality_rate = []
    for i in dates:
        confirmed_sum = confirmed[i].sum()
        death_sum = deaths[i].sum()
        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)
        mortality_rate.append(death_sum/confirmed_sum)
        
    window = 7

    # confirmed cases
    world_daily_increase = daily_increase(world_cases)
    world_confirmed_avg= moving_average(world_cases, window)
    world_daily_increase_avg = moving_average(world_daily_increase, window)

    # deaths
    world_daily_death = daily_increase(total_deaths)
    world_death_avg = moving_average(total_deaths, window)
    world_daily_death_avg = moving_average(world_daily_death, window)
    
    days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)

    days_in_future = 10
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-10]
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[50:], world_cases[50:], test_size=0.03, shuffle=False) 
    modelo = entrenamientoModelo(X_train_confirmed,X_test_confirmed,y_train_confirmed,future_forcast)
    return modelo,dates

def predicciones(prediccion,dates,modelo):
    start = '1/22/2020'
    future_forcast = np.array([i for i in range(len(dates)+20)]).reshape(-1, 1)
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    number_date = np.where(np.array(future_forcast_dates) == prediccion)[0][0]
    poly = PolynomialFeatures(degree=4)
    number_date = poly.fit_transform(number_date.reshape(-1,1))
    prediccion = modelo.predict(number_date)
    return prediccion
    

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        modelo,dates = obtencion_datos()
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        fecha = to_predict_list[0].split(sep = '-')
        fecha = fecha[1]+"/"+fecha[2]+'/'+fecha[0]
        result = predicciones(fecha,dates,modelo)
        copia_entrada = to_predict_list
        try:
            to_predict_list = list(map(float, to_predict_list))
            print("el resultado de la prediccion es: {}".format(result)) 
        except ValueError:
            prediction='Error en el formato de los datos'
        
        return render_template("result.html", prediction=resultado) 


if __name__=="__main__":

    app.run(port=5001)