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
    return modelo,dates,world_cases

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
    
def diferencia_covid(prediccion,world_cases):
    diferencia = prediccion - world_cases[-1] 
    porcentaje = diferencia/(prediccion+world_cases[-1])
    return porcentaje

def diferencia_pais(prediccion,pais,world_cases):
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    localizacion_pais = np.where(confirmed_df['Country/Region'] == pais)[0][0]
    casos_pais = confirmed_df.iloc[localizacion_pais][-1] 
    media_antes = casos_pais/world_cases[-1]
    media_despues = casos_pais/prediccion
    diferencia = (media_despues-media_antes)*100
    return diferencia

def calculaBMI(estatura,peso):
    BMI = (peso/((estatura*0.01)**2))
    if BMI >=30:
        return 1
    else:
        return 2

def prediccion_uci(datos):
    naive = pickle.load(open("naive_bayes.pkl","rb"))
    naive = naive.predict(np.array(datos).reshape(1,-1))
    return naive


def convierteString(elemento):
    if elemento == '1' or elemento == 1:
        return "Si."
    elif (elemento == '2' or elemento == 2):
        return "No."
    elif(elemento == '97' or elemento == 97):
        return "No aplica."
    elif(elemento == '98' or elemento == 98):
        return "Se ignora."
    else:
        return "No especificado."

def get_muertes():
    dataset_6 = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    gente_muerta = np.where(dataset_6['date_died'] == '9999-99-99')
    gente_muerta = np.delete(np.array(dataset_6),gente_muerta,axis = 0)
    total_muertos = gente_muerta.shape[0]
    gente_muerta=pd.DataFrame(gente_muerta,columns = dataset_6.columns)
    gente_muerta_covid_pos = np.where(gente_muerta['covid_res'] == 1)
    gente_muerta_covid = np.array(np.where(gente_muerta['covid_res'] == 1)).shape[1]
    gente_muerta = np.delete(np.array(gente_muerta),gente_muerta_covid_pos,axis = 0)
    gente_muerta=pd.DataFrame(gente_muerta,columns = dataset_6.columns)
    gente_uci = np.shape(np.where(gente_muerta['icu'] == 1))[1]
    cadena = "porcentaje de muertes totales del estudio: {}%, de los cuales padecen covid son un : {}%, de los cuales un {}% se encontraban en la uci con covid".format(round((total_muertos/np.array(dataset_6).shape[0])*100,2),round((gente_muerta_covid/total_muertos)*100,2),round((gente_uci/total_muertos)*100,2))
    return cadena

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        modelo,dates,world_cases = obtencion_datos()
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        fecha = to_predict_list[0].split(sep = '-')
        fecha = fecha[1]+"/"+fecha[2]+'/'+fecha[0]
        fecha2 = to_predict_list[1].split(sep = '-')
        fecha2 = fecha2[1]+"/"+fecha2[2]+'/'+fecha2[0]
        result = predicciones(fecha,dates,modelo)
        casos_pais1 = result
        diferencia_pais1 = diferencia_pais(result,to_predict_list[2],world_cases)
        result2 = predicciones(fecha2,dates,modelo)
        diferencia_pais2 = diferencia_pais(result2,to_predict_list[3],world_cases)
        casos_pais2 = result2
        result = diferencia_covid(result,world_cases)
        result2 = diferencia_covid(result2,world_cases)
        
        cadena = get_muertes()
        
        datos_uci = to_predict_list[4:]
        obesidad = calculaBMI(int(datos_uci[11]),int(datos_uci[12]))
        datos_uci = np.array(datos_uci)
        datos_uci = np.array(datos_uci)
        datos_uci = np.delete(datos_uci,12)
        datos_uci[11] = obesidad
        naive = prediccion_uci(datos_uci)
        copia_entrada = to_predict_list
        try:
            if result[0][0] < 0:
                prediction = 'El numero de casos de covid a nivel Global disminuye en un '+str(result[0][0])+' para la fecha indicada'
            else:
                prediction = 'El numero de casos de covid a nivel Global aumenta en un '+str(result[0][0])+' para la fecha indicada'
        except ValueError:
            prediction='Error en el formato de los datos'
        
        return render_template("result.html", fecha1=fecha,result=round(result[0][0],3)*100,result2 = round(result2[0][0],3)*100,fecha2 = fecha2,diferencia_pais1 = round(diferencia_pais1[0][0],3),diferencia_pais2 = round(diferencia_pais2[0][0],3),pais1 = to_predict_list[2],pais2 = to_predict_list[3],casos_pais1 = casos_pais1,casos_pais2 = casos_pais2,sexo = int(datos_uci[0]),intubacion = convierteString(datos_uci[1]),neumonia = convierteString(datos_uci[2]),edad = datos_uci[3],embarazo = convierteString(datos_uci[4]),diabetes = convierteString(datos_uci[5]),asma = convierteString(datos_uci[6]),inmunosupresores = convierteString(datos_uci[7]),hipertension = convierteString(datos_uci[8]),otra_enf = convierteString(datos_uci[9]),cardiovascular = convierteString(datos_uci[10]),bmi = datos_uci[11],renal = convierteString(datos_uci[12]),fumador = convierteString(datos_uci[13]),contacto = convierteString(datos_uci[14]),estatura = to_predict_list[15],peso = to_predict_list[16],resultado_nv = convierteString(naive),cadena = cadena)

if __name__=="__main__":

    app.run(port=5001)