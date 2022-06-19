# -*- coding: utf-8 -*-
"""
Created on Thu May 12 21:01:46 2022

@author: rubby
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime
from datetime import date
from pytimekr import pytimekr
import joblib
from flask import Flask, request
from flask_api import status
import json

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_predictions():
    if request.method == 'GET':
        station = request.args.get('station_name')
        day = request.args.get('date')
        time = request.args.get('time')
        p = predict_passengers(station, day, time)
        if int(p) > 1280:
            re = 5
        else:
            re = int(p) // 256
        if re > 5:
            re = 5
        result = json.dumps(re)
        return result, status.HTTP_200_OK, {"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}
    
def is_workday(d):
    format = '%Y-%m-%d'
    pydate = datetime.datetime.strptime(d, format)
    pydate = pydate.date()
    holidays = [pytimekr.chuseok(pydate.year), pytimekr.lunar_newyear(pydate.year), pytimekr.hangul(pydate.year), 
                pytimekr.children(pydate.year), pytimekr.independence(pydate.year), pytimekr.memorial(pydate.year), 
                pytimekr.buddha(pydate.year), pytimekr.samiljeol(pydate.year), pytimekr.constitution(pydate.year),
               datetime.date(pydate.year, 1, 1)]
    return pydate.weekday() < 5 and pydate not in holidays

input_var = ['year', 'station_id', 'in_out_en', 'workday', '05_06', '06_07', '07_08', '08_09',
       '09_10', '10_11', '11_12', '12_13', '13_14', '14_15', '15_16', '16_17',
       '17_18', '18_19', '19_20', '20_21', '21_22', '22_23', '23_24', '24_']

def predict_passengers(station_name, day, time):
    df2021 = pd.read_csv('data/2022.csv', index_col=0)
    df2021 = df2021[df2021['in_out'] == 'in']
    try:
        df2021 = df2021[df2021['station_name'] == station_name]
    except:
        return 'wrong station name'
    
    df2021['workday'] = df2021['workday'].astype('str')
    try:
        df2021 = df2021[df2021['workday'] == str(is_workday(day))]
    except:
        return 'wrong date format'
    
    df2021['workday'] = df2021['workday'].astype('bool')
    X = df2021[input_var]
    X = X.drop(time, axis=1)
    rf = joblib.load('model/{}.joblib'.format(time))
    y_pred = rf.predict(X)
    
    if len(y_pred) < 1:
        return 'wrong time format'
    elif len(y_pred) > 1:
        y_pred = y_pred.mean()
    return y_pred

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)