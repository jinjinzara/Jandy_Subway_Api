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
        
        if type(p) is str:
            content = {'error message': p}
            return json.dumps(content), status.HTTP_400_BAD_REQUEST
        
        if int(p) > 12 * 1280:
            re = 3
        elif int(p) > 12 * 852:
            re = 2
        else:
            re = 1
    
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

def search_station(stn, stn_list):
    if stn in stn_list:
        return stn
    
    if '(' in stn:
        stn1 = stn.split('(', 1)[0]
        if stn1 in stn_list:
            return stn1
    
    stn_list_pp = list(map((lambda x: x.split("(",1)[0]), stn_list))
    if stn in stn_list_pp:
        return stn_list[stn_list_pp.index(stn)]
    
    return 'wrong format'

def predict_passengers(station_name, day, time):
    df2022 = pd.read_csv('data/2022.csv', index_col=0)
    df2022 = df2022[df2022['in_out'] == 'in']
    stations = df2022['station_name'].tolist()
    
    if search_station(station_name, stations) in stations:
        df2022 = df2022[df2022['station_name'] == search_station(station_name, stations)]
    else:
        return 'wrong station name'
    
    df2022['workday'] = df2022['workday'].astype('bool')
    try:
        df2022 = df2022[df2022['workday'] == is_workday(day)]
    except:
        return 'wrong date format'
    
    input_var = ['year', 'station_id', 'in_out_en', 'workday', '05_06', '06_07', '07_08', '08_09',
           '09_10', '10_11', '11_12', '12_13', '13_14', '14_15', '15_16', '16_17',
           '17_18', '18_19', '19_20', '20_21', '21_22', '22_23', '23_24', '24_']
    
    X = df2022[input_var]
    try:
        X = X.drop(time, axis=1)
    except:
        return 'wrong time format'
    
    if len(X) < 1:
        return 'invalid data'
    
    rf = joblib.load('model/{}.joblib'.format(time))
    y_pred = rf.predict(X)
    y_pred = y_pred.mean()
    return y_pred

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
