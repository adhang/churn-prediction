# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:05:36 2022

@author: Adhang
"""
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
import joblib


app = Flask(__name__)

@app.route("/")
def index():
	return render_template('index.html')

@app.route('/result/', methods=["POST"])
def prediction_result():
    # demographic information
    gender = request.form.get('gender')
    senior_citizen = request.form.get('senior-citizen')
    partner = request.form.get('partner')
    dependents = request.form.get('dependents')
    
    # account information
    tenure = request.form.get('tenure')
    monthly_charges = request.form.get('monthly-charges')
    total_charges = request.form.get('total-charges')
    paperless_billing = request.form.get('paperless-billing')
    payment_method = request.form.get('payment-method')
    contract = request.form.get('contract')
    
    # subscribed services
    phone_service = request.form.get('phone-service')
    multiple_lines = request.form.get('multiple-lines')
    internet_service = request.form.get('internet-service')
    online_security = request.form.get('online-security')
    online_backup = request.form.get('online-backup')
    device_protection = request.form.get('device-protection')
    tech_support = request.form.get('tech-support')
    streaming_tv = request.form.get('streaming-tv')
    streaming_movies = request.form.get('streaming-movies')
    
    #create new dataframe
    data = {'gender':gender,
            'senior_citizen':senior_citizen,
            'partner':partner,
            'dependents':dependents,
            'tenure':tenure,
            'monthly_charges':monthly_charges,
            'total_charges':total_charges,
            'paperless_billing':paperless_billing,
            'payment_method':payment_method,
            'contract':contract,
            'phone_service':phone_service,
            'multiple_lines':multiple_lines,
            'internet_service':internet_service,
            'online_security':online_security,
            'online_backup':online_backup,
            'device_protection':device_protection,
            'tech_support':tech_support,
            'streaming_tv':streaming_tv,
            'streaming_movies':streaming_movies,
            }
    
    df_input = pd.DataFrame(data, index=[0])
    
    #load the trained model.
    path = 'trained_model.pkl'
    loaded_model= joblib.load(path)
    
    result = loaded_model.predict(df_input)
    
    for i in result:
      int_result = int(i)
      if (int_result == 0):
        decision = 'Retain'
      elif (int_result==1):
        decision = 'Churn'
      else:
        decision = 'Not defined'
        
    #return the output and load result.html
    return render_template('result.html', status=decision)
    #return render_template('index.html', status=decision)

if __name__ == "__main__":
    app.run()