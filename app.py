# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:05:36 2022

@author: Adhang
"""
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

#load the trained model.
path = 'static/model/trained_model.pkl'
model = joblib.load(path)

column_list = ['tenure', 'monthly_charges', 'total_charges', 'dependents_yes',
           'internet_service_fiber_optic', 'internet_service_no',
           'online_security_yes', 'tech_support_yes',
           'contract_month_to_month', 'contract_one_year',
           'contract_two_year', 'paperless_billing_yes',
           'payment_method_electronic_check']

transformer = Pipeline(steps=[('preprocessor', model.named_steps['preprocessor']),
                              ('feature_selection', model.named_steps['feature_selection'])])
    
def shap_plot(input_df):
    # transform the input dataframe
    input_transformed = transformer.transform(input_df)
    input_transformed = pd.DataFrame(input_transformed, columns=column_list)
    
    # inverse transform the numeric input
    input_numeric = input_transformed.iloc[:, :3]
    inverse_numeric = transformer.named_steps['preprocessor'].transformers_[0][1].named_steps['scaler'].inverse_transform(input_numeric)
    
    # shap explainer
    explainer = shap.Explainer(model.named_steps['estimator'])
    shap_values = explainer(input_transformed)
    expected_value = explainer.expected_value[1]
    
    # change shap values to 1-d array
    shap_values.values = shap_values.values[:,:,1].flatten()
    shap_values.base_values = shap_values.base_values[0,1]
    
    # overwrite the scaled numerical data with the original data
    shap_values.data[0][0:3] = inverse_numeric.flatten()
    shap_values.data = shap_values.data.flatten()
    
    # calculate final shap values
    shap_values_total = shap_values.values.sum() + shap_values.base_values
    
    shap_status = ''
    if (shap_values_total > shap_values.base_values):
        #print(shap_values_total, 'Churn')
        shap_status = 'greater than'
    elif (shap_values_total < shap_values.base_values):
        #print(shap_values_total, 'Retain')
        shap_status = 'less than'
    else:
        #print('Cannot say')
        shap_status = 'the same as'
        
    # waterfall plot
    shap.waterfall_plot(shap_values, max_display=20, show=False)
    # plt.tight_layout()
    plt.savefig('static/images/shap-output.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    return expected_value, shap_values_total, shap_status
    
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
    
    # create dictionary from the input
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
    
    # create dataframe
    input_df = pd.DataFrame(data, index=[0])
    
    # make prediction
    y_pred = model.predict(input_df)
    
    # convert prediction to text
    for i in y_pred:
      y_pred_int = int(i)
      if (y_pred_int == 0):
        prediction_result = 'retain'
      elif (y_pred_int == 1):
        prediction_result = 'churn'
      else:
        prediction_result = 'not defined'

    # calculate shapley values
    expected_value, shap_values_total, shap_status = shap_plot(input_df)
        
    #return the output and load result.html
    return render_template('result.html',
                           prediction_result=prediction_result,
                           expected_value=expected_value.round(3),
                           shap_values_total=shap_values_total.round(3),
                           shap_status=shap_status)

if __name__ == "__main__":
    app.run()