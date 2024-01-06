#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import plotly
from dice_ml import Data, Model, Dice
from lime.lime_tabular import LimeTabularExplainer
import pred_utils_v1
from pred_utils_v1 import *
import fraud_utils
from fraud_utils import *

app = Flask(__name__)

# Load models and data outside of request context for efficiency
try:
    rfc_pipeline = joblib.load('rfc_adasyn_pipeline.pkl')
    gbc_pipeline = joblib.load('gbc_smote_pipeline.pkl')
    mlp_pipeline = joblib.load('mlp_smotetomek_pipeline.pkl')
    X_train_ad, y_train_ad, X_train_smote, y_train_smote, X_train_stomek, y_train_stomek = get_data()
except Exception as e:
    app.logger.error("Failed to load models or data: " + str(e))
    # Consider whether to halt the app startup if critical resources fail to load

# Utility function to convert plotly figures to JSON
def convert_plotly_figure_to_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Function to select the model and corresponding training data based on name
def choose_model_and_data(model_name):
    if model_name == 'Random Forest':
        pipeline = rfc_pipeline
        X_train = X_train_ad
        y_train = y_train_ad
    elif model_name == 'Gradient Boost':
        pipeline = gbc_pipeline
        X_train = X_train_smote
        y_train = y_train_smote
    elif model_name == 'Neural Network':
        pipeline = mlp_pipeline
        X_train = X_train_stomek
        y_train = y_train_stomek
    else:
        raise ValueError("Invalid model name")
    return pipeline, X_train, y_train

# Endpoint to predict and explain
@app.route('/predict_and_explain', methods=['POST'])
def predict_and_explain():
    try:
        data = request.get_json()
        model_name = data['selected_model']
        step = data['step']
        transaction_type = data['transaction_type']
        amount = data['amount']
        oldbalanceOrg = data['oldbalanceOrg']

        # Mapping transaction type to integer and preparing input data
        transaction_type_num = 1 if transaction_type in ["Transfer", "Cash Out", "Payment", "Cash In"] else 0
        user_data = {
            'step': step,
            'type': transaction_type_num,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': oldbalanceOrg + amount
        }
        df = pd.DataFrame([user_data])
        live_df = generate_transaction_features(df)

        # Model selection
        pipeline, X_train, y_train = choose_model_and_data(model_name)

        # Making prediction
        prediction = pipeline.predict(live_df)[0]
        prediction_text = "Fraudulent" if prediction == 1 else "Not Fraudulent"

        # Generate explanations and visualizations
        cf_as_dict, original_instance = cf_explanations(pipeline, X_train, y_train, live_df, model_name, total_CFs=1)
        lime_explanation = interpret_lime_results(pipeline, X_train, live_df, model_name)
        radial_plot = visualize_counterfactuals_radar_plotly(cf_as_dict, original_instance)
        bar_chart = visualize_counterfactuals_plotly(original_instance, cf_as_dict)
        narrative = explain_counterfactual_percentage_v1(original_instance, cf_as_dict)

        # Convert plotly figures to JSON
        radial_plot_json = convert_plotly_figure_to_json(radial_plot)
        bar_chart_json = convert_plotly_figure_to_json(bar_chart)

        return jsonify({
            'prediction_text': prediction_text,
            'lime_explanation': lime_explanation,
            'radial_plot': radial_plot_json,
            'bar_chart': bar_chart_json,
            'narrative': narrative
        })
    except Exception as e:
        app.logger.error("Error in predict_and_explain: " + str(e))
        return jsonify({'error': 'An error occurred during processing.'}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in a production environment

