#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from flask_cors import CORS
import dice_ml
from dice_ml.utils import helpers 
from flask import send_file
from dice_ml.utils.serialize import DummyDataInterface
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
import os
from flask import Flask, render_template_string
import ml_explainability_utils
from ml_explainability_utils import *
from ml_explainability_utils import LIMEModelInterpreter, ModelExplainer, SHAPVisualizer


app = Flask(__name__)
CORS(app)

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


# Utility function to convert network graph to image
def plot_networkx_graph(model_name, technique_name, explainer):
    # Initialize return variables
    network_graph, network_explainer, top_main_effect, top_interaction = None, None, None, None

    try:
        if model_name == "Random Forest" and technique_name == "SHAP":
            network_graph = explainer.network_graph_interaction_strength()
            network_explainer = networkx_exp()
            top_main_effect, top_interaction = explainer.features_interaction_v2()

        elif model_name == "Gradient Boost" and technique_name == "SHAP":
            network_graph = explainer.network_graph_interaction_strength_gbc()
            network_explainer = networkx_exp()
            top_main_effect, top_interaction = explainer.features_interaction_gbc_v2()
        
    except Exception as e:
        print(f"An error occurred while generating the network graph: {e}")
  
    if network_graph is None:
        print("Network Graph Not Available.")

    return network_graph, network_explainer, top_main_effect, top_interaction


# Function to select the model interpretability technique- Options: LIME, 
def choose_model_interpretability(technique_name, feature_explanations, interpreter,visualizer, prediction):
    if technique_name == 'LIME':
        model_explanation = interpreter.interpret_lime_results(feature_explanations, prediction)
        mod_plot = interpreter.plot_lime_explanation(feature_explanations)
        features_influence = interpreter.calculate_and_print_feature_influence(feature_explanations)
    elif technique_name == 'SHAP':
        model_explanation = visualizer.shap_feature_influence_percentage()
        mod_plot = visualizer.shap_exp_plot()
        features_influence = visualizer.shap_feature_influence()
    else:
        raise ValueError("Invalid technique name")
    return model_explanation, mod_plot, features_influence



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


@app.route('/')
def index():
    return render_template_string("""
    <html>
        <head>
            <title>FraudSenseXAI</title>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 0 0 100px; /* Adjust bottom padding */
                    background-color: #f4f4f4;
                    color: #333;
                    line-height: 1.6;
                }
                .container {
                    width: 80%;
                    margin: auto;
                    overflow: hidden;
                    padding-bottom: 120px; /* Additional padding to push content up */
                }
                h1, h2 {
                    color: #0056b3;
                }
                h1 {
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }
                h2 {
                    font-size: 1.8em;
                    margin-top: 30px;
                }
                p {
                    font-size: 1.1em;
                }
                ul {
                    list-style-type: none;
                    padding: 0;
                }
                ul li {
                    background: #e9ecef;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                }
                ul li strong {
                    color: #007bff;
                }
                footer {
                    background-color: #333;
                    color: #fff;
                    text-align: center;
                    padding: 10px;
                    position: fixed;
                    left: 0;
                    bottom: 0;
                    width: 100%;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>FraudSenseXAI</h1>
                <h2>Overview</h2>
                <p><strong>FraudSenseXAI</strong> is an innovative Machine Learning (ML) and Explainable Artificial Intelligence (XAI) application, developed as a part of an MSc final project by Othniel Obasi. This application is dedicated to detecting and analyzing fraudulent activities, with a strong emphasis on the interpretability and transparency of its AI models.</p>
                <h2>Key Features</h2>
                <ul>
                    <li><strong>Robust Fraud Detection:</strong> Utilizes advanced ML techniques to identify fraudulent transactions accurately.</li>
                    <li><strong>Explainable AI Elements:</strong> Employs XAI approaches to provide clear insights into the decision-making processes of the AI.</li>
                    <li><strong>Interactive Web Interface:</strong> Features a user-friendly web application for easy access and interpretation of results.</li>
                    <li><strong>Dynamic Visualizations:</strong> Integrates Plotly for interactive and insightful data visualizations.</li>
                    <li><strong>Applicability Across Sectors:</strong> Suitable for use in finance, e-commerce, digital banking, and other sectors.</li>
                </ul>
                <h2>About the Author</h2>
                <p>This project is an MSc Dissertation on the XAI Application of Fraud Detection, authored by Othniel Obasi. It represents a significant contribution to the field of AI, offering practical solutions and valuable insights for the detection of fraudulent activities using AI. For more information, visit the <a href="https://huggingface.co/spaces/Othniel74/Fraudsense-v1" target="_blank">project page</a>  </p>
            
            </div>
            <footer>
                <p>FraudSenseXAI © 2024</p>
            </footer>
        </body>
    </html>
    """)



# Endpoint to predict and explain
@app.route('/predict_and_explain', methods=['GET', 'POST'])
def predict_and_explain():
    try:
        
        data = request.get_json()
        model_name = data['selected_model']
        technique_name = data['selected_interpretability_method']
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

        df = pd.DataFrame([user_data])
        live_df = generate_transaction_features(df)
        live_instance =  live_df
        
        
        # Model selection
        pipeline, X_train, y_train = choose_model_and_data(model_name)
        
        explainer = ModelExplainer(pipeline, live_df)
        interpreter = LIMEModelInterpreter(pipeline, X_train, live_instance, model_name)
        visualizer = SHAPVisualizer(pipeline, live_instance, X_train)

        # Making prediction
        prediction_v1 = pipeline.predict(live_df)[0]
        prediction_text = "Fraudulent Transaction Suspected!" if prediction_v1 == 1 else "Not Fraudulent"
       
    
        lime_explanation, prediction = interpreter.limeExplainer_live()
        feature_explanations = interpreter.lime_feature_extraction(lime_explanation)
        model_explanation, mod_plot, features_influence = choose_model_interpretability(technique_name, feature_explanations,
                                                                                        interpreter,visualizer, prediction)
        network_graph, network_explainer, top_main_effect, top_interaction = plot_networkx_graph(model_name, technique_name, 
                                                                                                 explainer)
       

        # Generate explanations and visualizations
        cf_as_dict, original_instance = cf_explanations(pipeline, X_train, y_train, live_df, model_name, total_CFs=1)
        radial_plot = visualize_counterfactuals_radar_plotly_v1(original_instance, cf_as_dict)
        bar_chart = visualize_counterfactuals_plotly_v1(original_instance, cf_as_dict)
        narrative = explain_counterfactual_percentage(original_instance, cf_as_dict)

        # Convert plotly figures to JSON
        mod_plot_json = convert_plotly_figure_to_json(mod_plot)
        network_graph_json = convert_plotly_figure_to_json(network_graph)
        radial_plot_json = convert_plotly_figure_to_json(radial_plot)
        bar_chart_json = convert_plotly_figure_to_json(bar_chart)
 

        return jsonify({
            'prediction_text': prediction_text,            
            'model_explanation': model_explanation,
            'mod_plot': mod_plot_json,            
            'features_influence': features_influence,
            'network_graph':network_graph_json,
            'network_explainer': network_explainer,
            'top_main_effect': top_main_effect,
            'top_interaction': top_interaction,
            'radial_plot': radial_plot_json,            
            'bar_chart': bar_chart_json,
            'narrative': narrative
        })
    except Exception as e:
        app.logger.error("Error in predict_and_explain: " + str(e))
        return jsonify({'error': 'An error occurred during processing.'}), 500




if __name__ == "__main__":
    # Set debug to False in a production environment
    debug_mode = os.environ.get("DEBUG", "False") == "True"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)




# In[ ]:





# In[ ]:





# In[ ]:




