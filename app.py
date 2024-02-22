#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# Endpoint to predict and explain
@app.route('/predict_and_explain', methods=['POST'])
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




