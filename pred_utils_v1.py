#!/usr/bin/env python
# coding: utf-8

# In[6]:


import dice_ml
from dice_ml.utils import helpers 
from dice_ml.constants import ModelTypes, _SchemaVersions
from dice_ml.utils.serialize import DummyDataInterface
import lime
from lime.lime_tabular import LimeTabularExplainer
import fraud_utils
from fraud_utils import *
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import json
import plotly



def limeExplainer_live(pipeline, X_train, live_instance, model_name):
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'bal_chg', 
                'orig_zero', 'amt_bal_ratio', 'chg_amt_ratio']

    def pipeline_predict(data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=features)
        return pipeline.predict_proba(data)

    # Convert X_train to NumPy array for LIME
    X_train_np = X_train.to_numpy()
    type_column_index = X_train.columns.get_loc('type')

    # Initialize LIME explainer with the training data
    explainer_lime = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=features,
        categorical_features=[type_column_index],
        class_names=['Legitimate', 'Fraudulent'],
        verbose=True,
        mode='classification',
        kernel_width=3
    )

    # Process the live instance and generate the LIME explanation
    live_instance_np = live_instance.iloc[0].to_numpy() if isinstance(live_instance, pd.DataFrame) else live_instance.to_numpy()
    lime_exp = explainer_lime.explain_instance(live_instance_np, pipeline_predict, num_features=len(features))

    # Extract only the feature explanations
    # Filtering out 'Intercept' and any other non-feature related information
    explanation_data = {feature: weight for feature, weight in lime_exp.as_list()}
    
    # Remove known non-feature keys
    non_feature_keys = ['Intercept', 'Prediction_local', 'Right']
    for non_feature_key in non_feature_keys:
        explanation_data.pop(non_feature_key, None)

    # Convert back to list of tuples and return
    feature_explanations = list(explanation_data.items())

    return feature_explanations


 
def interpret_lime_results(pipeline, X_train, live_instance, model_name):
    # Generate the LIME explanation using adapted_lime_Explainer
    lime_explanation_list = limeExplainer_live(pipeline, X_train, live_instance, model_name)

    # Check the model's prediction for the live instance
    predicted_class_index = pipeline.predict(live_instance)[0]
    class_map = {0: 'Legitimate', 1: 'Fraudulent'}
    predicted_class = class_map.get(predicted_class_index, 'Unknown')

    interpretations = []

    # Interpret each feature's effect from the explanation list
    for feature, effect in lime_explanation_list:
        feature_name = ''.join([i for i in feature if i.isalpha() or i == '_']).strip()
        impact_qualitative = 'considered' if effect != 0 else 'not considered a significant factor'
        explanation = f"The feature '{feature_name}' is {impact_qualitative} in the model's classification as '{predicted_class}'."
        interpretations.append(explanation)

    # Add a qualitative summary
    interpretations.append("The model considers multiple features to assess each transaction, focusing on patterns that indicate potentially fraudulent or legitimate activity.")

    # Joining all interpretations with new line character
    return '\n'.join(interpretations)
   



def generate_transaction_features(df):
    """
    Adds transaction-related features to the DataFrame with shorter column names.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame.

    Returns:
    - pandas.DataFrame: DataFrame with added transaction-related features.
    """
    # Calculate the change in balance for each transaction
    df['bal_chg'] = df['newbalanceOrig'] - df['oldbalanceOrg']

    # Create a binary flag indicating whether the original balance was zero
    df['orig_zero'] = np.where(df['oldbalanceOrg'].fillna(0.0) == 0.0, 1, 0)

    # Calculate the ratio of the transaction amount to the original balance
    df['amt_bal_ratio'] = df['amount'] / df['oldbalanceOrg']
    df['amt_bal_ratio'] = df['amt_bal_ratio'].replace(np.inf, 0)

    # Calculate the ratio of the balance change to the transaction amount
    df['chg_amt_ratio'] = df['bal_chg'] / df['amount']
    df['chg_amt_ratio'] = df['chg_amt_ratio'].replace([np.inf, -np.inf], 0)
    df.drop(columns='newbalanceOrig', inplace=True)

    return df


def get_data():
    df = load_fraud_data('fraud_data.csv')
   
    X_train_resampled_ad, X_test, y_train_resampled_ad, y_test = split_and_sample_data(df, target_column = 'isFraud',
                                                                 sampling_technique='adasyn', random_state=42)
    X_train_resampled_smote, X_test, y_train_resampled_smote, y_test = split_and_sample_data(df, target_column = 'isFraud',
                                                                 sampling_technique='smote', random_state=42)

    X_train_resampled_stomek, X_test, y_train_resampled_stomek, y_test = split_and_sample_data(df, target_column = 'isFraud',
                                                                 sampling_technique='smotetomek', random_state=42)
    return  X_train_resampled_ad, y_train_resampled_ad, X_train_resampled_smote, y_train_resampled_smote, X_train_resampled_stomek, y_train_resampled_stomek




def cf_explanations(pipeline, X_train, y_train, live_instance, model_name, total_CFs=1):
    outcome_X_train = X_train.copy()
    outcome_X_train['isFraud'] = y_train

    continuous_features = ['step', 'amount', 'oldbalanceOrg', 'bal_chg', 'amt_bal_ratio', 'chg_amt_ratio']
    d = dice_ml.Data(dataframe=outcome_X_train, continuous_features=continuous_features, outcome_name='isFraud')
    m = dice_ml.Model(model=pipeline, backend='sklearn')

    if not isinstance(live_instance, pd.DataFrame):
        live_instance = pd.DataFrame([live_instance], columns=X_train.columns)

    exp = dice_ml.Dice(d, m, method='random')
    cf = exp.generate_counterfactuals(live_instance, total_CFs=total_CFs, desired_class="opposite")

    cf_as_dict = cf.cf_examples_list[0].final_cfs_df.drop(columns=['isFraud']).iloc[0].to_dict()
    original_instance = live_instance.iloc[0].to_dict()

    return cf_as_dict, original_instance



import plotly.graph_objs as go

def visualize_counterfactuals_radar_plotly(original_instance, cf_as_dict):
    # Prepare data for visualization
    features = list(cf_as_dict.keys())
    original_values = []
    counterfactual_values = []

    for feature in features:
        original = original_instance.get(feature, None)
        counterfactual = cf_as_dict.get(feature, None)
        if original is not None and counterfactual is not None:
            original_values.append(original)
            counterfactual_values.append(counterfactual)

    # Creating a radar chart using Plotly Graph Objects
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=original_values,
        theta=features,
        fill='toself',
        name='Original',
        line=dict(color='blue'),
        # Set fill color with transparency for the Original area
        fillcolor='rgba(0, 0, 255, 0.1)'  
    ))

    fig.add_trace(go.Scatterpolar(
        r=counterfactual_values,
        theta=features,
        fill='toself',
        name='Counterfactual',
        # Set color for the Counterfactual line
        line=dict(color='red'), 
        # Set fill color with transparency for the Counterfactual area
        fillcolor='rgba(255, 0, 0, 0.1)'  
    ))

    # Update the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(original_values + counterfactual_values), 
                       max(original_values + counterfactual_values)]
            )),
        title='Counterfactual Radar Chart',
        showlegend=True
    )

    return fig



def visualize_counterfactuals_plotly(original_instance, cf_as_dict):
    # Prepare data for visualization
    features = list(cf_as_dict.keys())
    original_values = []
    counterfactual_values = []
    
    for feature in features:
        original = original_instance.get(feature, None)
        counterfactual = cf_as_dict.get(feature, None)
        if original is not None and counterfactual is not None:
            if original != counterfactual:  # Exclude features with no change
                original_values.append(original)
                counterfactual_values.append(counterfactual)
    
    # Creating bar charts using Plotly Graph Objects
    fig = go.Figure(data=[
        go.Bar(name='Original', x=features, y=original_values, marker_color='blue'),  # Set color for the Original bars
        go.Bar(name='Counterfactual', x=features, y=counterfactual_values, marker_color='red')  # Set color for the Counterfactual bars
    ])
    
    # Update the layout
    fig.update_layout(
        barmode='group',
        title='Counterfactual Explanations',
        xaxis_title='Feature',
        yaxis_title='Value'
    )

    return fig



def explain_counterfactual_percentage(original_instance, cf_as_dict):
    """
    Generate counterfactual explanations for a given live instance and transform them 
    into a user-friendly narrative.

    Parameters:
    pipeline (Pipeline): The trained pipeline object containing the scaler and the model.
    X_train (DataFrame): The training dataset.
    y_train (Series): The training data labels.
    live_instance (DataFrame/Series): The live instance for which counterfactual explanations are generated.
    model_name (str): Name or identifier for the machine learning model.
    total_CFs (int): Total number of counterfactuals to generate.

    Returns:
    str: A narrative explaining the counterfactuals.
    """

    # Generate counterfactual explanations
    #cf_dict, original_instance = cf_explanations2(pipeline, X_train, y_train, live_instance, model_name, total_CFs)

    # Construct narrative, skipping no-change scenarios
    narrative = "To change the model's prediction, consider the following adjustments: \n"
    for feature, new_value in cf_as_dict.items():
        original_value = original_instance[feature]
        if original_value != 0:
            percentage_change = ((new_value - original_value) / original_value) * 100
            if abs(percentage_change) > 0.01:  # Filter out negligible changes
                narrative += f"- Change '{feature}' by {percentage_change:.2f}% (from {original_value} to {new_value}).\n"
        elif new_value != 0:  # Handle cases where the original value is zero, but the new value is not
            narrative += f"- Set '{feature}' to {new_value} (currently zero or undefined).\n"

    return narrative


def visualize_counterfactuals_radar_plotly_v1(original_instance, cf_as_dict):
    # Prepare data for visualization
    features = list(cf_as_dict.keys())
    original_values = []
    counterfactual_values = []

    for feature in features:
        original = original_instance.get(feature, None)
        counterfactual = cf_as_dict.get(feature, None)
        if original is not None and counterfactual is not None:
            original_values.append(original)
            # Use counterfactual value if different from original, else use None
            counterfactual_values.append(counterfactual if original != counterfactual else None)

    # Filter out None values from counterfactual_values for plotting
    plot_features = [feature for feature, cf_value in zip(features, counterfactual_values) if cf_value is not None]
    plot_counterfactual_values = [cf_value for cf_value in counterfactual_values if cf_value is not None]

    # Creating a radar chart using Plotly Graph Objects
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=original_values,
        theta=features,
        fill='toself',
        name='Original',
        line=dict(color='blue'),
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))

    if plot_counterfactual_values:
        fig.add_trace(go.Scatterpolar(
            r=plot_counterfactual_values,
            theta=plot_features,
            fill='toself',
            name='Counterfactual',
            line=dict(color='red'),
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))

    # Update the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(original_values + [val for val in counterfactual_values if val is not None]), 
                       max(original_values + [val for val in counterfactual_values if val is not None])]
            )),
        title='Counterfactual Radar Chart',
        showlegend=True
    )

    return fig



def visualize_counterfactuals_plotly_v1(original_instance, cf_as_dict):
    # Prepare data for visualization
    features = list(cf_as_dict.keys())
    original_values = []
    counterfactual_values = []
    filtered_features = []
    
    for feature in features:
        original = original_instance.get(feature, None)
        counterfactual = cf_as_dict.get(feature, None)
        
        # Include only features where the original and counterfactual values are different
        if original is not None and counterfactual is not None and original != counterfactual:
            filtered_features.append(feature)
            original_values.append(original)
            counterfactual_values.append(counterfactual)
    
    # Creating bar charts using Plotly Graph Objects
    fig = go.Figure(data=[
        go.Bar(name='Original', x=filtered_features, y=original_values, marker_color='blue'),
        go.Bar(name='Counterfactual', x=filtered_features, y=counterfactual_values, marker_color='red')
    ])
    
    # Update the layout
    fig.update_layout(
        barmode='group',
        title='Counterfactual Explanations',
        xaxis_title='Feature',
        yaxis_title='Value'
    )

    return fig


