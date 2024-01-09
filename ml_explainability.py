#!/usr/bin/env python
# coding: utf-8

# In[2]:


from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import lime
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import plotly.express as px
import numpy as np
import shap
import dice_ml
from dice_ml.utils import helpers 
from dice_ml.constants import ModelTypes, _SchemaVersions
from dice_ml.utils.serialize import DummyDataInterface
from IPython.display import display, HTML


def permutation_importance_plot(pipeline, X_test, y_test, model_name):
    """
    Compute and plot permutation importances of features for a given model.

    Parameters:
    - model: A trained scikit-learn model.
    - X: DataFrame
        Feature data used for computing permutation importance.
    - y: Array-like
        Target data used for computing permutation importance.
    - features: List
        A list of feature names in the same order as in the DataFrame X.

    The function calculates the permutation importance for each feature in the model.
    It then plots these importances using a horizontal bar plot, with annotations showing
    the mean importance and its standard deviation.
    """
    features = X_test.columns.tolist()
    # Compute permutation importance
    perm = PermutationImportance(pipeline, random_state=42).fit(X_test, y_test)

    # Create a DataFrame to store feature importances
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': perm.feature_importances_,
        'Std': perm.feature_importances_std_
    })

    # Create a bar plot using Plotly Express
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title=f'Permutation Importances- {model_name}',
                 labels={'Importance': 'Decrease in Model Score', 'Feature': 'Features'})

    # Add annotations for each bar in the plot
    for i, row in importance_df.iterrows():
        fig.add_annotation(
            x=row['Importance'],
            y=row['Feature'],
            text=f"{row['Importance']:.4f} ± {row['Std']:.4f}",
            showarrow=False,
            xshift=10  
        )

    # Update layout settings
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500, width=700)
    fig.show()

    
    
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def partial_dependence_plot(pipeline, X_test):
    """
    Generate and display partial dependence plots for a list of features using a trained model.

    Parameters:
    - model: A trained machine learning model compatible with scikit-learn.
    - X: DataFrame
        Feature data used for generating partial dependence plots.
    - features_list: List
        A list of strings where each string is a feature name for which the partial dependence plot will be generated.

    This function iterates through the provided list of feature names, generating a partial dependence plot for each.
    Each plot displays the relationship between the feature and the predicted outcome, holding all other features constant.
    """
    features_list = X_test.columns
    for feature_name in features_list:
        # Generate the partial dependence plot
        fig, ax = plt.subplots(figsize=(6, 4))  
        PartialDependenceDisplay.from_estimator(
            pipeline, X_test, features=[feature_name], ax=ax
        )

        # Enhance the plot appearance
        ax.set_title(f'Partial Dependence Plot for "{feature_name}"', fontsize=10, fontweight='bold')
        ax.set_xlabel(feature_name, fontsize=10)
        ax.set_ylabel('Average Predicted Outcome', fontsize=10)
        ax.grid(True)  
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='--')  
        plt.tight_layout()  
        plt.show()
        
    
def get_class(y_test, value_num):
    if value_num == 1:
        indices = np.where(y_test == 1)[0][:20]
    if value_num == 0:
        indices = np.where(y_test == 0)[0][:20]
    print(indices.tolist())



def lime_Explainer(pipeline, X_train, X_test, index, model_name):
    
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'bal_chg', 
            'orig_zero', 'amt_bal_ratio', 'chg_amt_ratio']

    def pipeline_predict(data):
        # Check if the data is a NumPy array and convert it to DataFrame with feature names
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=features)
        # Apply the pipeline for prediction
        return pipeline.predict_proba(data)
    # Convert training and test data to NumPy arrays for LIME
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()

    type_column_index = X_train.columns.get_loc('type')
    # Initialize LIME explainer with the training data as a NumPy array
    explainer_lime = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=features,
        categorical_features=[type_column_index],
        class_names=['Legitimate', 'Fraudulent'],
        verbose=True,
        mode='classification',
        kernel_width=3
    )

    instance_index = index
    chosen_instance = X_test.iloc[instance_index].to_numpy()

    lime_exp = explainer_lime.explain_instance(
        chosen_instance, 
        pipeline_predict, 
        num_features=len(features)
    )
    
        
    if instance_index == 1:
        position = 'st'
        
    elif instance_index == 2:
        positon = 'nd'
        
    elif instance_index == 3:
        position = 'rd'
        
    else:
        position = 'th'

    print(f"LIME Explanation for {model_name} for the {instance_index}{position} test instance")
    lime_exp.save_to_file(f"LIME_Explanation_{model_name}.html")

    # Display the explanation in the notebook
    lime_exp.show_in_notebook(show_table=True, show_all=False)
    

    
def lime_instance_generation(pipeline, X_train, X_test, index):
    
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'bal_chg', 
            'orig_zero', 'amt_bal_ratio', 'chg_amt_ratio']

    def pipeline_predict(data):
        # Check if the data is a NumPy array and convert it to DataFrame with feature names
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=features)
        # Apply the pipeline for prediction
        return pipeline.predict_proba(data)


    # Convert training and test data to NumPy arrays for LIME
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()

    type_column_index = X_train.columns.get_loc('type')

    # Initialize LIME explainer with the training data as a NumPy array
    explainer_lime = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=features,
        categorical_features=[type_column_index],
        class_names=['Legitimate', 'Fraudulent'],
        verbose=True,
        mode='classification',
        kernel_width=3
    )

    instance_index = index
    chosen_instance = X_test.iloc[instance_index].to_numpy()

    lime_exp = explainer_lime.explain_instance(chosen_instance, pipeline_predict, num_features=len(features))
    return lime_exp                                          
  
    
def interpret_lime_results_1(pipeline, X_train, X_test, index, num_features=5):
    # Generate the LIME explanation for the specified instance
    lime_explanation = lime_instance_generation(pipeline, X_train, X_test, index)

    # Check the model's prediction for the chosen instance
    chosen_instance = X_test.iloc[index:index+1] 
    predicted_class_index = pipeline.predict(chosen_instance)[0]

    # Map the predicted class index to a more readable form
    class_map = {0: 'Legitimate', 1: 'Fraudulent'}
    predicted_class = class_map.get(predicted_class_index, 'Unknown')

    # Try to get explanations for the predicted class
    try:
        explanations = lime_explanation.as_list(label=predicted_class_index)[:num_features]
    except KeyError:
        # If KeyError, try using label 1 or 0 (common in binary classification)
        try:
            explanations = lime_explanation.as_list(label=1)[:num_features]
        except KeyError:
            explanations = lime_explanation.as_list(label=0)[:num_features]

    print(f"Explanation of the Model's Prediction (Class {predicted_class}) for instance {index}:\n")
    for feature, effect in explanations:
        direction = "increases" if effect > 0 else "decreases"

        if '<=' in feature or '>' in feature:
            condition, value = feature.replace(' ', '').split('<=') if '<=' in feature else feature.replace(' ', '').split('>')
            explanation = f"'{condition}' is {'not more' if '<=' in feature else 'more'} than {value}"
        else:
            explanation = f"based on the feature '{feature}'"
            
        print(f"The model's prediction {direction} likelihood of being classified as {predicted_class} because {explanation}.")

def interpret_lime_results(pipeline, X_train, X_test, index, num_features=5):
    # Generate the LIME explanation for the specified instance
    lime_explanation = lime_instance_generation(pipeline, X_train, X_test, index)
    # Check the model's prediction for the chosen instance
    chosen_instance = X_test.iloc[index:index+1] 
    predicted_class_index = pipeline.predict(chosen_instance)[0]
    # Map the predicted class index to a more readable form
    class_map = {0: 'Legitimate', 1: 'Fraudulent'}
    predicted_class = class_map.get(predicted_class_index, 'Unknown')
    # Try to get explanations for the predicted class
    try:
        explanations = lime_explanation.as_list(label=predicted_class_index)[:num_features]
    except KeyError:
        # If KeyError, try using label 1 or 0 (common in binary classification)
        try:
            explanations = lime_explanation.as_list(label=1)[:num_features]
        except KeyError:
            explanations = lime_explanation.as_list(label=0)[:num_features]
    # Initialize counters and lists for tracking
    increase_count, decrease_count, no_effect_count = 0, 0, 0
    no_effect_features = []
    print(f"Explanation of the Model's Prediction (Class {predicted_class}) for instance {index}:\n")
    for feature, effect in explanations:
        direction = "increases" if effect > 0 else "decreases"
        if effect > 0:
            increase_count += 1
        elif effect < 0:
            decrease_count += 1
        else:
            no_effect_count += 1
            no_effect_features.append(feature.split(' ')[0])  # Extract feature name

        if '<=' in feature or '>' in feature:
            condition, value = feature.replace(' ', '').split('<=') if '<=' in feature else feature.replace(' ', '').split('>')
            explanation = f"'{condition}' is {'not more' if '<=' in feature else 'more'} than {value}"
        else:
            explanation = f"based on the feature '{feature}'"            
        print(f"The model's prediction {direction} likelihood of being classified as {predicted_class} because {explanation}.")
    # Calculate percentages
    total = len(explanations)
    increase_percent = (increase_count / total) * 100
    decrease_percent = (decrease_count / total) * 100
    no_effect_percent = (no_effect_count / total) * 100
    # Print percentages and no-effect features
    print(f"\nPercentage of features that increase prediction: {increase_percent:.2f}%")
    print(f"Percentage of features that decrease prediction: {decrease_percent:.2f}%")
    print(f"Percentage of features with no effect: {no_effect_percent:.2f}%")
    if no_effect_features:
        print(f"Features with no effect on the prediction: {', '.join(no_effect_features)}")
    else:
        print("There are no features with no effect on the prediction.")

        
def shap_KernelExplainer(pipeline, X_train, X_test, sample, model_name):
    """
    Generates SHAP explanations using KernelExplainer for a given pipeline and dataset.

    Args:
    pipeline: A pre-trained sklearn pipeline object, which includes a scaler and a model.
    X_train: The training dataset used for fitting the model. This data will be used for SHAP background distribution.
    X_test: The test dataset for which SHAP values are to be calculated.
    sample: The number of instances to be considered from X_train and X_test for SHAP analysis.
    model_name: Name of the model, used for printing purposes.

    This function scales both training and test datasets using the scaler from the pipeline. It initializes
    a SHAP KernelExplainer with the predict_proba function of the model and a sample of the scaled training data.
    SHAP values are calculated for a sample of the scaled test dataset. Finally, it generates and displays a summary
    plot of the SHAP values using the unscaled test data.
    """
    print(f'SHAP Explainer for {model_name}')
    
    # Scale the training and test data using the scaler from the pipeline
    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
    X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)

    # Initialize SHAP KernelExplainer
    explainer = shap.KernelExplainer(pipeline.named_steps['model'].predict_proba, shap.sample(X_train_scaled, sample))

    # Calculate SHAP values for a sample of the scaled test set
    shap_values = explainer.shap_values(shap.sample(X_test_scaled, sample))

    # Generate and display a SHAP summary plot using the original, unscaled test data
    shap.summary_plot(shap_values, shap.sample(X_test, sample), feature_names=X_test.columns.tolist())
    plt.show()
    

def counterfactual_Explainer1(pipeline, X_train, X_test, y_train, instance_index, model_name):
    """
    Generates counterfactual explanations for a given instance using DiCE.

    Args:
    pipeline: A pre-trained sklearn pipeline that includes a scaler and a model.
    X_train: Training dataset used for DiCE background data preparation.
    X_test: Test dataset for which counterfactuals are to be generated.
    y_train: Training labels corresponding to X_train.
    instance_index: Index of the instance in X_test for which counterfactual is to be generated.

    This function scales X_train using the scaler from the pipeline and prepares the background data for DiCE.
    It then initializes a DiCE data object and model. Counterfactuals are generated for the specified test instance
    after scaling. Finally, the generated counterfactuals are visualized.
    """
    # Scale X_train using the pipeline's scaler
    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Extract the model from the pipeline
    model = pipeline.named_steps['model']

    # Add the outcome column to the scaled X_train
    X_train_with_outcome = X_train_df.copy()
    X_train_with_outcome['isFraud'] = y_train

    # Prepare data for DiCE
    data_for_dice = dice_ml.Data(dataframe=X_train_with_outcome, 
                                 continuous_features=X_train.columns.tolist(), 
                                 outcome_name='isFraud')

    # Initialize DiCE model and explainer
    dice_model = dice_ml.Model(model=model, backend='sklearn')
    exp = dice_ml.Dice(data_for_dice, dice_model)

    # Select and scale the test instance
    test_instance = X_test.iloc[instance_index:instance_index + 1]
    test_instance_scaled = pipeline.named_steps['scaler'].transform(test_instance)
    test_instance_scaled_df = pd.DataFrame(test_instance_scaled, columns=X_test.columns)

    # Generate and visualize counterfactuals for the selected test instance
    counterfactuals = exp.generate_counterfactuals(test_instance_scaled_df, total_CFs=3, desired_class="opposite")
    counterfactuals.visualize_as_dataframe()
          
          
def counterfactual_Mlexplainer(pipeline, X_train, X_test, y_train, instance_index, model_name):
    """
    Generate counterfactual explanations for a given instance using DiCE.

    Parameters:
    - pipeline: Trained machine learning model pipeline.
    - X_train: Training data features.
    - X_test: Test data features.
    - y_train: Training data labels.
    - instance_index: Index of the instance for which counterfactual explanations are generated.
    - model_name: Name or identifier for the machine learning model.
    - cft: Method for generating counterfactuals ('g' for genetic, 'r' for random).

    Returns:
    None
    """
    # Prepare outcome data with labels
    outcome_X_train = X_train.copy()
    outcome_X_train['isFraud'] = y_train

    # Define features for DiCE
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'bal_chg', 'orig_zero',
                'amt_bal_ratio', 'chg_amt_ratio']

    # Create DiCE data object
    d = dice_ml.Data(dataframe=outcome_X_train, continuous_features=features,
                     outcome_name='isFraud')
    # Create DiCE model object
    backend = 'sklearn'
    m = dice_ml.Model(model=pipeline, backend=backend)
    # Select the instance for counterfactual explanation
    query_instances = X_test.iloc[instance_index:instance_index + 1]

    print(f"Counterfactual Explanation for {model_name}")
    print('Genetic Method')
    # Generate counterfactuals using genetic algorithm
    exp_genetic = dice_ml.Dice(d, m, method='genetic')
    dice_exp_genetic = exp_genetic.generate_counterfactuals(query_instances, total_CFs=2, desired_class="opposite", verbose=True)
    dice_exp_genetic.visualize_as_dataframe(show_only_changes=True)        
    print('Kdtree Method')
    # initiate DiceKD
    exp_KD = dice_ml.Dice(d, m, method='kdtree')
    # generate counterfactuals
    dice_exp_KD = exp_KD.generate_counterfactuals(query_instances, total_CFs=2, desired_class="opposite")
    dice_exp_KD.visualize_as_dataframe(show_only_changes=True)        
    print('Random Sampling Method')
    # Generate counterfactuals using random sampling
    exp_random = dice_ml.Dice(d, m, method="random")
    dice_exp_random = exp_random.generate_counterfactuals(query_instances,
                                                          total_CFs=2, desired_class="opposite", verbose=False)
    dice_exp_random.visualize_as_dataframe(show_only_changes=True)
        
 
  
    
def shap_globalExplainer(pipeline, X_train, X_test, model_name):
    """
    Generates SHAP summary plots for global explanations of a model.

    Args:
    pipeline: A pre-trained sklearn pipeline that includes a scaler and a model.
    X_train: Training dataset used for generating background data for SHAP.
    X_test: Test dataset for which SHAP values are to be calculated.
    model_name: Name of the model, used to determine the SHAP explainer type.

    The function first scales the training and test datasets using the scaler from the pipeline.
    It then selects the appropriate SHAP explainer based on the model type. For 'MLP', KernelExplainer is used,
    while for tree-based models, TreeExplainer is used. The SHAP summary plot is generated and displayed.
    """
    
    # Scale X_train and X_test using the scaler from the pipeline
    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
    X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)

    # Convert the scaled datasets back to DataFrames
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Extract the model from the pipeline
    model = pipeline.named_steps['model']

    # Initialize the appropriate SHAP explainer based on the model type
    if model_name == 'MLP Classifier':
        background = shap.sample(X_train_df, 50)  # Sample background data
        explainer_shap = shap.KernelExplainer(model.predict_proba, background)
        
            # Calculate SHAP values for a sample of the scaled test set
        shap_values = explainer_shap.shap_values(shap.sample(X_train_df, 50), check_additivity=False)
    else:    
        explainer_shap = shap.TreeExplainer(model, X_train_df, check_additivity=False)
        # Calculate SHAP values
        shap_values = explainer_shap.shap_values(X_test_df, check_additivity=False)

    # Generate and display SHAP summary plot
    print(f"SHAP Summary Plot for {model_name}")
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.savefig(f"SHAP_Summary_Plot_{model_name}.png")
    plt.show()

# Example usage of the function
# shap_globalExplainaer(your_pipeline, X_train_data, X_test_data, 'MLP')
    
def instance_plot(pipeline, X_train, X_test, model_name, pred_class, baseline,  instance_index):
    """
    Generates SHAP plots for a given instance using a trained machine learning model.

    Parameters:
    pipeline (Pipeline): The trained pipeline object containing the scaler and the model.
    X_train (DataFrame): The training dataset.
    X_test (DataFrame): The testing dataset.
    model_name (str): The name of the model used ('MLP Classifier', 'Random Forest Classifier', etc.).
    plot_name (str): The type of SHAP plot to generate ('decision Plot', 'force Plot', 'dependence Plot').
    pred_class (int): The predicted class index for which the SHAP values are calculated.
    instance_index (int): The index of the instance in the dataset for which the plot is to be generated.

    Returns:
    None: Plots are saved as images and shown.
    """

    # Scale X_train and X_test using the scaler from the pipeline
    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
    X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)

    # Convert the scaled datasets back to DataFrames
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Extract the model from the pipeline
    model = pipeline.named_steps['model']

    # Handling different model types
    if model_name == 'MLP Classifier':
        # SHAP Explainer for MLP Classifier
        background = shap.sample(X_train_df, 100)  # Using X_train_df for background
        explainer_shap = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer_shap.shap_values(shap.sample(X_test_df, 100), check_additivity=False)

        # Generating specified plot
        #if plot_name == 'decision Plot':
        print(f'{model_name} Decision Plot')
        shap.decision_plot(explainer_shap.expected_value[baseline], shap_values[pred_class][instance_index, :], X_test_df.iloc[instance_index, :])
        plt.savefig(f"SHAP_Decision_Plot_{model_name}.png")
        #elif plot_name == 'force Plot':
        print(f'{model_name} Force Plot')
        shap.initjs()
        shap.force_plot(explainer_shap.expected_value[baseline], shap_values[pred_class][instance_index, :], X_test_df.iloc[instance_index, :], matplotlib=True)
        plt.savefig(f"SHAP_Force_Plot_{model_name}.png")

    else:
        # SHAP Explainer for tree-based models
        explainer_shap = shap.TreeExplainer(model)
        X_test_sampled = X_test_df.sample(n=1000, random_state=42)  # 'random_state' ensures reproducibility
        # Calculate SHAP values for the sampled dataset
        shap_values = explainer_shap.shap_values(X_test_sampled)

        # Generating specified plot
        #if plot_name == 'decision Plot':
        print(f'{model_name} Decision Plot')
        shap.decision_plot(explainer_shap.expected_value[baseline], shap_values[pred_class][instance_index, :], X_test_df.iloc[instance_index, :])
        plt.savefig(f"SHAP_Decision_Plot_{model_name}.png")
        #elif plot_name == 'force Plot':
        print(f'{model_name} Force Plot')
        shap.initjs()
        shap.force_plot(explainer_shap.expected_value[baseline], shap_values[pred_class][instance_index, :], X_test_df.iloc[instance_index, :], matplotlib=True);
        plt.savefig(f"SHAP_Force_Plot_{model_name}.png")
    # Display the plot
    plt.show()


def instance_plot_(pipeline, X_train, X_test, model_name, pred_class, instance_index):
    # Scale X_train and X_test using the scaler from the pipeline
    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
    X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)

    # Convert the scaled datasets back to DataFrames
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Extract the model from the pipeline
    model = pipeline.named_steps['model']

    # SHAP Explainer for tree-based models
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(X_test_df)

    # Check if the shap_values is a list (for multi-class) or array (for binary)
    if isinstance(shap_values, list):
        shap_values_instance = shap_values[pred_class][instance_index]
    else:
        shap_values_instance = shap_values[instance_index]

    # Expected value for the plot
    expected_value = explainer_shap.expected_value[pred_class] if isinstance(shap_values, list) else explainer_shap.expected_value

    # Decision Plot
    print(f'{model_name} Decision Plot')
    shap.decision_plot(expected_value, shap_values_instance, X_test.iloc[instance_index, :], show=True)
    
    # Force Plot
    print(f'{model_name} Force Plot')
    shap.initjs()
    force_plot_html = shap.force_plot(expected_value, shap_values_instance, X_test.iloc[instance_index, :], matplotlib=False, show=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
    display(HTML(shap_html))
    plt.close()  # Close any residual plot elements


# In[ ]:




