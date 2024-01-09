#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[3]:


import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
import seaborn as sns
import uuid

def load_fraud_data(file_path):
    """
    Loads fraud data from a CSV file into a Pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pandas.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
    
    
    
def filter_fraud_data(df):
    """
    Filters the DataFrame to include only rows with 'type' values 'CASH_OUT' or 'TRANSFER'.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame.

    Returns:
    - pandas.DataFrame: DataFrame containing only rows with 'type' values 'CASH_OUT' or 'TRANSFER'.
    """
    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])]
    return df 




def plot_value_counts(df, column_name):
    """
    Creates a bar plot for the value counts of a specified column in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to plot from.
    - column_name (str): The name of the column to plot value counts for.
    """
    value_counts = df[column_name].value_counts()
    
    fig = px.bar(
        data_frame=value_counts,
        x=value_counts.index,
        y=value_counts.values,
        color=value_counts.index,  # Add color based on index
        labels={'x': 'Transaction Type', 'y': 'Transaction Count'},
        height=500,
        width=800  # Set the height and width of the plot
    )
    fig.show()

# Example usage
# Assuming df is your DataFrame and 'type' is the column of interest
#plot_value_counts(df, 'type')




# Function to automatically interpret a box plot

def interpret_boxplot(df, x_column, y_column):
    # Generate descriptive statistics
    stats = df.groupby(x_column)[y_column].describe()

    interpretations = []
    for category, data in stats.iterrows():
        interpretation = f"Category '{category}':\n"
        interpretation += f"- Median transaction amount is {data['50%']}.\n"
        interpretation += f"- The middle 50% of transactions range from {data['25%']} to {data['75%']}.\n"
        iqr = data['75%'] - data['25%']
        upper_whisker = data['75%'] + 1.5 * iqr
        lower_whisker = data['25%'] - 1.5 * iqr
        num_outliers = ((df[df[x_column] == category][y_column] < lower_whisker) | 
                        (df[df[x_column] == category][y_column] > upper_whisker)).sum()
        interpretation += f"- There are approximately {num_outliers} outliers.\n"
        interpretations.append(interpretation)

    return "\n".join(interpretations)


def plot_box(df, x_column, y_column):
    """
    Creates a box plot for specified columns in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to plot from.
    - x_column (str): The name of the column for the x-axis.
    - y_column (str): The name of the column for the y-axis.
    - title (str): The title of the plot.
    - labels (dict): A dictionary for labeling axes. Keys should match the column names.
    - height (int): The height of the plot. Default is 400.
    - width (int): The width of the plot. Default is 800.
    """
    fig = px.box(
        df, 
        x=x_column, 
        y=y_column, 
        title='Fraud Class Box Plot Distribution',
        height=400, 
        width=800
    )
    fig.show()

    
    
def plot_clustered_bar(df, column_to_group, column_to_count, title, height=500, width=800):
    """
    Creates a clustered bar chart for specified columns in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to plot from.
    - column_to_group (str): The name of the column to group by ('isFraud' in this case).
    - column_to_count (str): The name of the column to count values ('type' in this case).
    - title (str): The title of the plot.
    - height (int): The height of the plot. Default is 500.
    - width (int): The width of the plot. Default is 800.
    """
    # Splitting the data into groups
    groups = df[column_to_group].unique()
    frames = []
    for group in groups:
        filtered_data = df[df[column_to_group] == group]
        count_data = filtered_data[column_to_count].value_counts()
        frame = pd.DataFrame({
            column_to_count: count_data.index.tolist(),
            'Count': count_data.values.tolist(),
            'Name': group
        })
        frames.append(frame)
    
    combined_data = pd.concat(frames, ignore_index=True)

    # Creating the plot
    fig = px.bar(
        combined_data, 
        x=column_to_count, 
        y='Count', 
        color='Name',
        barmode='group', 
        height=height, 
        width=width
    )

    fig.update_layout(
        title=title,
        xaxis_title=column_to_count,
        yaxis_title='Log Count'
    )

    # Apply logarithmic scale to the y-axis
    fig.update_yaxes(type="log", exponentformat='none')  

    fig.show()
    
    
    
def plot_pie_chart(df, column_name, title, height=400, width=800, hole=0.3):
    """
    Creates a pie chart for the value counts of a specified column in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to plot from.
    - column_name (str): The name of the column to plot value counts for.
    - title (str): The title of the plot.
    - height (int): The height of the plot. Default is 400.
    - width (int): The width of the plot. Default is 800.
    - hole (float): The size of the hole in the middle for a donut chart appearance. Default is 0.3.
    """
    transaction_counts = df[column_name].value_counts()

    fig = px.pie(
        names=transaction_counts.index,
        values=transaction_counts.values,
        title=title,
        height=height,
        width=width,
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=hole
    )

    # Update layout for a more polished look
    fig.update_traces(
        pull=[0.1 if i == transaction_counts.idxmax() else 0 for i in transaction_counts.index],
        textinfo='percent+label'
    )
    fig.update_layout(
        margin=dict(t=40, b=0, l=0, r=0),
        font=dict(size=12),
        legend_title_text='Transaction Type',
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    fig.show()
    

# # Example usage
# # Assuming fraud_list is your DataFrame and 'type' is the column of interest
# plot_pie_chart(df=fraud_list, column_name='type', title='Distribution of Fraud Transaction Types')



def plot_binary_distribution(df, column_name, title, color_map=None, height=600, width=800):
    """
    Creates a bar chart for the distribution of a binary label in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to plot from.
    - column_name (str): The name of the binary column to plot distribution for.
    - title (str): The title of the plot.
    - color_map (list): List of colors for the bars. Default is None.
    - height (int): The height of the plot. Default is 600.
    - width (int): The width of the plot. Default is 800.
    """
    dist = (df[column_name].value_counts(normalize=True) * 100).round(3)

    fig = go.Figure(data=[go.Bar(
        x=dist.index.astype(str),  # Convert to string for categorical values
        y=dist.values, 
        name=f'{column_name} Labels',
        text=dist.values,  # Adding text annotations
        textposition='outside',  # Position of the annotations
        marker=dict(color=color_map if color_map else ['#2ECC71', '#E74C3C'])  # Default or provided colors
    )])

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,  # Center the title
            font=dict(size=18)
        ),
        xaxis_title=f'{column_name} Label',
        xaxis=dict(
            tickmode='array',
            tickvals=dist.index,
            ticktext=dist.index.astype(str)
        ),
        yaxis_title='Percentage',
        legend_title="Legend",
        font=dict(
            family="Arial, monospace",
            size=14,
            color="#000000"
        ),
        margin=dict(l=100, r=100, t=100, b=100),
        paper_bgcolor="white",
        autosize=False,
        width=width,
        height=height
    )

    fig.show()

# # Example usage
# plot_binary_distribution(
#     df=df, 
#     column_name='isFraud', 
#     title='Fraud Label Distribution'
# )
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.stats import skew

def plot_histogram(df, column_name):
    """
    Creates a histogram for a specified column in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to plot from.
    - column_name (str): The name of the column to create a histogram for.
    - title (str): The title of the plot.
    - color (str): Color of the histogram bars.
    - height (int): The height of the plot.
    - width (int): The width of the plot.
    """
    fig = go.Figure(data=[go.Histogram(
        x=df[column_name],
        marker=dict(color='#3498DB'),
        opacity=0.7
    )])

    fig.update_layout(
        title=dict(
            text='Transaction Amount Distribution',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title=column_name,
        yaxis_title='Frequency',
        font=dict(family="Arial, monospace", size=14, color="#000000"),
        xaxis=dict(gridcolor='gray'),
        yaxis=dict(gridcolor='gray', type="log", exponentformat='none'),
        margin=dict(l=100, r=100, t=100, b=100),
        paper_bgcolor="white",
        autosize=False,
        width=800,
        height=500
    )

    fig.show()

def interpret_distribution_(data):
    """
    Generates an interpretation of the distribution based on statistical measures.

    Parameters:
    - data (array-like): The data to interpret.

    Returns:
    - str: A string containing the interpretation.
    """
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    skewness = skew(data)

    interpretation = ""

    # Shape of the distribution
    if skewness > 1:
        interpretation += "Shape: Highly skewed to the right.\n"
    elif skewness > 0.5:
        interpretation += "Shape: Moderately skewed to the right.\n"
    elif skewness < -1:
        interpretation += "Shape: Highly skewed to the left.\n"
    elif skewness < -0.5:
        interpretation += "Shape: Moderately skewed to the left.\n"
    else:
        interpretation += "Shape: Approximately symmetric.\n"

    # Central Tendency
    interpretation += f"Central Tendency: Mean = {mean:.2f}, Median = {median:.2f}.\n"

    # Spread of the distribution
    interpretation += f"Spread: Standard Deviation = {std_dev:.2f}.\n"

    # Tails
    tail = "right" if skewness > 0 else "left"
    interpretation += f"Tail: Longer tail on the {tail} side.\n"

    return interpretation

# # Example usage
# data = df['amount']
# plot_histogram(df=df, column_name='amount', title='Transaction Amount Distribution')

# # Generate interpretation

# print("Interpretation of Histogram:")
# print(interpret_distribution(data))
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_overlaid_histograms(df, column_name, class_column, title, colors, height=600, width=800):
    """
    Creates overlaid histograms for different classes in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to plot from.
    - column_name (str): The name of the column to create histograms for.
    - class_column (str): The name of the column that contains class labels.
    - title (str): The title of the plot.
    - colors (dict): A dictionary of colors for each class.
    - height (int): The height of the plot.
    - width (int): The width of the plot.
    """
    fig = go.Figure()

    for class_label in df[class_column].unique():
        fig.add_trace(go.Histogram(
            x=df[df[class_column] == class_label][column_name],
            name=f'Class {class_label}',
            marker=dict(color=colors[class_label]),
            opacity=0.7
        ))

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title=column_name,
        yaxis_title='Frequency',
        font=dict(family="Arial, monospace", size=14, color="#000000"),
        xaxis=dict(gridcolor='gray'),
        yaxis=dict(gridcolor='gray', type='log'),
        barmode='overlay',
        margin=dict(l=100, r=100, t=100, b=100),
        paper_bgcolor="white",
        autosize=False,
        width=width,
        height=height
    )

    fig.show()

# # Example usage
# colors = {0: '#3498DB', 1: '#E74C3C'}
# plot_overlaid_histograms(df=df, column_name='amount', class_column='isFraud', title='Transaction Amount Distribution by Class', colors=colors)
from scipy.stats import skew

def interpret_distributions(df, column_name, class_column):
    """
    Generates interpretations of distributions for different classes in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to interpret.
    - column_name (str): The name of the column to interpret.
    - class_column (str): The name of the column that contains class labels.
    """
    interpretations = []

    for class_label in df[class_column].unique():
        data = df[df[class_column] == class_label][column_name]
        mean = np.mean(data)
        median = np.median(data)
        std_dev = np.std(data)
        skewness = skew(data)

        interpretation = f"Statistics for Class {class_label}:\n"
        interpretation += f"Shape: {'Highly skewed to the right' if skewness > 1 else 'Moderately skewed to the right' if skewness > 0.5 else 'Highly skewed to the left' if skewness < -1 else 'Moderately skewed to the left' if skewness < -0.5 else 'Approximately symmetric'}.\n"
        interpretation += f"Central Tendency: Mean = {mean:.2f}, Median = {median:.2f}\n"
        interpretation += f"Spread: Standard Deviation = {std_dev:.2f}\n"
        interpretation += "Peaks: Advanced analysis like kernel density estimation required for detailed peak analysis.\n"
        interpretation += f"Tail: {'Longer tail on the right side, indicating positive skewness' if skewness > 0 else 'Longer tail on the left side, indicating negative skewness'}.\n"

        interpretations.append(interpretation)

    return "\n".join(interpretations)

# # Generate and print interpretations
# print(interpret_distributions(df=df, column_name='amount', class_column='isFraud'))



def data_encoding(df, column1, column2):
    """
    Apply label encoding to two specified columns of a pandas DataFrame.

    Label encoding converts categorical text data into a numerical format
    that is understandable by machine learning models. This function takes
    a DataFrame and two column names and applies label encoding to these
    columns independently.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be encoded.
    - column1 (str): The name of the first column to apply label encoding.
    - column2 (str): The name of the second column to apply label encoding.

    Returns:
    - pandas.DataFrame: The DataFrame with the specified columns encoded.

    Example:
    >>> encoded_df = data_encoding(df, 'category1', 'category2')
    """

    label_encoder = LabelEncoder()
    # Encoding the first column
    df[column1] = label_encoder.fit_transform(df[column1])
    # Encoding the second column
    df[column2] = label_encoder.fit_transform(df[column2])
    
    return df



def test_data_encoding(df, column1, column2):
    """
    Test the data_encoding function by applying it to specified columns and
    checking if the encoding is performed correctly.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be tested.
    - column1 (str): The name of the first column to test label encoding.
    - column2 (str): The name of the second column to test label encoding.

    Returns:
    - None: The function prints a message indicating whether all tests have passed or a specific assertion has failed.
    """
    # Create a copy of the DataFrame for comparison
    original_df = df.copy()

    # Apply the encoding function
    encoded_df = data_encoding(df.copy(), column1, column2)

    # Perform assertions
    try:
        assert original_df[column1].tolist() != encoded_df[column1].tolist(), "Column1 encoding failed"
        assert original_df[column2].tolist() != encoded_df[column2].tolist(), "Column2 encoding failed"
        assert isinstance(encoded_df[column1][0], int), "Encoded Column1 is not integer"
        assert isinstance(encoded_df[column2][0], int), "Encoded Column2 is not integer"
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

# Example usage:
# Assuming df is your DataFrame and 'category1', 'category2' are the columns of interest
#test_data_encoding(df, 'category1', 'category2')

import pandas as pd
import numpy as np

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

    return df



import pandas as pd

def trim_data(df, target_column, target_size, random_state=42):
    """
    Trims the majority class in a DataFrame to a specified size.

    Parameters:
    df (pd.DataFrame): The original DataFrame.
    target_column (str): The name of the column indicating the class.
    target_size (int): The desired size of the majority class after trimming.
    random_state (int): A seed for the random number generator for reproducibility.

    Returns:
    pd.DataFrame: A new DataFrame with the majority class trimmed.
    """
    # Separate the dataset into two based on the target column
    df_minority = df[df[target_column] == 1]
    df_majority = df[df[target_column] == 0]

    # Randomly sample from the majority class
    df_majority_sampled = df_majority.sample(n=target_size, random_state=random_state)

    # Concatenate the minority class with the trimmed majority class
    df_reduced = pd.concat([df_minority, df_majority_sampled])

    # Shuffle the dataset
    df_reduced = df_reduced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_reduced

import pandas as pd


def filter_fraud_data(df):
    """
    Filters the DataFrame to include only rows with 'type' values 'CASH_OUT' or 'TRANSFER'.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame.

    Returns:
    - pandas.DataFrame: DataFrame containing only rows with 'type' values 'CASH_OUT' or 'TRANSFER'.
    """
    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])]
    return df



def transform_transaction_type(df, column_name):
    """
    Transforms the values in the specified column of a DataFrame.
    Values 'TRANSFER' and 'CASH_OUT' are set to 1, and all other values are set to 0.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the column to be transformed.
    - column_name (str): The name of the column to apply the transformation to.

    Returns:
    - pandas.DataFrame: The DataFrame with the transformed column.
    """
    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].apply(lambda x: 1 if x in ['TRANSFER', 'CASH_OUT'] else 0)
    return df_copy

# Example usage:
# Assuming df is your original DataFrame with a 'type' column
#transformed_df = transform_transaction_type(df, 'type')



def plot_anova_feature_scores(df, feature_cols, target_col):
    """
    Performs ANOVA F-test for feature selection and visualizes the scores of each feature using a heatmap.
    Outputs a DataFrame of features and their corresponding ANOVA scores.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the features and target.
    - feature_cols (list): List of column names to be used as features.
    - target_col (str): Column name of the target variable.

    Returns:
    - pandas.DataFrame: A DataFrame containing the features and their ANOVA scores.
    """
    features = df[feature_cols]
    target = df[target_col]

    # Perform ANOVA F-test for feature selection
    anova_selector = SelectKBest(score_func=f_classif, k='all')
    fit = anova_selector.fit(features, target)

    # Create DataFrame for feature scores
    feature_scores = pd.DataFrame({'Feature': feature_cols, 'ANOVA Score': fit.scores_})
    feature_scores = feature_scores.sort_values(by='ANOVA Score', ascending=False)

    # Create heatmap visualization of feature scores
    plt.figure(figsize=(12, 6))
    sns.heatmap(feature_scores.set_index('Feature').T, annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".2f")
    plt.title('ANOVA F-test Scores for Features')

    # Display the plot
    plt.show()

    return feature_scores

# Example usage
# df_fraud = # assuming df_fraud is your DataFrame
# feature_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'transaction_velocity', 'amount_deviation', 'balance_change_orig', 'nameOrig_frequency', 'origzeroFlag']
# target_col = 'isFraud'

# # Get and display feature scores
# feature_scores_df = plot_anova_feature_scores(df_fraud, feature_cols, target_col)
# print(feature_scores_df)


def find_highly_correlated_features(df, threshold=0.7):
    """
    Identifies and returns pairs of highly correlated features in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing features to be analyzed.
    - threshold (float): The threshold for considering features as highly correlated.

    Returns:
    - dict: A dictionary where keys are features and values are lists of features that are highly correlated with them.
    """
    correlation_matrix = df.corr()
    high_correlation = correlation_matrix[(correlation_matrix >= threshold) & (correlation_matrix < 1)]
    correlated_pairs = {}

    for column in high_correlation:
        correlated_features = high_correlation[column].dropna().index.tolist()
        if correlated_features:
            correlated_pairs[column] = correlated_features

    return correlated_pairs

# # Example usage:
# df_fraud = # assuming df_fraud is your DataFrame
# threshold = 0.7
# correlated_features = find_highly_correlated_features(df_fraud, threshold)

# # Print out the highly correlated feature pairs
# for feature, correlations in correlated_features.items():
#     print(f"{feature} is highly correlated with {correlations}")



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Function to safely create heatmaps
def safe_heatmap(data, title, ax, colors, **kwargs):
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes)
        ax.set_title(title)
    else:
        sns.heatmap(data, ax=ax, cmap=colors, **kwargs)
        ax.set_title(title)

# Function to create and display descriptive statistics heatmaps
def plot_descriptive_stats_heatmaps(df, target_col):
    # Compute descriptive statistics
    fraud_stats = df[df[target_col] == 1].describe().T
    nofraud_stats = df[df[target_col] == 0].describe().T

    # Define the color palette
    colors = ['#FFD700', '#3B3B3C']

    # Create figure and axes
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 10))

    # Plot heatmaps
    safe_heatmap(fraud_stats[['mean']][:6], 'Fraud Samples: Part 1', ax=ax[0, 0], colors=colors, annot=True, linewidths=0.5, linecolor='black', cbar=False, fmt='.2f')
    safe_heatmap(fraud_stats[['mean']][6:12], 'Fraud Samples: Part 2', ax=ax[0, 1], colors=colors, annot=True, linewidths=0.5, linecolor='black', cbar=False, fmt='.2f')
    safe_heatmap(nofraud_stats[['mean']][:6], 'No Fraud Samples: Part 1', ax=ax[1, 0], colors=colors, annot=True, linewidths=0.5, linecolor='black', cbar=False, fmt='.2f')
    safe_heatmap(nofraud_stats[['mean']][6:12], 'No Fraud Samples: Part 2', ax=ax[1, 1], colors=colors, annot=True, linewidths=0.5, linecolor='black', cbar=False, fmt='.2f')

    # Adjust layout
    fig.tight_layout(w_pad=2)

    # Display the plot
    plt.show()

# Usage example
# plot_descriptive_stats_heatmaps(df_selected, 'isFraud')



# Function to perform feature selection and plot results
def feature_selection_and_plot(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate SelectKBest with mutual_info_classif as the scoring function
    selector = SelectKBest(score_func=mutual_info_classif, k='all')

    # Fit and transform on the training data
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Get the scores for each feature
    feature_scores = selector.scores_

    # Create a DataFrame with feature names and their scores
    feature_scores_df = pd.DataFrame({'Feature': X.columns, 'Score': feature_scores})

    # Sort the DataFrame by scores in descending order
    sorted_feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    # Plotting using Plotly Express
    fig = px.bar(sorted_feature_scores_df, x='Feature', y='Score', title='Feature Importance')

    return sorted_feature_scores_df, fig

# Using the function
# df_scores, plot_figure = feature_selection_and_plot(df_fraud, 'isFraud')

# # Displaying the DataFrame
# df_scores



def drop_columns(df, columns_to_drop):
    """
    Drops specified columns from a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame from which columns are to be dropped.
    - columns_to_drop (list): A list of column names to be dropped from the DataFrame.

    Returns:
    - DataFrame: A new DataFrame with the specified columns removed.
    """
    # Ensure that columns_to_drop is a list
    if not isinstance(columns_to_drop, list):
        raise TypeError("columns_to_drop should be a list of column names")

    # Drop the specified columns and return the new DataFrame
    df = df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
    return df

# Example usage:
# new_df = drop_columns(df, columns_to_remove)



def plot_trx_histogram(df, x_column, y_column):
    """
    Plots a histogram without log-transforming the y-axis.

    Parameters:
    - df (DataFrame): The DataFrame containing the data to plot.
    - x_column (str): The name of the column to use for the x-axis.
    - y_column (str): The name of the column to use for the y-axis.
    - title (str): The title of the histogram. Default is 'Log Distribution of Transactions at Different Times'.
    - height (int): The height of the plot in pixels. Default is 500.
    - width (int): The width of the plot in pixels. Default is 800.

    Returns:
    - A Plotly Express histogram figure.
    """
    # Check if columns exist in the DataFrame
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError("Specified columns are not in the DataFrame")

    # Plotting a histogram without log-transformed y-axis
    fig = px.histogram(df, x=x_column, y=y_column, title='Distribution of Transactions at Different Times', height=500, width=800)

    return fig

# Example usage:
# histogram_fig = plot_log_transformed_histogram(df, 'step', 'amount', 'Log Distribution of Transactions at Different Times', 400, 800)
# histogram_fig.show()


import pandas as pd
import plotly.express as px

def plot_fraud_class_histogram(df, x_column, y_column):
    """
    Plots a histogram without log-transforming the y-axis, categorized by fraud class.

    Parameters:
    - df (DataFrame): The DataFrame containing the data to plot.
    - x_column (str): The name of the column to use for the x-axis.
    - y_column (str): The name of the column to use for the y-axis.
    - title (str): The title of the histogram. Default is 'Distribution of Transactions at Different Times by Fraud Class'.
    - height (int): The height of the plot in pixels. Default is 500.
    - width (int): The width of the plot in pixels. Default is 800.

    Returns:
    - A Plotly Express histogram figure.
    """
    # Check if columns exist in the DataFrame
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError("Specified columns are not in the DataFrame")

    # Define a custom color palette for the 'isFraud' classes
    color_discrete_map = {0: 'blue', 1: 'red'}

    # Plotting a histogram without log-transformed y-axis
    fig = px.histogram(df, x=x_column, y=y_column, color='isFraud', barmode='overlay',
                       title='Distribution of Transactions at Different Times by Fraud Class', color_discrete_map=color_discrete_map,
                       height=500, width=800)

    return fig

# Example usage:
# df_fraud = pd.DataFrame(...)  # Replace with your actual DataFrame
# histogram_fig = plot_fraud_class_histogram(df_fraud, 'step', 'amount', '', 400, 800)
# histogram_fig.show()


def plot_fraud_class_histogram_2(df, x_column, y_column, title='Distribution of Transactions for Fraudulent Class', height=500, width=800):
    """
    Plots a histogram without log-transforming the y-axis for the fraudulent class (isFraud == 1).

    Parameters:
    - df (DataFrame): The DataFrame containing the data to plot.
    - x_column (str): The name of the column to use for the x-axis.
    - y_column (str): The name of the column to use for the y-axis.
    - title (str): The title of the histogram. Default is 'Distribution of Transactions for Fraudulent Class'.
    - height (int): The height of the plot in pixels. Default is 500.
    - width (int): The width of the plot in pixels. Default is 800.

    Returns:
    - A Plotly Express histogram figure.
    """
    # Check if columns exist in the DataFrame
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError("Specified columns are not in the DataFrame")

    # Filter the DataFrame to include only rows where 'isFraud' is 1
    fraud_df = df[df['isFraud'] == 1]

    # Plotting a histogram without log-transformed y-axis for isFraud == 1
    fig = px.histogram(fraud_df, x=x_column, y=y_column,
                       title=title, height=height, width=width, color_discrete_sequence=['red'])

    return fig


# In[ ]:




