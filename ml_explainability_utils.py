#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dice_ml
from dice_ml.utils import helpers 
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
import shap
import numpy as np
import re
import plotly.figure_factory as ff
import eli5
import seaborn as sns
import matplotlib.pyplot as plt
# from IPython.display import display, HTML
import networkx as nx
import community as community_louvain
from matplotlib import patheffects, colors, cm
import plotly.graph_objects as go
import matplotlib.cm as cm
import numpy as np



class LIMEModelInterpreter:
    def __init__(self, pipeline, X_train, live_instance, model_name):
        self.pipeline = pipeline
        self.X_train = X_train
        self.model_name = model_name
        self.live_instance = live_instance
        self.features = self.live_instance.columns.tolist()

    def limeExplainer_live(self):

        def pipeline_predict(data):
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=self.features)
            return self.pipeline.predict_proba(data)

        X_train_np = self.X_train.to_numpy()
        type_column_index = self.X_train.columns.get_loc('type')

        explainer_lime = LimeTabularExplainer(
            training_data=X_train_np,
            feature_names=self.features,
            categorical_features=[type_column_index],
            class_names=['Legitimate', 'Fraudulent'],
            verbose=True,
            mode='classification',
            kernel_width=3
        )

        live_instance_np = self.live_instance.iloc[0].to_numpy() if isinstance(self.live_instance, pd.DataFrame) else self.live_instance.to_numpy()
        prediction = self.pipeline.predict([live_instance_np])[0]

        lime_explanation = explainer_lime.explain_instance(live_instance_np, pipeline_predict, num_features=len(self.features))
        return lime_explanation, prediction

    @staticmethod
    def lime_feature_extraction(lime_explanation):
        explanation_data = {feature: weight for feature, weight in lime_explanation.as_list()}
        non_feature_keys = ['Intercept', 'Prediction_local', 'Right']
        for non_feature_key in non_feature_keys:
            explanation_data.pop(non_feature_key, None)
        feature_explanations = list(explanation_data.items())
        return feature_explanations

    def lime_explanation_visualisation(self, lime_explanation, prediction):
        print(f"LIME Explanation for {self.model_name} on the predicted instance (Predicted as: {'Legitimate' if prediction == 0 else 'Fraudulent'})")
        lime_explanation.show_in_notebook(show_table=True, show_all=False)

    
    @staticmethod
    def interpret_lime_results(feature_explanations, prediction, num_features=8):
        class_map = {0: 'Legitimate', 1: 'Fraudulent'}
        predicted_class = class_map.get(prediction, 'Unknown')

        explanations = feature_explanations[:num_features]

        output_lines = []
        increase_fraudulent, increase_legitimate, no_effect_count = 0, 0, 0
        no_effect_features = []

        for feature, effect in explanations:
            if effect == 0:
                no_effect_count += 1
                no_effect_features.append(feature.split(' ')[0])
                continue  # Skip the rest of the loop for features with no effect

            influence_class = 'Fraudulent' if effect > 0 else 'Legitimate'
            if effect > 0:
                increase_fraudulent += 1
            else:
                increase_legitimate += 1

            # Format feature explanation
            if '<=' in feature or '>' in feature:
                condition, value = feature.replace(' ', '').split('<=') if '<=' in feature else feature.replace(' ', '').split('>')
                explanation = f"'{condition}' is {'not more' if '<=' in feature else 'more'} than {value}"
            else:
                explanation = f"based on the feature '{feature}'"
            output_lines.append(f"The model's prediction likelihood of being classified as {influence_class} increases because {explanation}.")

        # Calculate statistics about feature influence
        total = len(explanations) - no_effect_count
        increase_fraudulent_percent = (increase_fraudulent / total) * 100 if total > 0 else 0
        increase_legitimate_percent = (increase_legitimate / total) * 100 if total > 0 else 0
        no_effect_percent = (no_effect_count / num_features) * 100 if num_features > 0 else 0

        output_lines.append(f"\nPercentage of features that increase likelihood of 'Fraudulent': {increase_fraudulent_percent:.2f}%")
        output_lines.append(f"Percentage of features that increase likelihood of 'Legitimate': {increase_legitimate_percent:.2f}%")
        output_lines.append(f"Percentage of features with no effect: {no_effect_percent:.2f}%")

        if no_effect_features:
            output_lines.append(f"Features with no effect on the prediction: {', '.join(no_effect_features)}")
        else:
            output_lines.append("There are no features with no effect on the prediction.")

        # Join all lines into a single string with new lines
        summary_text = "\n".join(output_lines)
        return summary_text
           
    
    @staticmethod
    def plot_lime_explanation(feature_explanations):
        def extract_string(feature):
            return ' '.join(re.findall(r'[^\d<>=.]+', feature)).strip()

        feature_name, importance = zip(*feature_explanations)
        df = pd.DataFrame({'Features': feature_name, 'Importances': importance})
        df['Feature_Names'] = df['Features'].apply(extract_string)
        df = df.sort_values(by='Importances')

        fig = px.bar(df, x='Importances', y='Feature_Names', orientation='h',
                     labels={'x': 'Feature Importance', 'y': 'Features'},
                     title='LIME Plot')
        # Assign colors based on the sign of the importance values
        fig.data[0].marker.color = ['orange' if imp > 0 else 'blue' for imp in df['Importances']]

        # Add annotations for color significance
        fig.add_annotation(x=max(df['Importances'])/2, y=len(feature_explanations) + 1,
                           text="Fraudulent(Orange)", showarrow=False,
                           font=dict(color='orange'))
        fig.add_annotation(x=min(df['Importances'])/2, y=len(feature_explanations) + 1,
                           text="Legitimate(Blue)", showarrow=False,
                           font=dict(color='blue'))

        # Adjust layout
        fig.update_layout(showlegend=False, margin=dict(t=40, b=40, l=200, r=20),
                          yaxis=dict(title='Features'), xaxis=dict(title='Weight')
                          )#,width=600, height=600

        return fig 


    @staticmethod
    def calculate_and_print_feature_influence(feature_explanations):
        def extract_string(feature):
            # Extracts readable string from feature name
            return ' '.join(re.findall(r'[^\d<>=.]+', feature)).strip()

        # Unpack feature names and their importances
        feature_name, importance = zip(*feature_explanations)
        df = pd.DataFrame({'Features': feature_name, 'Importances': importance})

        # Apply the string extraction function to each feature name
        df['Feature_Names'] = df['Features'].apply(extract_string)
        total_sum = df['Importances'].abs().sum()

        output_lines = []  # Initialize a list to hold output lines

        if total_sum > 0:
            # Calculate the influence percentage
            df['Influence_Percent'] = (df['Importances'] / total_sum) * 100
            for index, row in df.iterrows():
                influence_description = "towards Fraudulent Class" if row['Importances'] >= 0 else "away from Fraudulent to Legitimate Class"
                output_lines.append(f"{row['Feature_Names']} pushed this prediction outcome {influence_description} by {round(row['Influence_Percent'], 1)}%")
        else:
            output_lines.append("No significant features found.")

        # Join all lines into a single string with new lines
        summary_text = "\n".join(output_lines)
        return summary_text
            
            


class ModelExplainer:
    def __init__(self, pipeline, live_df):
        self.pipeline = pipeline
        self.live_df = live_df

    def shap_interactions(self):
        features = self.live_df.columns.tolist()
        live_data_instance = self.live_df.iloc[0]
        live_data_instance_df = pd.DataFrame([live_data_instance], columns=features)
        model = self.pipeline.named_steps['model']
        scaler = self.pipeline.named_steps['scaler']
        live_data_instance_scaled = scaler.transform(live_data_instance_df)
        df_live = pd.DataFrame(live_data_instance_scaled, columns=features)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_live.iloc[0])
        predicted_class = model.predict(df_live)[0]
        shap_interaction_values = explainer.shap_interaction_values(df_live.iloc[0])
        interaction_values = shap_interaction_values[predicted_class]
        interaction_df = pd.DataFrame(interaction_values, columns=features, index=features)
        return interaction_df

    def features_interaction(self):
        interaction_df = self.shap_interactions()
        main_effects = np.diag(interaction_df.values)
        main_effects_series = pd.Series(main_effects, index=interaction_df.columns)
        top_main_effects = main_effects_series.abs().sort_values(ascending=False).head(3)
        flat_interactions = interaction_df.mask(np.eye(len(interaction_df), dtype=bool)).stack()
        top_interactions = flat_interactions.abs().sort_values(ascending=False).head(4)
        top_main_effect = f"\nTop 3 Features with Strongest Impact:\n{top_main_effects}\n"
        top_interaction = f"Top 4 Feature Interactions:\n{top_interactions}\n"
        return top_main_effect, top_interaction


        # Format the top interactions output
        top_interaction_output = "Top 4 Feature Interactions:\n"
        for (index1, index2), value in top_interactions.iteritems():
            top_interaction_output += f"{index1} & {index2}: {value:.6f}\n"

        return top_main_effect_output.strip(), top_interaction_output.strip()

    def features_interaction_v2(self):
        interaction_df = self.shap_interactions()
        main_effects = np.diag(interaction_df.values)
        main_effects_series = pd.Series(main_effects, index=interaction_df.columns)
        top_main_effects = main_effects_series.abs().sort_values(ascending=False).head(4)
        flat_interactions = interaction_df.mask(np.eye(len(interaction_df), dtype=bool)).stack()
        top_interactions = flat_interactions.abs().sort_values(ascending=False).head(4)

        # Format the top main effects output
        top_main_effect_output = "Top 4 Features with Strongest Impact:\n"
        for i, (index, value) in enumerate(top_main_effects.iteritems(), start=1):
            top_main_effect_output += f"{i}. {index}: Impact Value = {value:.6f}\n"

        # Format the top interactions output
        top_interaction_output = "Top 4 Feature Interactions:\n"
        for i, ((index1, index2), value) in enumerate(top_interactions.iteritems(), start=1):
            top_interaction_output += f"{i}. {index1} & {index2}: Interaction Impact = {value:.6f}\n"

        return top_main_effect_output.strip(), top_interaction_output.strip()

    
    
        
    def prepare_interaction_data_for_graph(self):
        interaction_df = self.shap_interactions()
        # Use only the lower triangle of the interaction matrix, as it is symmetric
        lower_triangle_mask = np.tril(np.ones(interaction_df.shape)).astype(bool)
        interaction_data = interaction_df.where(lower_triangle_mask)

        # Convert to a long format suitable for network graph
        interaction_data_long = interaction_data.stack().reset_index()
        interaction_data_long.columns = ['Feature1', 'Feature2', 'Interaction']

        # Filter out zero interactions to reduce graph complexity
        interaction_data_long = interaction_data_long[interaction_data_long['Interaction'] != 0]

        return interaction_data_long        
        
        
    def features_interaction_gbc(self):
        interaction_df = self.shap_interactions_gbc()
        if interaction_df.shape[0] != interaction_df.shape[1]:
            raise ValueError("interaction_df must be a square DataFrame.")
        main_effects = np.diag(interaction_df.values)
        main_effects_series = pd.Series(main_effects, index=interaction_df.columns)
        top_main_effects = main_effects_series.abs().sort_values(ascending=False).head(4)
        flat_interactions = interaction_df.where(~np.eye(len(interaction_df), dtype=bool)).stack()
        top_interactions = flat_interactions.abs().sort_values(ascending=False).head(4)
        top_main_effect = f"\nTop 4 Features with Strongest Impact:\n{top_main_effects}\n"
        top_interaction = f"Top 4 Feature Interactions:\n{top_interactions}\n"
        return top_main_effect, top_interaction

    def features_interaction_gbc_v2(self):
        interaction_df = self.shap_interactions_gbc()
        main_effects = np.diag(interaction_df.values)
        main_effects_series = pd.Series(main_effects, index=interaction_df.columns)
        top_main_effects = main_effects_series.abs().sort_values(ascending=False).head(4)
        flat_interactions = interaction_df.mask(np.eye(len(interaction_df), dtype=bool)).stack()
        top_interactions = flat_interactions.abs().sort_values(ascending=False).head(4)

        # Format the top main effects output
        top_main_effect_output = "Top 4 Features with Strongest Impact:\n"
        for i, (index, value) in enumerate(top_main_effects.iteritems(), start=1):
            top_main_effect_output += f"{i}. {index}: Impact Value = {value:.6f}\n"

        # Format the top interactions output
        top_interaction_output = "Top 4 Feature Interactions:\n"
        for i, ((index1, index2), value) in enumerate(top_interactions.iteritems(), start=1):
            top_interaction_output += f"{i}. {index1} & {index2}: Interaction Impact = {value:.6f}\n"

        return top_main_effect_output.strip(), top_interaction_output.strip()
    
    

    def shap_interactions_gbc(self):
        features = self.live_df.columns.tolist()
        live_data_instance = self.live_df.iloc[0]
        live_data_instance_df = pd.DataFrame([live_data_instance], columns=features)
        model = self.pipeline.named_steps['model']
        scaler = self.pipeline.named_steps['scaler']
        live_data_instance_scaled = scaler.transform(live_data_instance_df)
        df_live = pd.DataFrame(live_data_instance_scaled, columns=features)
        explainer = shap.TreeExplainer(model)
        shap_interaction_values = explainer.shap_interaction_values(df_live.iloc[0])
        predicted_class = model.predict(df_live)[0]
        if shap_interaction_values.ndim == 3:
            interaction_values = shap_interaction_values[predicted_class]
        elif shap_interaction_values.ndim == 2:
            interaction_values = shap_interaction_values
        if interaction_values.shape[1] == 1:
            interaction_values = np.tile(interaction_values, (1, len(features)))
        interaction_df = pd.DataFrame(interaction_values, columns=features, index=features)
        return interaction_df



    def explain_prediction(self):
        feature_names = self.live_df.columns.tolist()
        live_data_instance = self.live_df[feature_names].iloc[0]
        scaler = self.pipeline.named_steps['scaler']
        live_data_instance_scaled = scaler.transform([live_data_instance])
        model = self.pipeline.named_steps['model']
        predict_class = model.predict(live_data_instance_scaled)[0]
        if predict_class == 1:
            prediction = "Fraudulent"
        else:
            prediction = "Legitimate"
        explanation = eli5.explain_prediction(model, live_data_instance_scaled[0], feature_names=feature_names)
        interpretation = self.interpret_eli5_explanation(explanation,feature_names, prediction)
        return interpretation
    
                 
        
    def prepare_interaction_data_for_graph_gbc(self):
        interaction_df = self.shap_interactions_gbc()
        # Use only the lower triangle of the interaction matrix, as it is symmetric
        lower_triangle_mask = np.tril(np.ones(interaction_df.shape)).astype(bool)
        interaction_data = interaction_df.where(lower_triangle_mask)

        # Convert to a long format suitable for network graph
        interaction_data_long = interaction_data.stack().reset_index()
        interaction_data_long.columns = ['Feature1', 'Feature2', 'Interaction']

        # Filter out zero interactions to reduce graph complexity
        interaction_data_long = interaction_data_long[interaction_data_long['Interaction'] != 0]

        return interaction_data_long




    def create_network_graph_with_communities(self):
        interaction_data_long = self.prepare_interaction_data_for_graph()

        G = nx.Graph()
        for _, row in interaction_data_long.iterrows():
            G.add_edge(row['Feature1'], row['Feature2'], weight=abs(row['Interaction']))

        partition = community_louvain.best_partition(G)
        pos = nx.spring_layout(G)

        community_cmap = cm.coolwarm
        community_colors = [community_cmap(i / max(partition.values())) for i in partition.values()]

        fig, ax = plt.subplots(figsize=(11, 8))
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=community_colors, node_size=500)

        edges = G.edges(data=True)
        weights = [edata['weight'] for _, _, edata in edges]
        # Calculate the maximum weight for normalizing edge widths
        max_weight = max(weights) if weights else 1  # Avoid division by zero if weights list is empty

        edge_norm = colors.Normalize(vmin=min(weights), vmax=max(weights))
        edge_cmap = cm.plasma
        # Adjust the calculation for edge width based on max_weight
        for (u, v, edata), width in zip(edges, [edata['weight'] / max_weight * 10 for _, _, edata in edges]):
            edge_color = edge_cmap(edge_norm(edata['weight']))
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, ax=ax, edge_color=edge_color, alpha=0.7)

        for node, (x, y) in pos.items():
            text = ax.text(x, y, node, ha='center', va='center', fontdict={'color': 'black', 'size': 8})
            text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])

        sm = cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Interaction Strength')

        community_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f"Community {i+1}",
                                       markerfacecolor=community_cmap(i / max(partition.values())), markersize=15)
                            for i in range(max(partition.values()) + 1)]
        ax.legend(handles=community_legend, title='Communities', loc='upper left', bbox_to_anchor=(1, 1.1))

        ax.set_title("Feature Interactions with Community Detection - Network Graph")
        plt.axis('off')

        return fig, ax  
    
 
    def network_graph_interaction_strength(self):
        
        interaction_data_long = self.prepare_interaction_data_for_graph()
        G = nx.Graph()

        # Add edges to the graph with interaction strength as weight
        for _, row in interaction_data_long.iterrows():
            G.add_edge(row['Feature1'], row['Feature2'], weight=abs(row['Interaction']))

        pos = nx.spring_layout(G)

        # Calculate interaction strength for each node
        node_strength = {node: np.sum([data['weight'] for _, _, data in G.edges(node, data=True)]) for node in G.nodes()}
        # Normalize the strengths for color mapping
        min_strength, max_strength = min(node_strength.values()), max(node_strength.values())
        strength_norm = {node: (strength - min_strength) / (max_strength - min_strength) for node, strength in node_strength.items()}

        # Edge Traces
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']

            edge_width = weight * 12  # Edge width based on weight
            edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                         line=dict(width=edge_width, color='rgba(0,0,0,1)'),
                                         hoverinfo='none', mode='lines'))

        # Node Trace
        node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
                                marker=dict(showscale=True, colorscale='plasma',
                                            color=[strength_norm[node] for node in G.nodes()],
                                            size=[len(G[node]) * 6 for node in G.nodes()],  
                                            colorbar=dict(thickness=15, title='Interaction Strength (scaled)', xanchor='left', 
                                                          titleside='right')))
       


        # Add node positions to the trace
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([f'{node}'])

        fig = go.Figure(data=edge_trace + [node_trace],
                        layout=go.Layout(title='Features Interactions  with Network Graph',
                                         showlegend=False,
                                         hovermode='closest',
#                                          height=600,
#                                          width=600,                                     
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        return fig
   
 

    def create_network_graph_with_communities_gbc(self):
        interaction_data_long = self.prepare_interaction_data_for_graph_gbc()

        G = nx.Graph()
        for _, row in interaction_data_long.iterrows():
            G.add_edge(row['Feature1'], row['Feature2'], weight=abs(row['Interaction']))

        partition = community_louvain.best_partition(G)
        pos = nx.spring_layout(G)  # positions for all nodes

        community_cmap = cm.coolwarm
        community_colors = [community_cmap(i / max(partition.values())) for i in partition.values()]

        fig, ax = plt.subplots(figsize=(11, 8))
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=community_colors, node_size=500)

        edges = G.edges(data=True)
        weights = [edata['weight'] for _, _, edata in edges]
        # Ensure there's at least one weight to avoid division by zero
        max_weight = max(weights) if weights else 1

        edge_norm = colors.Normalize(vmin=min(weights), vmax=max(weights))
        edge_cmap = cm.plasma  # Using a different colormap for edge strengths
        for (u, v, edata), width in zip(edges, [edata['weight'] / max_weight * 10 for _, _, edata in edges]):
            edge_color = edge_cmap(edge_norm(edata['weight']))
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, ax=ax, edge_color=edge_color, alpha=0.7)

        for node, (x, y) in pos.items():
            text = ax.text(x, y, node, ha='center', va='center', fontdict={'color': 'black', 'size': 8})
            text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])

        sm = cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Interaction Strength')

        community_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f"Community {i+1}",
                                       markerfacecolor=community_cmap(i / max(partition.values())), markersize=15)
                            for i in range(max(partition.values()) + 1)]
        ax.legend(handles=community_legend, title='Communities', loc='upper left', bbox_to_anchor=(1, 1.1), borderaxespad=0.)

        ax.set_title("Feature Interactions with Community Detection - Network Graph")

        plt.axis('off')

        return fig, ax 
    
    
    def network_graph_interaction_strength_gbc(self):
        
        interaction_data_long = self.prepare_interaction_data_for_graph_gbc()
        G = nx.Graph()

        # Add edges to the graph with interaction strength as weight
        for _, row in interaction_data_long.iterrows():
            G.add_edge(row['Feature1'], row['Feature2'], weight=abs(row['Interaction']))

        pos = nx.spring_layout(G)

        # Calculate interaction strength for each node
        node_strength = {node: np.sum([data['weight'] for _, _, data in G.edges(node, data=True)]) for node in G.nodes()}
        # Normalize the strengths for color mapping
        min_strength, max_strength = min(node_strength.values()), max(node_strength.values())
        strength_norm = {node: (strength - min_strength) / (max_strength - min_strength) for node, strength in node_strength.items()}

        # Edge Traces
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']

            edge_width = weight * 12  # Edge width based on weight
            edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                         line=dict(width=edge_width, color='rgba(0,0,0,1)'),
                                         hoverinfo='none', mode='lines'))

        # Node Trace
        node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
                                marker=dict(showscale=True, colorscale='plasma',
                                            color=[strength_norm[node] for node in G.nodes()],
                                            size=[len(G[node]) * 6 for node in G.nodes()],  
                                            colorbar=dict(thickness=15, title='Interaction Strength (scaled)', xanchor='left', 
                                                          titleside='right')))




        # Add node positions to the trace
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([f'{node}'])

        fig = go.Figure(data=edge_trace + [node_trace],
                        layout=go.Layout(title='Features Interactions  with Network Graph',
                                         showlegend=False,
                                         hovermode='closest',
#                                          height=700,  
#                                          width=1000,                                     
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        return fig
    
    
        
       
        
class SHAPVisualizer:
    def __init__(self, pipeline, live_instance, X_train):
        self.pipeline = pipeline
        self.live_instance = live_instance
        self.features = X_train.columns.tolist()
        self.scaler = pipeline.named_steps['scaler']
        self.model = pipeline.named_steps['model']
        self.background = shap.sample(pd.DataFrame(self.scaler.transform(X_train), columns=self.features), 100)
        self.explainer = shap.KernelExplainer(self.model.predict, data=self.background)  

    def _prepare_instance(self):
        live_data_instance = self.live_instance.iloc[0]
        return self.scaler.transform([live_data_instance])


    def shap_exp_plot(self):
        live_data_instance_scaled = self._prepare_instance()
        shap_values = self.explainer.shap_values(live_data_instance_scaled)

        shap_values_1d = shap_values[0]  # Assuming a single output model
        sorted_features = sorted(zip(self.features, shap_values_1d), key=lambda x: x[1])
        features, values = zip(*sorted_features)
        colors = ['blue' if val < 0 else 'red' for val in values]

        # This creates a horizontal bar chart
        fig = go.Figure(data=[go.Bar(y=features, x=values, marker_color=colors, orientation='h')])
        fig.update_layout(
            title='SHAP Values (Blue: Legitimate, Red: Fraudulent)',
            yaxis_title='Features',
            xaxis_title='SHAP Value',
#            width=1000, height=600,
            margin=dict(l=200, r=200, t=50, b=50)
        )

        return fig  
        
    def shap_values(self):
        live_data_instance_scaled = self._prepare_instance()
        shap_values = self.explainer.shap_values(live_data_instance_scaled)

        shap_values_1d = shap_values[0]  # Assuming a single output model
        sorted_features = sorted(zip(self.features, shap_values_1d), key=lambda x: x[1])
        features, values = zip(*sorted_features)
        return features, values
        
        
    def shap_feature_influence(self):
        live_data_instance_scaled = self._prepare_instance()
        shap_values = self.explainer.shap_values(live_data_instance_scaled)

        # Assuming a single output model
        shap_values_1d = shap_values[0]
        sorted_features = sorted(zip(self.features, shap_values_1d), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_features)
        df = pd.DataFrame({'Features': features, 'Importances': values})

        total_sum = df['Importances'].abs().sum()
        output_lines = []  # Initialize a list to hold output lines

        if total_sum > 0:
            df['Influence_Percent'] = (df['Importances'] / total_sum) * 100
            for index, row in df.iterrows():
                influence_description = "towards Fraudulent Class" if row['Importances'] >= 0 else "away from Fraudulent to Legitimate Class"
                output_lines.append(f"{row['Features']} pushed this prediction outcome {influence_description} by {round(row['Influence_Percent'], 1)}%")
        else:
            output_lines.append("No significant features found.")

        # Join all lines into a single string with new lines
        summary_text = "\n".join(output_lines)
        return summary_text
          

    def shap_feature_influence_percentage(self):
        live_data_instance_scaled = self._prepare_instance()
        shap_values = self.explainer.shap_values(live_data_instance_scaled)

        # Assuming a single output model
        shap_values_1d = shap_values[0]
        sorted_features = sorted(zip(self.features, shap_values_1d), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_features)
        df = pd.DataFrame({'Features': features, 'Importances': values})

        # Count the total number of feature values
        positive_count = (df['Importances'] > 0).sum()
        negative_count = (df['Importances'] < 0).sum()
        zero_count = (df['Importances'] == 0).sum()

        # Calculate each of their percentages
        total_features = len(df)
        positive_percentage = (positive_count / total_features) * 100
        negative_percentage = (negative_count / total_features) * 100
        zero_percentage = (zero_count / total_features) * 100

        # Accumulate output in a list
        output_lines = [
            f"Percentage of features that increase likelihood of being classified as 'Fraudulent': {positive_percentage:.2f}%",
            f"Percentage of features that increase likelihood of being classified as 'Legitimate': {negative_percentage:.2f}%",
            f"Percentage of features with no effect on the classification: {zero_percentage:.2f}%"
        ]

        # Join all lines into a single string with new lines
        summary_text = "\n".join(output_lines)
        return summary_text

                
def networkx_exp():
    explanation_text = (
        "Each node in this network graph represents a feature, with the color indicating its interaction strength.\n"
        "Warmer colors indicate a higher contribution to the prediction outcome, while cooler colors "
        "suggest a lower contribution.\n\n"
        "Linkages between nodes illustrate interactions, with thicker lines implying stronger "
        "interactions."
    )
    return explanation_text
           
    

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



# In[ ]:




