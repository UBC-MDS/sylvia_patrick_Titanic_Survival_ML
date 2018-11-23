#!/usr/bin/env python

# summarize_data.py
# Patrick Tung, Sylvia Lee (Nov 22, 2018)

# Description: 

# Usage: python summarize_data.py <train.csv path> <test.csv path> <output_folder path>

# import data_analysis
import argparse
import graphviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

def main():
    # Issue: We need to find a way to have "tree" here. It's not possible to get the feature ranks without the tree. 
    # Do you know a way?
    tree = get_tree()
    
    # Read data
    predicted_train = pd.read_csv("results/summaries/train_predictions.csv", index_col = 0)
    predicted_test = pd.read_csv("results/summaries/test_predictions.csv", index_col = 0)
    print("Data Import Success")

    # Get accuracy scores
    accuracies_df = pd.DataFrame(columns = ["set", "n_total", "n_correct_pred", "n_incorrect_pred", "accuracy"])
    accuracies_df.loc[0] = get_accuracies(predicted_train, "train")
    accuracies_df.loc[1] = get_accuracies(predicted_test, "test")

    # Export accuracy scores to csv
    accuracies_df.to_csv("results/summaries/accuracies.csv")
    print("Export Accuracies")

    # Rank the most predictive features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    feature_rank_df = feature_rank(tree, features)

    # Export feature ranks to csv
    feature_rank_df.to_csv("results/summaries/feature_ranks.csv")
    print("Export Feature Ranks")
    
    # Export Decision Tree
    save_tree(tree, save_file_prefix='decision_tree')
    print("Export Decision Tree")
    
def get_accuracies(df, set_name):
    correct_predictions = df.Survived[df.Survived == df.Prediction].sum()
    incorrect_predictions = df.Survived[df.Survived != df.Prediction].sum()
    total = correct_predictions + incorrect_predictions
    accuracy = round(correct_predictions / total, 4)
    return([set_name, total, correct_predictions, incorrect_predictions, accuracy])

# The following two doesnt work because we dont have the tree model
def feature_rank(tree, features):
    importances = tree.feature_importances_
    importance_indices = importances.argsort()[::-1]

    feature_rank_df = pd.DataFrame(columns = ['Rank', 'Feature', 'Importance'])

    for i in range(len(features)):
        feature_rank_df.loc[i] = [i+1, features[importance_indices[i]], importances[importance_indices[i]]]

    return(feature_rank_df)

def save_tree(model, class_names = ["Not Survived", "Survived"], save_file_prefix = 'decision_tree', **kwargs):
    dot_data = export_graphviz(model, out_file=None, 
                             feature_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'],  
                             class_names=class_names,  
                             filled=True, rounded=True,  
                             special_characters=True, **kwargs)  

    graph = graphviz.Source(dot_data) 
    graph.render("results/images"+save_file_prefix)

if __name__ == "__main__":
    main()
