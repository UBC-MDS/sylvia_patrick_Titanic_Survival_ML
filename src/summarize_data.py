#!/usr/bin/env python

# summarize_data.py
# Patrick Tung, Sylvia Lee (Nov 22, 2018)

# Description: Generate analysis summaries from the predictions made
#              with the classification tree model. Output includes prediction accurcies,
#              feature importance ranking and a graphical depiction of the tree model.

# Usage:   python summarize_data.py <tree.sav path> <train.csv path> <test.csv path> <output_folder path>
# Example: python summarize_data.py results/model/decision_tree_model.sav results/train_prediction.csv results/test_prediction.csv results/

# Load depencies
import argparse
import graphviz
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
import pickle

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("tree_model")
parser.add_argument("predicted_train_data")
parser.add_argument("predicted_test_data")
parser.add_argument("output_folder")
args = parser.parse_args()

def main():
    # Import data
    tree = pickle.load(open(args.tree_model, "rb"))
    predicted_train = pd.read_csv(args.predicted_train_data, index_col = 0)
    predicted_test = pd.read_csv(args.predicted_test_data, index_col = 0)
    print("Data Import Success")

    # Get accuracy scores
    accuracies_df = pd.DataFrame(columns = ["set", "n_total", "n_correct_pred", "n_incorrect_pred", "accuracy"])
    accuracies_df.loc[0] = get_accuracies(predicted_train, "train")
    accuracies_df.loc[1] = get_accuracies(predicted_test, "test")

    # Export accuracy scores to csv
    accuracies_df.to_csv(args.output_folder + "accuracies.csv")
    print("Export Accuracies")

    # Rank the most predictive features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    feature_rank_df = feature_rank(tree, features)

    # Export feature ranks to csv
    feature_rank_df.to_csv(args.output_folder + "feature_ranks.csv")
    print("Export Feature Ranks")

    # Export Decision Tree
    save_tree(tree, features)
    print("Export Decision Tree")


# Description: evaluate accuracies of the predictions be comparing the targets and the predictions
# Parameter:   df(dataframe) = dataframe with an addition prediction column
#                                        appended to the whole_set dataframe
#              set_name(string) = name of the set being evaluated ("train" or "test")
# Return:      (list) = list containing the set name(str),
#                       total number of predicted samples(int), number of correct predictions(int),
#                       number of incorrect predictions(int), prediction accuracy(float)
def get_accuracies(df, set_name):
    correct_predictions = df.Survived[df.Survived == df.Prediction].sum()
    incorrect_predictions = df.Survived[df.Survived != df.Prediction].sum()
    total = correct_predictions + incorrect_predictions
    accuracy = round(correct_predictions / total, 4)
    return([set_name, total, correct_predictions, incorrect_predictions, accuracy])


# Description: rank the features from the most predictive to the least predictive
# Parameter:   tree(DecisionTreeClassifier object) = classification tree model
#              features(list) = list of feature names(str)
# Return:      feature_rank_df(dataframe) = dataframe that contains the rank, feature
#              name and importance measure in ascending order. Rank of 1 is most predictive
def feature_rank(tree, features):
    importances = tree.feature_importances_
    importance_indices = importances.argsort()[::-1]

    feature_rank_df = pd.DataFrame(columns = ['Rank', 'Feature', 'Importance'])

    for i in range(len(features)):
        feature_rank_df.loc[i] = [i+1, features[importance_indices[i]], importances[importance_indices[i]]]

    return(feature_rank_df)

# Description: create the decision tree pictorical depiction using Graphviz package
# Parameter:   model(DecisionTreeClassifier object) = classification tree model_selection
#              feature_names(list) = list of feature names(str)
#              class_names(list) = list of class names, optional
#              save_file_prefix(str) = name for the exported file
# Return:      no return variable, but a .png file will be exported in a "model" directory in the
#              output_folder path.
def save_tree(model, feature_names, class_names = ["Not Survived", "Survived"] , save_file_prefix = 'decision_tree'):
    dot_data = export_graphviz(model, out_file = None,
                             feature_names = feature_names,
                             class_names = class_names,
                             filled = True, rounded = True,
                             special_characters = True)

    graph = graphviz.Source(dot_data, format = "png")
    graph.render(args.output_folder + save_file_prefix)

if __name__ == "__main__":
    main()
