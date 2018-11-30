#!/usr/bin/env python

# data_analysis.py
# Patrick Tung, Sylvia Lee (Nov 22, 2018)

# Description: This script takes in the cleaned titanic datasets and fits a
#              decision tree to predict which passengers survived the Titanic.
#              This includes cross validating decision trees to determine
#              the value for hyperparameters. It then returns the top three
#              most important features.

# Usage: python 03_data_analysis.py <cleaned_train.csv path> <cleaned_test.csv path> <output_folder path/>
# Example: python 03_data_analysis.py data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv results/

# Load dependencies
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
import pickle

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("training_data")
parser.add_argument("testing_data")
parser.add_argument("output_folder")
args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('training_data')
parser.add_argument('testing_data')
parser.add_argument('output_folder')
args = parser.parse_args()

def main():
    # Read data
    titanic_train = pd.read_csv(args.training_data, index_col = 0)
    titanic_test = pd.read_csv(args.testing_data, index_col = 0)
    print("Data Import Success")

    # Split data into feature and target dataframes
    Xtrain, ytrain = split_data(titanic_train)
    Xtest, ytest = split_data(titanic_test)

    # Cross Validation to find the best max_depth for decision classification tree
    best_depth = calc_depth(Xtrain, ytrain)

    # Create decision tree and fit model
    tree = fit(Xtrain, ytrain, best_depth)

    # Predict using train and test set
    predicted_train = predict(tree, Xtrain, titanic_train)
    predicted_test = predict(tree, Xtest, titanic_test)

    # Export predictions to csv
    pickle.dump(tree, open(args.output_folder + "model/decision_tree_model.sav", "wb"))
    predicted_train.to_csv(args.output_folder+"train_prediction.csv")
    predicted_test.to_csv(args.output_folder+"test_prediction.csv")
    print("Exports complete")

def split_data(data):
    """
    Description: split the data sets into a X-feature set and y-target sets
    Parameter:   data(dataframe) = dataframe with target being in the last column named "Survived"
    Return:      X(dataframe) = dataframe containing feature columns
                 y(dataframe) = dataframe containing the target column
    """
    X = data.iloc[:, 0:-1]
    y = data.Survived
    return(X, y)


def calc_depth(Xtrain,ytrain):
    """
    Description: Find the best max_depth hyperparameter by 10-fold cross valiation
    Parameter:   Xtrain(dataframe) = dataframe containing the training feature columns
                 ytrain(dataframe) = dataframe containing the training target column
    Return:      best_depth(integer) = the max_depth that gave the best accuracies
    """
    max_depths = range(1, 50)

    accuracies = []
    for depth in max_depths:
        tree = DecisionTreeClassifier(max_depth=depth)
        cross_vals = cross_val_score(tree, Xtrain, ytrain, cv=10)
        accuracies.append(cross_vals.mean())

    plt.plot(max_depths,accuracies)
    plt.legend()
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy Score")
    plt.savefig(args.output_folder + "/figure/CV_accuracy_score_lineplot.png")
    print("CV Accuracy Exported")

    best_depth = max_depths[np.argmax(accuracies)]
    return(best_depth)


def fit(Xtrain, ytrain, best_depth):
    """
    Description: create decision classification tree
    Parameter:   Xtrain(dataframe) = dataframe containing the training feature columns
                 ytrain(dataframe) = dataframe containing the training target column
                 best_depth(integer) = the max_depth that gave the best accuracies
    Return:      tree(DecisionTreeClassifier object) = classification tree model
    """
    tree = DecisionTreeClassifier(max_depth=best_depth)
    tree.fit(Xtrain,ytrain)
    return(tree)

def predict(tree, feature_set, whole_set):
    """
    Description: predict targets from feature set using the classification tree
    Parameter:   tree(DecisionTreeClassifier object) = classification tree model
                 feature_set(dataframe) = dataframe containing the feature columns
                 whole_set(dataframe) = dataframe containing the feature and target columns
    Return:      tree_predict(dataframe) = dataframe with an addition prediction column
                 appended to the whole_set dataframe
    """
    predictions = tree.predict(feature_set)
    tree_predict = whole_set.copy()
    tree_predict["Prediction"] = predictions
    return(tree_predict)


if __name__ == "__main__":
    main()
