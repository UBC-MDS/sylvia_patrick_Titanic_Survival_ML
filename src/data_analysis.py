#!/usr/bin/env python

# data_analysis.py
# Patrick Tung, Sylvia Lee (Nov 22, 2018)

# Description: This script takes in the cleaned titanic datasets and fits a
#              decision tree to predict which passengers survived the Titanic.
#              This includes cross validating decision trees to determine
#              the value for hyperparameters. It then returns the top three
#              most important features.

# Usage: python data_analysis.py <train.csv path> <test.csv path> <gender_submission.csv path>
#        <clean_train.csv path> <clean_test.csv path> <clean_total.csv path>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def main():
    # Read data
    titanic_train = pd.read_csv("../data/cleaned/cleaned_train.csv", index_col = 0)
    titanic_test = pd.read_csv("../data/cleaned/cleaned_test.csv", index_col = 0)

    # Split data into feature and target dataframes
    Xtrain, ytrain = split_data(titanic_train)
    Xtest, ytest = split_data(titanic_test)

    # Cross Validation to find the best max_depth for decision classification tree
    best_depth = calc_depth(Xtrain, ytrain)
    print(best_depth)

    # Create decision tree and fit model
    tree = fit(Xtrain, ytrain, best_depth)

    # Predict using train and test set
    predicted_train = predict(tree, Xtrain, titanic_train)
    predicted_test = predict(tree, Xtest, titanic_test)

    # Get accuracy scores
    accuracies_df = pd.DataFrame(columns = ["set", "n_total", "n_correct_pred", "n_incorrect_pred", "accuracy"])
    accuracies_df.loc[0] = get_accuracies(predicted_train, "train")
    accuracies_df.loc[1] = get_accuracies(predicted_test, "test")

    # Rank the most predictive features
    features = list(Xtrain)
    feature_rank_df = feature_rank(tree, features)

def split_data(data):
    X = data.iloc[:, 0:-1]
    y = data.Survived
    return(X, y)

def calc_depth(Xtrain,ytrain):
    max_depths = range(1, 50)

    accuracies = []
    for depth in max_depths:
        tree = DecisionTreeClassifier(max_depth=depth)
        cross_vals = cross_val_score(tree, Xtrain, ytrain, cv=10)
        accuracies.append(cross_vals.mean())

    best_depth = max_depths[np.argmax(accuracies)]
    return(best_depth)

def fit(Xtrain, ytrain, best_depth):
    tree = DecisionTreeClassifier(max_depth=best_depth)
    tree.fit(Xtrain,ytrain)
    return(tree)

def predict(tree, feature_set, whole_set):
    predictions = tree.predict(feature_set)
    tree_predict = whole_set.copy()
    tree_predict["Prediction"] = predictions
    return(tree_predict)

def get_accuracies(df, set_name):
    correct_predictions = df.Survived[df.Survived == df.Prediction].sum()
    incorrect_predictions = df.Survived[df.Survived != df.Prediction].sum()
    total = correct_predictions + incorrect_predictions
    accuracy = round(correct_predictions / total, 4)
    return([set_name, total, correct_predictions, incorrect_predictions, accuracy])

def feature_rank(tree, features):
    importances = calc_importances(tree)
    importance_indices = importances.argsort()[::-1]

    feature_rank_df = pd.DataFrame(columns = ['Rank', 'Feature', 'Importance'])

    for i in range(len(features)):
        feature_rank_df.loc[i] = [i+1, features[importance_indices[i]], importances[importance_indices[i]]]

    return(feature_rank_df)

def calcImportances(model):
    return model.feature_importances_

if __name__ == "__main__":
    main()
