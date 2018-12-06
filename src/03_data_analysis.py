#!/usr/bin/env python

# 03_data_analysis.py
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
from sklearn.model_selection import cross_val_score, KFold
import pickle
import os.path

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
    best_depth, accuracies = calc_depth(Xtrain, ytrain)
    create_cv_plot(accuracies)

    # Create decision tree and fit model
    tree = DecisionTreeClassifier(max_depth=best_depth, random_state=1234)
    tree.fit(Xtrain,ytrain)

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

    kfold = KFold(n_splits=10, random_state=1234)

    accuracies = []
    for depth in max_depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=1234)
        cross_vals = cross_val_score(tree, Xtrain, ytrain, cv=kfold)
        accuracies.append(cross_vals.mean())

    best_depth = max_depths[np.argmax(accuracies)]

    return(best_depth, accuracies)

def create_cv_plot(accuracies):
    """
    Description: Plot accuracies vs different values of max_depth hyperparameter
    Parameter:   accuracies = list containing accuracies from cross validation
    Return:      None. Plot output to results/figure/ directory
    """
    max_depths = range(1, len(accuracies) + 1)
    plt.plot(max_depths,accuracies)
    plt.legend()
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy Score")
    plt.savefig(args.output_folder + "/figure/CV_accuracy_score_lineplot.png")
    print("CV Accuracy Exported")

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


# Unit testing
# ============

# Create toy data set for unit testing
unit_train_df = pd.DataFrame({'Age': [1, 2, 3, 3, 5, 4, 5, 2, 5, 2], 'Fare': [7, 2, 3, 2, 9, 4, 5, 2, 5, 2], "Survived": [0, 1, 1, 0, 1, 1, 1, 1, 0, 1]})

# Unit test for split_data()
unit_Xtrain, unit_ytrain = split_data(unit_train_df)
assert unit_Xtrain.equals(unit_train_df.loc[:,"Age":"Fare"]), 'The data was split incorrectly.'
assert unit_ytrain.equals(unit_train_df.Survived), 'The data was split incorrectly.'

# Unit test for calc_depth()
assert calc_depth(unit_Xtrain, unit_ytrain) == (1, [0.7, 0.7, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]) , 'The best depth is calculated incorrectly.'

#Unit test for create_cv_plot()
assert os.path.isfile("results/figure/CV_accuracy_score_lineplot.png"), 'CV_accuracy_score_lineplot does not exist.'

# Unit test for predict()
tree = DecisionTreeClassifier(max_depth=1, random_state=1234)
tree.fit(unit_Xtrain, unit_ytrain)
tree.predict(unit_Xtrain)
unit_pred = predict(tree, unit_Xtrain, unit_train_df)

assert isinstance(unit_pred, pd.DataFrame), 'Return is not a dataframe'
assert len(unit_pred.index) == len(unit_train_df.index), 'Number of instance in data frame is incorrect'
assert len(unit_pred.columns) == len(unit_train_df.columns) + 1, 'Number of columns incorrect, check if Prediction column exists'
assert list(unit_pred.Prediction) == list(tree.predict(unit_Xtrain)), 'Predictions not matched with scikit-learn prediction method output'
