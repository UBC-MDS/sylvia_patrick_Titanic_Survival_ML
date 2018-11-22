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

# Read data
titanic_train = pd.read_csv("cleaned_train.csv", index_col = 0)
titanic_test = pd.read_csv("cleaned_test.csv", index_col = 0)

Xtrain = titanic_train.iloc[:, 0:-1]
ytrain = titanic_train.Survived

Xtest = titanic_test.iloc[:, 0:-1]
ytest = titanic_test.Survived

# Cross Validate
max_depths = range(1, 50)

accuracies = []
for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth)
    cross_vals = cross_val_score(tree, Xtrain, ytrain, cv=10)
    accuracies.append(cross_vals.mean())
    
best_depth = max_depths[np.argmax(accuracies)]
# best_depth = 7

# Create decision tree and fit model
tree = DecisionTreeClassifier(max_depth=7, random_state=1234)
tree.fit(Xtrain,ytrain)
# train_accuracy = tree.score(Xtrain,ytrain)
# print(train_accuracy)

# Predict target with Xtest
predictions = tree.predict(Xtest)
tree_predict = titanic_test.copy()
tree_predict["prediction"] = predictions

# Export predictions to csv?
# tree_predict.to_csv("predictions.csv")

# Feature ranking
importances = tree.feature_importances_
importance_indices = importances.argsort()[::-1]

for i in range(Xtrain.shape[1]):
    print("Rank {}. {} ({})".format(i+1, tree_predict.columns[importance_indices[i]], importances[importance_indices[i]]))

