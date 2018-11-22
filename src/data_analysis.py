import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Read data
titanic_train = pd.read_csv("cleaned_train.csv", index_col = 0)
titanic_test = pd.read_csv("cleaned_test.csv", index_col = 0)
real_output = pd.read_csv("gender_submission.csv")

# Print correlation (do we need it?)
correlation = pd.DataFrame(titanic_train.corr()["Survived"])
# print(correlation)

# Create Xtrain, ytrain, Xtest
Xtrain = titanic_train.iloc[:, 0:-1]
ytrain = titanic_train.Survived

Xtest = titanic_test

# Create decision tree and fit model
tree = DecisionTreeClassifier(random_state=1234)
tree.fit(Xtrain,ytrain)

# Predict target with Xtest
predictions = tree.predict(Xtest)
tree_predict = titanic_test.copy()
tree_predict["prediction"] = predictions
tree_predict["true"] = np.array(real_output.Survived)

# print(tree_predict)
# Export predictions to csv
tree_predict.to_csv("predictions.csv")

# Feature ranking
importances = tree.feature_importances_
importance_indices = importances.argsort()[::-1]

for i in range(Xtrain.shape[1]):
    print("Rank {}. Feature {} ({})".format(i+1, importance_indices[i], importances[importance_indices[i]]))
