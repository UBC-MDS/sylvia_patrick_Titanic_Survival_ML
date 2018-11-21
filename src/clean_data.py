import pandas as pd
import numpy as np

# Read in raw data
titanic_train = pd.read_csv("../data/raw/train.csv", index_col = 0)
titanic_test = pd.read_csv("../data/raw/test.csv", index_col = 0)

# Drop columns that we are not interested in
titanic_train = titanic_train.loc[:,["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
titanic_test = titanic_test.loc[:,["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

# Replace NaN values
def fillNAN(df):
    for i in df:
        values = {'Age': i.Age.mean(), 'Fare': i.Fare.median()}
        i.fillna(value=values, inplace = True)

# Replace Sex to 1 or 0
def process_sex(df):
    for i in df:
        i["Sex"] = i["Sex"].map({"male": 1, "female": 0})


fillNAN([titanic_train, titanic_test])
process_sex([titanic_train, titanic_test])

# Export data
titanic_train.to_csv("../data/cleaned/cleaned_train.csv")
titanic_test.to_csv("../data/cleaned/cleaned_test.csv")
