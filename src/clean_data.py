import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('training_data')
parser.add_argument('testing_data')
parser.add_argument('cleaned_train')
parser.add_argument("cleaned_test")
args = parser.parse_args()

# Read in raw data
titanic_train = pd.read_csv(args.training_data, index_col = 0)
titanic_test = pd.read_csv(args.testing_data, index_col = 0)

# Drop columns that we are not interested in
titanic_train = titanic_train.loc[:,["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
titanic_test = titanic_test.loc[:,["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

# Replace NaN values
def calcNAN(df):
    return {'Age': df.Age.mean(), 'Fare': df.Fare.median()}

def fillNAN(df):
    values = calcNAN(titanic_train)
    for i in df:
        i.fillna(value=values, inplace = True)

fillNAN([titanic_train, titanic_test])

# Replace Sex to 1 or 0
def process_sex(df):
    for i in df:
        i["Sex"] = i["Sex"].map({"male": 1, "female": 0})


fillNAN([titanic_train, titanic_test])
process_sex([titanic_train, titanic_test])

# Export data
titanic_train.to_csv(args.cleaned_train)
titanic_test.to_csv(args.cleaned_test)
