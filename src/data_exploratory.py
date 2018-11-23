#!/usr/bin/env python
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_folder')
args = parse_args()

def main():
    # import file
    titanic_train = pd.read_csv(train_data, index_col = 0)

    survived = titanic_train.query("Survived == 1")
    died = titanic_train.query("Survived != 1")

    # Create plot for Pclass
    sns.countplot(data = titanic_train, x = "Pclass", hue = "Survived")
    plt.legend(title = "Survival", labels = ["Did not survive", "Survived"])
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    plt.savefig(str(output_folder) + "/pclass.png")

    # Create plot for Sex groups
    sns.countplot(data = titanic_train, x = "Sex", hue = "Survived")
    plt.legend(title = "Survival", labels = ["Did not survive", "Survived"])
    plt.xticks(np.array([0,1]), ("Female", "Male"))
    plt.xlabel("Sex")
    plt.ylabel("Count")
    plt.savefig(str(output_folder)"/sex.png")


    # plot Histograms
    cont_variables = {"Age" : "Age", "SibSp" : "Number of siblings/spouses onborad", "Parch": "Number of parents/children onbard", "Fare": "Fare prices"}
    cont_plot(survived, died, cont_variables)


def cont_plot(survived_df, died_df, cont_variables):
    for i in cont_variables:
        fig = plt.subplot()

        survived_df.plot.hist(y = i, ax = fig, label = "Survived", alpha = 0.4)
        died_df.plot.hist(y = i, ax = fig, label = "Did not survive", alpha = 0.4)

        plt.xlabel(cont_variables[i])
        plt.savefig("../results/figures/" + str(i) + "_plot.png")
        plt.show()
