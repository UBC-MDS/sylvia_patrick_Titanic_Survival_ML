#!/usr/bin/env python
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_folder')
args = parser.parse_args()

def main():
    # import file
    titanic_train = pd.read_csv(args.input_file, index_col = 0)

    survived = titanic_train.query("Survived == 1")
    died = titanic_train.query("Survived != 1")

    # plot Histograms for continuous variables
    cont_variables = {"Age" : "Age",
    "SibSp" : "Number of siblings/spouses onborad",
    "Parch": "Number of parents/children onbard",
    "Fare": "Fare prices"}

    cont_plot(survived, died, cont_variables)

    # Create plot for Pclass
    sns.countplot(data = titanic_train, x = "Pclass", hue = "Survived")
    plt.legend(title = "Survival", labels = ["Did not survive", "Survived"])
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    plt.savefig(args.output_folder + "pclass.png")

    # Create plot for Sex groups
    sns.countplot(data = titanic_train, x = "Sex", hue = "Survived")
    plt.legend(title = "Survival", labels = ["Did not survive", "Survived"])
    plt.xticks(np.array([0,1]), ("Female", "Male"))
    plt.xlabel("Sex")
    plt.ylabel("Count")
    plt.savefig(args.output_folder + "sex.png")

def cont_plot(survived_df, died_df, cont_variables):
    for i in cont_variables:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.hist(survived_df[i])
        ax2.hist(died_df[i])

        # Axis labels
        ax1.set_ylabel("Frequency")
        fig.text(0.5, 0.04, cont_variables[i], ha='center', va='center')
        plt.savefig(args.output_folder + str(i) + "_plot.png")
        plt.close()

if __name__ == "__main__":
    main()
