#!/usr/bin/env python

# data_exploratory.py
# Patrick Tung, Sylvia Lee (Nov 22, 2018)

# Description: This script takes in the clean titanic training sets
#              and generate exploratory figures. Barplots are generated for
#              categorical variables, and histograms are generated for numeric
#              variables.

# Usage:   python 02_data_exploratory_vis.py <train.csv path> <output_folder path>
# Example: python 02_data_exploratory_vis.py data/cleaned/cleaned_train.csv results/figure/

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
    # Import file
    titanic_train = pd.read_csv(args.input_file, index_col = 0)

    survived = titanic_train.query("Survived == 1")
    died = titanic_train.query("Survived != 1")

    # Plot Histograms for continuous variables
    cont_variables = {"Age" : "Age",
    "SibSp" : "Number of siblings/spouses onboard",
    "Parch": "Number of parents/children onboard",
    "Fare": "Fare prices"}
    cont_plot(survived, died, cont_variables)

    # Create plot for Pclass
    sns.countplot(data = titanic_train, x = "Pclass", hue = "Survived")
    plt.legend(title = "Survival", labels = ["Did not survive", "Survived"])
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    plt.savefig(args.output_folder + "Pclass_plot.png")

    # Create plot for Sex groups
    sns.countplot(data = titanic_train, x = "Sex", hue = "Survived")
    plt.legend(title = "Survival", labels = ["Did not survive", "Survived"])
    plt.xticks(np.array([0,1]), ("Female", "Male"))
    plt.xlabel("Sex")
    plt.ylabel("Count")
    plt.savefig(args.output_folder + "Sex_plot.png")

    print("Plots saved")

def cont_plot(survived_df, died_df, cont_variables):
    for i in cont_variables:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.hist(survived_df[i])
        ax1.set_title("Survived")
        ax2.hist(died_df[i])
        ax2.set_title("Did not survive")

        # Axis labels
        ax1.set_ylabel("Frequency")
        fig.text(0.5, 0.04, cont_variables[i], ha='center', va='center')
        plt.savefig(args.output_folder + str(i) + "_plot.png")
        plt.close()

if __name__ == "__main__":
    main()
