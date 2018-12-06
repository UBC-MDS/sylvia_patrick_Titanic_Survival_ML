# Predicting survival on the Titanic

**Project Contributors:** Sylvia Lee([sylvia19](https://github.ubc.ca/MDS-2018-19/DSCI_522_proposal_sylvia19/blob/master/README.md)) and Patrick Tung([ptung](https://github.ubc.ca/mds-2018-19/DSCI_522_proposal_ptung))

**Start Date:** 15 Nov, 2018

## Project Description

### Introduction

*Who will survive through the Titanic disaster?*

![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg)

> RMS Titanic departing Southampton on April 10, 1912. From [Wikipedia Commons](https://en.wikipedia.org/wiki/File:RMS_Titanic_3.jpg)

For most people, "Titanic" is both a classic movie and a beautiful love story. However, the infamous Titanic catastrophe had also been said to be a prime example of social stratification and status discriminations in the 1900s. In addition to the "women and children first" evacuation policy, it had been rumored that the lives of the people with social prestige and high class standing were prioritized in the moment of danger. In this analysis, we used supervised machine learning (ML) to answer the question **"What are the 3 strongest predictors of people who survived on the Titanic?"**

We retrieved the data from [Kaggle's Titanic:Machine Learning from Disaster](https://www.kaggle.com/c/titanic) and developed a decision-classification-tree machine learning model focusing on following features:

| Feature | Type | Description |
| --- | --- | --- |
| Pclass | Categorical | Passenger Class |
| Sex | Categorical | Sex of Passenger |
| Age | Continuous | Age of Passenger |
| SibSp | Discrete | Number of siblings/spouses onboard |
| Parch | Discrete | Number of parents/children onboard |
| Fare | Continuous | Fare price |

In our project, we explored the dataset by generating graphs of the features' distribution in the population of passengers. Subsequently we developed the decision tree model using Python's scikit-learn package and applied the model to a test dataset to predict the survival of the passenger given the same list of features. Lastly, we summarized our analysis by calculating the accuracy of our ML model and ranking the list of features' predictive power.


### Usage

There are two recommended methods of running this analysis:

#### 1. Docker

1. Install [Docker](https://www.docker.com/get-started)
2. Download and clone this repository
3. Run the following code in terminal to download the Docker image:
```
docker pull patricktung/sylvia_patrick_titanic_survival_ml
```

4. Use the command line to navigate to the root of this repo
5. Type the following code into terminal to run the analysis:

```
docker run --rm -e PASSWORD=test -v <ABSOLUTE PATH OF REPO>:/home/titanic_predictive_analysis patricktung/sylvia_patrick_titanic_survival_ml make -C '/home/titanic_predictive_analysis' all
```

6. If you would like a fresh start, type the following:

```
docker run --rm -e PASSWORD=test -v <ABSOLUTE PATH OF REPO>:/home/titanic_predictive_analysis patricktung/sylvia_patrick_titanic_survival_ml make -C '/home/titanic_predictive_analysis' clean
```

#### 2. Make (without Docker)

1. Clone this repository

2. Run the following commands:

```
# Removes all unnecessary files to start the analysis from scratch
make clean

# Runs all necessary scripts in order to generate the report
make all
```

**The `Makefile` would run the following scripts:**

*Step 1*: This script takes in the raw Titanic data and cleans it into a data set that fits our research question.

*Inputs*: Raw training data, Raw test data

*Outputs*: Cleaned training data, Cleaned test data

```
python src/01_data_clean.py data/raw/train.csv data/raw/test.csv data/raw/gender_submission.csv data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv
```


*Step 2*: This script takes the cleaned training data and creates some visualizations that are ready for exploratory data analysis.

*Inputs*: Cleaned training data

*Outputs*: 6 figures for EDA
```
python src/02_data_exploratory_vis.py data/cleaned/cleaned_train.csv results/figure/
```


*Step 3*: This script takes in the cleaned training data and testing data and fits a decision tree to predict which passengers survived the Titanic.

*Inputs*: Cleaned training data, Cleaned testing data

*Outputs*: Decision tree model, Predictions for training set, Predictions for testing set, Cross validation accuracy plot
```
python src/03_data_analysis.py data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv results/
```


*Step 4*: This script takes in the decision tree model and the predictions to create summary data of the accuracy, feature ranks, and the graphic representation of our decision tree.

*Inputs*: Decision tree model, Predictions for training set, Predictions for testing

*Outputs*: Accuracies, Ranks of the Features, Decision Tree graphic representation
```
python src/04_data_summarization.py results/model/decision_tree_model.sav results/train_prediction.csv results/test_prediction.csv results/
```


*Step 5*: This line renders the RMarkdown file with the appropriate files created from the steps before.
```
Rscript -e 'rmarkdown::render("docs/Titanic_Predictive_Data_Analysis.Rmd")'
```

### Dependency Diagram of the Makefile

<img src="https://github.com/tungpatrick/sylvia_patrick_Titanic_Survival_ML/blob/master/Makefile.png" alt="dependency diagram" height="280">

### Dependencies

+ Python libraries:
    + argparse v1.1
    + pandas v0.23.4
    + numpy v1.15.3
    + sklearn v0.20.0
    + matplotlib v3.0.1
    + seaborn v0.9.0
    + pickle v4.0
    + graphviz v0.8.4


+ R packages:
    + here v0.1
