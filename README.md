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

- Passenger class (Categorical)
- Sex (Categorical)
- Age (Quantitative Continuous)
- Number of siblings/spouses onboard (Quantitative Discrete)
- Number of parents/children onboard (Quantitative Discrete)
- Fare price (Quantitative Continuous)

In our project, we explored the dataset by generating graphs of the features' distribution in the population of passengers. Subsequently we developed the decision tree model using Python's scikit-learn package and applied the model to a test dataset to predict the survival of the passenger given the same list of features. Lastly, we summarized our analysis by calculating the accuracy of our ML model and ranking the list of features' predictive power.


### Usage

Multiple Python scripts were written in the analysis procedure. The following outlined the steps taken to run this project.

1. Clone this repository.

2. Run the following code in the terminal at the project's root repository.

```
python src/clean_data.py data/raw/train.csv data/raw/test.csv data/raw/gender_submission.csv data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv    
python src/data_exploratory.py data/cleaned/cleaned_train.csv results/figure/
python src/data_analysis.py data/cleaned/cleaned_train.csv data/cleaned/cleaned_test.csv results/
python src/summarize_data.py results/model/decision_tree_model.sav results/train_prediction.csv results/test_prediction.csv results/
Rscript -e 'rmarkdown::render("docs/Titanic_Predictive_Data_Analysis.Rmd")'
```

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
    + imager v0.41.1
