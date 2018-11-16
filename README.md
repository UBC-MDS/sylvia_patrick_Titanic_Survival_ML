# DSCI_522_Project: Predicting survival on the Titanic

**Project Contributors:** Sylvia Lee([sylvia19](https://github.ubc.ca/MDS-2018-19/DSCI_522_proposal_sylvia19/blob/master/README.md)) and Patrick Tung([ptung](https://github.ubc.ca/mds-2018-19/DSCI_522_proposal_ptung))

**Start Date:** 15 Nov, 2018

## Project Description

*Who will survive through the Titanic crash?*

We will do an analysis on the data set from [Kaggle's Titanic:Machine Learning from Disaster](https://www.kaggle.com/c/titanic). It includes information about the passengers on the Titanic and if they survived the disaster or not. All data was downloaded from the site as csv files and uploaded into the [data/raw/](https://github.com/UBC-MDS/sylvia_patrick_Titanic_Survival_ML/tree/master/data/raw) directory. 

To prove that we can load the dataset, we have created a `data_import.py` Python script to read and import the `training.csv` dataset. The script can be found in the `src` directory.

**Research Question:** What are the strongest predictors of people who survived on the Titanic?

This is a predictive research question, so we will be implementing a classification decision tree model. 

To effectively predict the survival of passengers onboard, we will analyze features including:
- Passenger classes
- Sex
- Age
- Number of siblings/spouses onborad
- Number of parents/children onbard
- Fare prices

We have just started learning about how to do this in our DSCI 571 class and would like to see how much we can learn throughout the course. 

**Project Goal:**

- Create a classification tree that can efficiently predict if a passenger will survive the disaster based on the features listed above.
- Inquire which features are the most predictive of the passengers' survivals. 

**Project Overview**

We forsee that at the end of the project, we will have the following summarizing results/figures:

- Histograms for each feature to visualizing differential distribution depending on survival state of the passengers(Survived or not). 
- Accuracy score that evaluates the efficiency of our classification tree 
- Graphvis figure to visualize the decision tree 
- Table of importance/point graph that depicts the predictiveness of each feature in the survival of the passenger. 
