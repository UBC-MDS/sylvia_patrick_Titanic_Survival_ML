# DSCI_522_Project README
## Sylvia Lee(sylvia19) and Patrick Tung(ptung)

1. The data set we chose to do was from [Kaggle](https://www.kaggle.com/c/titanic) and includes information about the passengers on the Titanic. To prove that we can load the dataset, we have created a [`data_import.py`](https://github.com/UBC-MDS/sylvia_patrick_Titanic_Survival_ML/tree/master/src) Python script to read and import the `training.csv` dataset. The datasets can be found in the [data/raw/directory](https://github.com/UBC-MDS/sylvia_patrick_Titanic_Survival_ML/tree/master/data/raw). 

2. Predictive: What are the strongest predictors of people who survived on the Titanic?

3. To effectively predict the groups of passengers that were more likely to survive, we have decided to classify the data using a decision tree. For the features that we are analysing we are including passenger classes, sex, age, number of siblings/spouses onborad, number of parents/children onbard, and fare prices. We have just started learning about how to do this in our DSCI 571 class and would like to see how much we can learn throughout the course. 

4. To summarize our preliminary analysis on the data set, we would create histograms of distributions for each feature to visualize how each feature affects the target (Survived or not). After the decision tree model is create, we will visuallize the decision tree using the  Graphviz to generate the figure. We will also use the ML model to evaluate the decisiveness of each feature and summarize the results as a table of importance. 
