#load packages
import pandas as pd
import numpy as np

#read training data
training = pd.read_csv("../data/raw/train.csv", index_col = 0)
print(training.head(10))
