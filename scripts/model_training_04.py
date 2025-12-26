import pandas as pd
import matplotlib.pyplot as plt
from data_loading_01 import load_steel_data
from data_preprocessing_02 import process_data
from sklearn.model_selection import train_test_split
data_train = load_steel_data("data/normalized_train_data.csv")
data_test = load_steel_data("data/normalized_test_data.csv")

####################################################
# Data preparing
####################################################
# create a data copy
train_ori = data_train.copy()
test_ori = data_test.copy()
y_ori = train_ori["output"]
X_ori = train_ori.drop(columns=["output"])
y_test = test_ori["output"]
X_test = test_ori.drop(columns=["output"])

# split  data into train / validation / test sets 
X_train, X_val, y_train, y_val = train_test_split(X_ori, y_ori, test_size=0.2, random_state=42)

# data preprocessing
X_train_pre = process_data(X_train)
X_val_pre = process_data(X_val)