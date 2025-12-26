from data_loading_01 import load_steel_data
from data_preprocessing_02 import process_data

data_train = load_steel_data("data/normalized_train_data.csv")
data_test = load_steel_data("data/normalized_test_data.csv")

# create a data copy
train_ori = data_train.copy()

# data preprocessing
train_pre = process_data(train_ori, replace="mean")
