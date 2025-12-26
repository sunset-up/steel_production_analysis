import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import eda
from data_loading import load_steel_data
from data_preprocessing import process_data


#################################################################
# 1. Data load
data_train = load_steel_data("data/normalized_train_data.csv")
data_test = load_steel_data("data/normalized_test_data.csv")

#################################################################
# 2. Data processing
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

#################################################################
# 3. Data analysis plots
train_set = pd.concat([X_train_pre, y_train], axis=1)
corr_matrix = train_set.corr()
# plot the matrix correlation heatmap
eda.plot_correlation_matrix(corr_matrix)
plt.show()

# feature distributions
eda.plot_feature_distributions(X_train_pre)
plt.show()

# target variable distribution
eda.plot_target_distributions(y_train)
plt.show()

# box plots
eda.plot_box(train_set)
plt.show()

# pair plots
# Select the most correlated features with the output based on the correlation matrix heatmap
features = ["output", "input1", "input2", "input3", "input4"]
eda.plot_pair(train_set, features=features)
plt.show()

#################################################################
# 4. model training
