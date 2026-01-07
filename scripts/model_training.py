import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit
import eda
from data_loading import load_steel_data
from data_preprocessing import data_preprocessing_pipeline
from data_preprocessing import process_data
from results_analysis import calculate_metrics
from results_analysis import plot_predictions_vs_actual
from results_analysis import plot_learning_curve


#################################################################
# 1. Data load
data_train = load_steel_data("data/normalized_train_data.csv")
data_test = load_steel_data("data/normalized_test_data.csv")
# create a data copy
train_ori = data_train.copy()
test_ori = data_test.copy()
data = pd.concat([train_ori, test_ori], axis=0).reset_index(drop=True)
#################################################################
# 2. Data processing
# epsilon = 1e-8
# data["input6/input1"] = data["input6"] / (data["input1"] + epsilon)
# data["input19/input5"] = data["input19"] / (data["input5"] + epsilon)
# data["input3/input7"] = data["input3"] / (data['input7'] + epsilon)
# data["input9/input18"] = data['input9'] / (data["input18"] + epsilon)
labels = data["output"]
features = data.drop(columns=["output"]) 

# split  data into train / validation / test sets 
X_train_split, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val_split, X_test_split, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# data preprocessing
pipeline = data_preprocessing_pipeline()
# feature engineering

print("Training set:")
X_train = pipeline.fit_transform(pd.DataFrame(X_train_split))
print("Validation set:")
X_val = pipeline.transform(pd.DataFrame(X_val_split))
print("Test set:")
X_test = pipeline.transform(pd.DataFrame(X_test_split))

#################################################################
# # 3. Data analysis plots
train_set = pd.concat([X_train, y_train], axis=1)
corr_matrix = train_set.corr()
# plot the matrix correlation heatmap
eda.plot_correlation_matrix(corr_matrix)
plt.show()
# plt.savefig("figures/eda/correlation_matrix.png", dpi=600)

# feature distributions
eda.plot_feature_distributions(X_val)
# plt.savefig("figures/eda/feature_distributions.png", dpi=600)
plt.show()

# # target variable distribution
# eda.plot_target_distributions(y_train)
# plt.savefig("figures/eda/target_distributions.png", dpi=600)

# # box plots
# eda.plot_box(train_set)
# plt.savefig("figures/eda/box_plots.png", dpi=600)

# # pair plots
# # Select the most correlated features with the output based on the correlation matrix heatmap
# features = ["input1", "input2", "input3", "input4",  "output"]
# eda.plot_pair(train_set, features=features)
# plt.savefig("figures/eda/pair_plot.png", dpi=600)

#################################################################
# 4. model training
# (1). random forest
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_train_rnd = poly.fit_transform(X_train)
# X_val_rnd = poly.transform(X_val)
# pca = PCA(n_components=0.95)
# X_train_pca = pca.fit_transform(X_train)
# X_val_pca = pca.transform(X_val)
# print(f'可解释方差比:{pca.explained_variance_}')
# feature combine

# print(X_train.columns.tolist())
rnd_reg = RandomForestRegressor(random_state=42)
param_grid = {
     'n_estimators':[100, 200],
     'max_features':[1.0],
     'max_depth': [25],
     'min_samples_leaf': [1],
     'min_samples_split': [2],
     'bootstrap': [True],

}
grid_search = GridSearchCV(rnd_reg, param_grid, cv=5, n_jobs=-1)


start_time = time.time()
grid_search.fit(X_train, y_train)
train_time = time.time() - start_time
print(f'训练时间：{train_time}')
rnd_best = grid_search.best_estimator_

selected_params = [
    'n_estimators', 'max_features', 'max_depth', 'min_samples_leaf', 
    'min_samples_split', 'bootstrap'
    ]

params = rnd_best.get_params()
for param, value in params.items():
     if param in selected_params:
          print(f'{param}={value}')

importances = rnd_best.feature_importances_
features_importance = pd.DataFrame({
    "features": X_train.columns,
    "importance": importances,
})
features_importance = features_importance.sort_values(by="importance", ascending=False)
print(features_importance)


# # feature select
# rfe = RFE(rnd_best, n_features_to_select=5)
# rfe.fit(X_train, y_train)
# #
# print("Selected features: ", X_train.columns[rfe.support_])
# print("Feature ranking: ", rfe.ranking_)
# X_train_selected = X_train.loc[:, rfe.support_]
# X_val_selected = X_val.loc[:,rfe.support_]

# rnd_best.fit(X_train, y_train)
rnd_y_train_pred = rnd_best.predict(X_train)
rnd_y_val_pred = rnd_best.predict(X_val)
# rnd_y_test_pred = rnd_reg.predict(X_test)

rnd_train_results = calculate_metrics(y_train, rnd_y_train_pred)
rnd_val_results = calculate_metrics(y_val, rnd_y_val_pred)
# rnd_test_results = calculate_metrics(y_test, rnd_y_test_pred)

print("RandomForest:")
print(f'训练集：{rnd_train_results}')
print(f'验证集：{rnd_val_results}')
print(f'r2之差:{(rnd_train_results["r2"]-rnd_val_results["r2"]):.4f}')
# print(f'测试集：{rnd_test_results}')

# plot_predictions_vs_actual(y_val, rnd_y_val_pred, "RandomForest")
# plt.show()

# plot_learning_curve(X_train, y_train, X_val, y_val, rnd_best)
# plt.show()


# (2).SVR
# svr = SVR()
# param_grid = {
#      'C':[1, 10, 100],
#      'epsilon':[0.01, 0.05, 0.1],
#      'kernel':["rbf"],
#      'degree':[3],
#      'gamma':["scale", 0.01, 0.1, 1],
# }
# grid_search = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1)
# start_time = time.time()
# grid_search.fit(X_train, y_train)
# svr_train_time = time.time() - start_time
# print(f'Training time:{svr_train_time:.2f}s')
# svr_selected_params = [
#     'C', 'epsilon', 'kernel', 'degree', 'gamma' 
#     ]

# params = svr.get_params()
# for param, value in params.items():
#      if param in svr_selected_params:
#           print(f'{param}={value}')
# svr_bset = grid_search.best_estimator_
# svr_train_pred = svr_bset.predict(X_train)
# svr_val_pred = svr_bset.predict(X_val)

# svr_train_results = calculate_metrics(y_train, svr_train_pred)
# svr_val_results = calculate_metrics(y_val, svr_val_pred)

# print("SVR:")
# print(f'Training set:{svr_train_results}')
# print(f'Validation set:{svr_val_results}')


# # (3).MLP
# mlp = MLPRegressor()
# mlp.fit(X_train, y_train)


# mlp_train_pred = mlp.predict(X_train)
# mlp_val_pred = mlp.predict(X_val)

# mlp_train_results = calculate_metrics(y_train, mlp_train_pred)
# mlp_val_results = calculate_metrics(y_val, mlp_val_pred)

# print("MLP:")
# print(f'Training set:{mlp_train_results}')
# print(f'Validation set:{mlp_val_results}')


# # 4. GPR
# gpr = GaussianProcessRegressor()
# gpr.fit(X_train, y_train)

# gpr_train_pred = gpr.predict(X_train)
# gpr_val_pred = gpr.predict(X_val)

# gpr_train_results = calculate_metrics(y_train, gpr_train_pred)
# gpr_val_results = calculate_metrics(y_val, gpr_val_pred)

# print("GPR:")
# print(f'Training set:{gpr_train_results}')
# print(f'Validation set:{gpr_val_results}')
