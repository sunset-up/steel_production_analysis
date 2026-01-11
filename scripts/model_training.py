import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import eda
from data_loading import load_steel_data
from data_preprocessing import data_preprocessing_pipeline
from results_analysis import calculate_metrics
from results_analysis import plot_predictions_vs_actual
from results_analysis import plot_learning_curve


#################################################################
# 1. Data load
data_train = load_steel_data("normalized_train_data.csv")
data_test = load_steel_data("normalized_test_data.csv")
# create a data copy
train_ori = data_train.copy()
test_ori = data_test.copy()
data = pd.concat([train_ori, test_ori], axis=0).reset_index(drop=True)
# data['log_input10'] = np.log1p(data['input10'])
# data.drop(columns=['input6','input10','input18','input19', 'input2', 'input4'], inplace=True)
# data['log_input10'] = np.log(data['input10'])
#################################################################
# # 2. Data analysis plots
corr_matrix = data.corr()
# plot the matrix correlation heatmap
eda.plot_correlation_matrix(corr_matrix)
eda.save_figure("correlation_matrix.png")

# feature distributions
eda.plot_feature_distributions(data.iloc[:,1:22])
eda.save_figure("feature_distributions.png")

# target variable distribution
eda.plot_target_distributions(data['output'])
eda.save_figure("target_distributions.png")

# box plots
eda.plot_box(data)
eda.save_figure("box_plots.png")

# pair plots
# Select the most correlated features with the output based on the correlation matrix heatmap
features = ["input1", "input2", "input3", "input4",  "output"]
eda.plot_pair(data, features=features)
eda.save_figure("pair_plot.png")

#################################################################
# 3. Data processing
# labels = data["output"]
# features = data.drop(columns=["output"]) 

# split  data into train / validation / test sets 
train_set, temp_set = train_test_split(data, test_size=0.3, random_state=42)
print(temp_set.shape)
val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)
X_train_split = train_set.drop(columns=['output'])
print(X_train_split.columns.tolist())
y_train = train_set["output"]
X_val_split = val_set.drop(columns=['output'])
y_val = val_set['output']

X_test_split = test_set.drop(columns=['output'])
y_test = test_set['output']

# data preprocessing
pipeline = data_preprocessing_pipeline()
# feature engineering

print("Training set:")
X_train = pipeline.fit_transform(pd.DataFrame(X_train_split))
print("Validation set:")
X_val = pipeline.transform(pd.DataFrame(X_val_split))
print("Test set:")
X_test = pipeline.transform(pd.DataFrame(X_test_split))
# print(f'训练集特征：{X_train.columns.tolist()}')
# eda.plot_feature_distributions(X_train)
# plt.show()
# eda.plot_feature_distributions(X_test)
# plt.show()
# eda.plot_feature_distributions(X_test)
# plt.show()
# feature engineering
# def feature_engineering(df):
#     """
#     根据特征分布 & 相关性设计的特征工程
#     注意：
#     - 所有变换都只依赖输入特征本身
#     - 不使用标签，避免数据泄露
#     """

#     df = df.copy()

#     # --------
#     # (1) 删除强共线特征（根据热力图人工指定）
#     # --------
#     # input1 与 input6 强负相关 → 保留 input6
#     # input4 与 input3 强相关 → 保留 input3
#     # input19 与 input9 强相关 → 保留 input9
#     drop_cols = ["input1", "input4", "input19"]
#     df.drop(columns=drop_cols, inplace=True, errors="ignore")

#     # --------
#     # (2) 偏态特征修正（log 变换）
#     # --------
#     # input10 分布明显右偏
#     if "input10" in df.columns:
#         df["input10_log"] = np.log1p(df["input10"])
#         df.drop(columns=["input10"], inplace=True)

#     # --------
#     # (3) 构造差值 / 比值特征（降低共线性）
#     # --------
#     # 针对相关性较高但都保留的特征
#     if {"input3", "input9"}.issubset(df.columns):
#         df["diff_3_9"] = df["input3"] - df["input9"]

#     if {"input16", "input17"}.issubset(df.columns):
#         df["ratio_16_17"] = df["input16"] / (df["input17"] + 1e-6)

#     # --------
#     # (4) 构造特征簇统计量（RF 非常友好）
#     # --------
#     cluster_1 = [c for c in ["input3", "input9", "input17"] if c in df.columns]
#     if len(cluster_1) >= 2:
#         df["cluster1_mean"] = df[cluster_1].mean(axis=1)
#         df["cluster1_std"] = df[cluster_1].std(axis=1)

#     # --------
#     # (5) 非线性单特征增强
#     # --------
#     if "input21" in df.columns:
#         df["input21_sqrt"] = np.sqrt(df["input21"])

#     return df

# X_train_fe = feature_engineering(X_train)
# X_val_fe = feature_engineering(X_val)
# X_test_fe = feature_engineering(X_test)
# eda.plot_feature_distributions(X_train_fe)
# plt.show()
# eda.plot_feature_distributions(X_val_fe)
# plt.show()
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
     'n_estimators':[2000],
     'max_features':[1.0],
     'max_depth': [15],
     'min_samples_leaf': [2],
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

# importances = rnd_best.feature_importances_
# features_importance = pd.DataFrame({
#     "features": X_train.columns,
#     "importance": importances,
# })
# features_importance = features_importance.sort_values(by="importance", ascending=False)
# print(features_importance)


# # feature select
# rfe = RFE(rnd_best, n_features_to_select=15)
# rfe.fit(X_train_fe, y_train)
# #
# print("Selected features: ", X_train_fe.columns[rfe.support_])
# print("Feature ranking: ", rfe.ranking_)
# X_train_selected = X_train_fe.loc[:, rfe.support_]
# X_val_selected = X_val_fe.loc[:,rfe.support_]

# rnd_best.fit(X_train_selected, y_train)
rnd_y_train_pred = rnd_best.predict(X_train)
# X_val_stable = X_val_fe[selected_features]
rnd_y_val_pred = rnd_best.predict(X_val)
rnd_y_test_pred = rnd_best.predict(X_test)

rnd_train_results = calculate_metrics(y_train, rnd_y_train_pred)
rnd_val_results = calculate_metrics(y_val, rnd_y_val_pred)
rnd_test_results = calculate_metrics(y_test, rnd_y_test_pred)

print("RandomForest:")
print(f'训练集：{rnd_train_results}')
print(f'验证集：{rnd_val_results}')
print(f'r2之差:{(rnd_train_results["r2"]-rnd_val_results["r2"]):.4f}')
print(f'测试集：{rnd_test_results}')

# # plot_predictions_vs_actual(y_val, rnd_y_val_pred, "RandomForest")
# # plt.show()

# plot_learning_curve(X_stable, y_train, X_val_stable, y_val, rnd_best)
# plt.show()


# (2).SVR
scaler = StandardScaler()
X_train_pre = scaler.fit_transform(X_train)
X_val_pre = scaler.transform(X_val)
X_test_pre = scaler.transform(X_test)
svr = SVR()
param_grid = {
     'C':[1.0],
     'epsilon':[0.05, 0.1],
     'kernel':["rbf"],
     'degree':[3],
     'gamma':['scale', 0.05, 0.1],
}
grid_search = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1)

start_time = time.time()
grid_search.fit(X_train, y_train)
svr_train_time = time.time() - start_time
print(f'Training time:{svr_train_time:.2f}s')
svr_selected_params = [
    'C', 'epsilon', 'kernel', 'degree', 'gamma' 
    ]

params = grid_search.best_estimator_.get_params()
for param, value in params.items():
     if param in svr_selected_params:
          print(f'{param}={value}')
svr_bset = grid_search.best_estimator_
svr_train_pred = svr_bset.predict(X_train)
svr_val_pred = svr_bset.predict(X_val)

svr_train_results = calculate_metrics(y_train, svr_train_pred)
svr_val_results = calculate_metrics(y_val, svr_val_pred)

print("SVR:")
print(f'Training set:{svr_train_results}')
print(f'Validation set:{svr_val_results}')


plot_learning_curve(X_train, y_train, X_val, y_val, svr_bset)
plt.show()


# (3).MLP
mlp = MLPRegressor(
     learning_rate='adaptive',
     hidden_layer_sizes=(420, 336, 252, 168, 84),
     max_iter=200,
     activation='relu',
     batch_size='auto',
     alpha=0.01,
     solver='adam',
     random_state=42
)
mlp.fit(X_train, y_train)
mlp_selected_params = [
    'learning_rate', 'hidden_layer_sizes', 'max_iter', 'activation', 'batch_size', 'alpha', 'solver' 
    ]

params = mlp.get_params()
for param, value in params.items():
     if param in mlp_selected_params:
          print(f'{param}={value}')

mlp_train_pred = mlp.predict(X_train)
mlp_val_pred = mlp.predict(X_val)

mlp_train_results = calculate_metrics(y_train, mlp_train_pred)
mlp_val_results = calculate_metrics(y_val, mlp_val_pred)

print("MLP:")
print(f'Training set:{mlp_train_results}')
print(f'Validation set:{mlp_val_results}')

plot_learning_curve(X_train, y_train, X_val, y_val, mlp)
plt.show()


# 4. GPR
pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train_pre)
X_val_pca = pca.transform(X_val_pre)
X_test_pca = pca.transform(X_test_pre)
print(f'可解释方差比:{pca.explained_variance_}')
# scaler = StandardScaler()
# X_train_pre = scaler.fit_transform(X_train)
# X_val_pre = scaler.transform(X_val)
# X_test_pre = scaler.transform(X_test)
# eda.plot_feature_distributions(X_train_pre)
# plt.show()
# eda.plot_feature_distributions(X_val_pre)
# plt.show()
# n_features = 21
# # matern = Matern(length_scale=100, nu=1.0)
kernel = RBF(length_scale_bounds=(1, 50)) + WhiteKernel(noise_level_bounds=(1e-2, 1))
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=5e-2,
    normalize_y=True,
    n_restarts_optimizer=5,
    random_state=42
     )
# param_grid = {
#      'kernel': [rbf],
#      'alpha': [1e-2],
# }
# grid_search = GridSearchCV(gpr, param_grid, cv=3, n_jobs=-1)
start_time = time.time()
gpr.fit(X_train_pca, y_train)
train_time = time.time() - start_time
print(f'training time: {train_time:.2f}s')

gpr_selected_params = [
    'kernel', 'alpha', 'normalize_y'
    ]

params = gpr.get_params()
for param, value in params.items():
    if param in gpr_selected_params:
        print(f'{param}={value}')
# gpr_best = grid_search.best_estimator_
gpr_train_pred = gpr.predict(X_train_pca)
gpr_val_pred = gpr.predict(X_val_pca)
gpr_test_pred = gpr.predict(X_test_pca)

gpr_train_results = calculate_metrics(y_train, gpr_train_pred)
gpr_val_results = calculate_metrics(y_val, gpr_val_pred)
gpr_test_results = calculate_metrics(y_test, gpr_test_pred )

print("GPR:")
print(f'Training set:{gpr_train_results}')
print(f'Validation set:{gpr_val_results}')
print(f'Test set:{gpr_test_results}')
