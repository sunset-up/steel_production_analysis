import time
from pathlib import Path
from datetime import datetime
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
import eda
from data_loading import load_steel_data
from data_preprocessing import data_preprocessing_pipeline
from results_analysis import calculate_metrics
from results_analysis import plot_predictions_vs_actual
from results_analysis import plot_learning_curve


# model training funtions
###################################################
def build_random_forest(**params):
     return RandomForestRegressor(
          random_state=42,
          n_jobs=-1,
          **params
     )


def build_svr(**params):
     return SVR(**params)


def build_mlp(**params):
     return MLPRegressor(
          max_iter=2000,
          random_state=42,
          **params
     )


def build_gpr(**params):
     return GaussianProcessRegressor(**params)


def train_with_search(
    model_builder,
    param_space,
    X_train,
    y_train,
    X_val,
    y_val,
    search_method="grid",   # "grid" | "random" | "bayes"
    cv=5,
    n_iter=30,
    scoring="neg_mean_squared_error",
    log_name="Training_log.txt"
):
    
    model = model_builder()

    if search_method == "grid":
        searcher = GridSearchCV(
            estimator=model,
            param_grid=param_space,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

    elif search_method == "random":
        searcher = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=-1
        )

    searcher.fit(X_train, y_train)
    best_model = searcher.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    train_results = calculate_metrics(y_train, y_train_pred)
    val_results = calculate_metrics(y_val, y_val_pred)

    base_dir = Path(__file__).resolve().parent.parent
    log_path = base_dir /"results"/"log"/log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write(f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Model: {type(searcher.best_estimator_).__name__}\n")
        f.write(f"Search Method: {search_method}\n")
        f.write(f"CV: {cv}\n")
        f.write(f"Scoring: {scoring}\n\n")

        f.write("Best Params:\n")
        f.write(str(searcher.best_params_) + "\n\n")

        f.write("Best CV Score:\n")
        f.write(str(searcher.best_score_) + "\n\n")

        f.write("Train Metrics:\n")
        f.write(str(train_results) + "\n\n")

        f.write("Validation Metrics:\n")
        f.write(str(val_results) + "\n")
        f.write("=" * 40 + "\n\n")
    return {
        "best_model": searcher.best_estimator_,
        "best_params": searcher.best_params_,
        "best_score": searcher.best_score_,
        "train_results": train_results,
        "val_results": val_results,
    }


def print_results(results):
     print(f"======{type(results['best_model']).__name__}======")
     print(f"Best Params:{results['best_params']}")
     print(f"Best Score:{results['best_score']}")
     print(f"Training Perfomance:{results['train_results']}")
     print(f"Validation Performance:{results['val_results']}")


# evaluate models
def repeated_test_evaluation(
    base_model,
    X_train,
    y_train,
    X_test,
    y_test,
    n_runs=5,
    random_seed_start=0
):
    """
    Repeatedly train and evaluate a model using cloned instances,
    while keeping the test set fixed.

    Parameters
    ----------
    base_model : sklearn estimator (already instantiated)
        Base model with chosen hyperparameters.
    n_runs : int
        Number of repeated runs with different random seeds.

    Returns
    -------
    dict: metric_name -> list of values
    """

    results = {
        "RMSE": [],
        "MAE": [],
        "R2": [],
        "Training Time": [],
        "Inference Time": []
    }

    for i in range(n_runs):
        seed = random_seed_start + i

        # ---- Clone model to avoid parameter carry-over ----
        model = clone(base_model)

        # ---- Set random state if supported ----
        if "random_state" in model.get_params():
            model.set_params(random_state=seed)

        # ---- Measure training time (TRAIN SET) ----
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - train_start

        # ---- Measure inference time (TEST SET) ----
        infer_start = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - infer_start

        # ---- Performance metrics (TEST SET) ----
        metrics = calculate_metrics(y_test, y_pred)

        # ---- Store results ----
        results["RMSE"].append(metrics['rmse'])
        results["MAE"].append(metrics['mae'])
        results["R2"].append(metrics['r2'])
        results["Training Time"].append(train_time)
        results["Inference Time"].append(infer_time)

    return results


def main():
    #################################################################
    # 1. Data load
    data_train = load_steel_data("normalized_train_data.csv")
    data_test = load_steel_data("normalized_test_data.csv")
    # create a data copy
    train_ori = data_train.copy()
    test_ori = data_test.copy()
    data = pd.concat([train_ori, test_ori], axis=0).reset_index(drop=True)
    #################################################################
    # 2. Data split
    # split  data into train / validation / test sets(0.7/0.15/0.15)
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
    ##################################################################
    # 3. Data analysis plots
    corr_matrix = train_set.corr()
    # plot the matrix correlation heatmap
    eda.plot_correlation_matrix(corr_matrix)
    eda.save_figure("correlation_matrix.png")

    # feature distributions
    eda.plot_feature_distributions(X_train_split[:,1:22])
    eda.save_figure("feature_distributions.png")

    # target variable distribution
    eda.plot_target_distributions(y_train)
    eda.save_figure("target_distributions.png")

    # box plots
    eda.plot_box(X_train_split)
    eda.save_figure("box_plots.png")

    # pair plots
    # Select the most correlated features with the output based on the correlation matrix heatmap
    features = ["input1", "input2", "input3", "input4",  "output"]
    eda.plot_pair(X_train_split, features=features)
    eda.save_figure("pair_plot.png")

    #################################################################
    # 4.data preprocessing
    pipeline = data_preprocessing_pipeline()
    # feature engineering

    print("Training set:")
    X_train = pipeline.fit_transform(pd.DataFrame(X_train_split))
    print("Validation set:")
    X_val = pipeline.transform(pd.DataFrame(X_val_split))
    print("Test set:")
    X_test = pipeline.transform(pd.DataFrame(X_test_split))
    #################################################################
    # 5. model training
    rf_results = train_with_search(
        build_random_forest,
        param_space={
            'n_estimators':[2000],
            'max_features':[1.0],
            'max_depth': [15],
            'min_samples_leaf': [2],
            'min_samples_split': [2],
            'bootstrap': [True],     
        },
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        search_method="grid",
        cv=5,
        log_name='Training_log.txt'
    )
    print_results(rf_results)

    svr_results = train_with_search(
         build_svr,
         param_space={
            'C':[1.0],
            'epsilon':[0.05, 0.1],
            'kernel':["rbf"],
            'degree':[3],
            'gamma':['scale', 0.05, 0.1],   
         },
         X_train=X_train,
         y_train=y_train,
         X_val=X_val,
         y_val=y_val,
         cv=5,
         log_name='Training_log.txt',
    )
    print_results(svr_results)

    mlp_results = train_with_search(
         build_mlp,
         param_space={
            'learning_rate':['adaptive'],
            'hidden_layer_sizes':[(420, 336, 252, 168, 84)],
            'activation':['relu'],
            'batch_size':['auto'],
            'alpha':[0.01],
            'solver':['adam'],     
         },
         X_train=X_train,
         y_train=y_train,
         X_val=X_val,
         y_val=y_val,
         cv=5,
         log_name='Training_log.txt',
    )
    print_results(mlp_results)

    kernel = RBF(length_scale_bounds=(1, 50)) + WhiteKernel(noise_level_bounds=(1e-2, 1))
    gpr_results = train_with_search(
        build_gpr,
        param_space={
            'kernel':[kernel],
            'alpha':[5e-2],
            'normalize_y':[True],
            'n_restarts_optimizer':[5],
        },
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cv=3,
        log_name='Training_log.txt',
    )
    print_results(gpr_results)



# model_training
if __name__ == "__main__":
    main()
