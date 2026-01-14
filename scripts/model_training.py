import time
import json
import joblib
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
from data_loading import load_steel_data
from data_preprocessing import data_preprocessing_pipeline
from data_preprocessing import split_set
from eda import save_figure
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


# train the final model with (training set + validation set)
def train_final_model(best_model, X_train, y_train, random_state=42):
    final_model = clone(best_model)
    if "random_state" in final_model.get_params():
        final_model.set_params(random_state=random_state)
    start_time = time.perf_counter()
    final_model.fit(X_train, y_train)
    training_time = float(f"{(time.perf_counter() - start_time):.4f}")
    return final_model, training_time


def main():
    #################################################################
    # 1. Data load
    data_train = load_steel_data("normalized_train_data.csv")
    data_test = load_steel_data("normalized_test_data.csv")
    data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)
    #################################################################
    # 2. Data split
    # split  data into train / validation / test sets(0.7/0.15/0.15)
    X_train_split, X_val_split, X_test_split, y_train, y_val, y_test=split_set(data)
    ##################################################################
    # 3.data preprocessing
    pipeline = data_preprocessing_pipeline()
    print("Training set:")
    X_train = pipeline.fit_transform(pd.DataFrame(X_train_split))
    print("Validation set:")
    X_val = pipeline.transform(pd.DataFrame(X_val_split))
    print("Test set:")
    X_test = pipeline.transform(pd.DataFrame(X_test_split))
    #################################################################
    # 4. model training
    rf_results = train_with_search(
        build_random_forest,
        param_space={
            'n_estimators':[100, 500, 1000],
            'max_features':[1.0, 0.7, 0.5, 'sqrt'],
            'max_depth': [15, 25, 35],
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
            'C':[0.5, 1.0, 10, 100],
            'epsilon':[0.05, 0.1],
            'kernel':["rbf"],
            'degree':[2, 3, 4],
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
            'alpha':[0.05],
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

    kernel = RBF(length_scale_bounds=(1, 1000)) + WhiteKernel(noise_level_bounds=(1e-3, 1))
    gpr_results = train_with_search(
        build_gpr,
        param_space={
            'kernel':[kernel],
            'alpha':[1e-8],
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


    models = {
    "RandomForest": rf_results["best_model"],
    "SVR": svr_results["best_model"],
    "MLP": mlp_results["best_model"],
    "GPR": gpr_results["best_model"],
    }

    # plot the learning curves
   
    for name, model in models.items():
        plot_learning_curve(X_train, y_train, X_val, y_val, model)
        save_figure(f"{name.lower()}_learning_curve.png", "figures/training_performances")

    X_train_final = np.vstack((X_train, X_val))
    y_train_final = np.hstack((y_train, y_val))

    # refit the final model
    final_models = {}
    for name, model in models.items():
        final_model, training_time = train_final_model(model, X_train_final, y_train_final)
        final_models[name] = [final_model, training_time]
        joblib.dump(final_model, f"results/models/{name.lower()}_final.joblib")

    
    performances_results = {}
    for name, final_model in final_models.items():
        start_time = time.perf_counter()
        y_pred = final_model[0].predict(X_test)
        inference_time = float(f"{(time.perf_counter() - start_time):.4f}")
        performance = calculate_metrics(y_test, y_pred)
        performance["Training Time"] = final_model[1]
        performance["Inference Time"] = inference_time
        performances_results[name] = performance

    base_dir = Path(__file__).resolve().parent.parent
    results_path = base_dir/"results"/"table"/"test_performances.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results = pd.DataFrame.from_dict(performances_results, orient="index")
    evaluation_results.to_csv(results_path)




# model_training
if __name__ == "__main__":
    main()
