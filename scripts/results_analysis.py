import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from data_loading import load_steel_data
from data_preprocessing import process_data
from eda import save_figure

# 4.1 performance metrics
def calculate_metrics(y_true, y_pred):
    """
    Compute regression performance metrics.

    Returns RMSE, MAE and R2 as floats rounded to 4 decimals.
    """
    rmse = f'{root_mean_squared_error(y_true, y_pred):.4f}'
    mae = f'{mean_absolute_error(y_true, y_pred):.4f}'
    r2 = f'{r2_score(y_true, y_pred):.4f}'
    results ={
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
    }
    return results


# 4.2 visualize
def plot_model_comparison(results):
    """
    Create bar plots with error bars to compare model performance.

    Each bar represents the mean metric value across repeated runs,
    with standard deviation shown as error bars.
    """
    metrics = ['RMSE', 'MAE', 'R2']
    metric_names = {
        "RMSE": "RMSE",
        "MAE": "MAE",
        "R2": r"$R^2$",
    }
    models = list(results.keys())
    x = np.arange(len(models))
    plt.figure(figsize=(10, 8))
    bar_width = 0.15
    for i, metric in enumerate(metrics):
        means = []
        stds = []

        # Compute mean and std for each model
        for model in models:
            values = results[model][metric]
            means.append(np.mean(values))
            stds.append(np.std(values))

        plt.bar(
            x + i * bar_width,
            means,
            width=bar_width,
            yerr=stds,
            capsize=4,
            label=metric_names[metric],
            alpha=0.85
        )
    plt.xticks(x + bar_width * (len(metrics) - 1) / 2, models, rotation=30)
    plt.ylabel("Metric Value")
    plt.title("Model Performance Comparison with Error Bars")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
            
            
def plot_predictions_vs_actual(y_test, X_test, model):
    """
    Plot predicted values against true values for a given model.
    """
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, c="b", alpha=0.7)

    # Ideal prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val],[min_val, max_val], linestyle="--", c="g", label="ideal line")
    plt.title(f"{type(model).__name__} Predictions vs Actual")
    plt.legend()


def plot_residuals(y_test, X_test, model):
    """
    Plot residual analysis
    """
    y_pred = model.predict(X_test)
    residuals = y_pred - y_test

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, residuals)
    plt.axhline(0, linestyle="--", color="r")
    plt.title(f"{type(model).__name__}——residual")


def plot_learning_curve(X_train, y_train, X_val, y_val, model, curve_type="auto"):
    """
    Plot learning curves based on data size or training iterations.

    If curve_type is 'auto', the function selects the appropriate
    curve based on model attributes.
    """
    # Automatically determine learning curve type
    if curve_type == 'auto':
        if hasattr(model, 'loss_curve_'):
            curve_type = 'iteration'
        else:
            curve_type = 'data'

    # -----------------------------------------------
    # Data-size learning curve
    if curve_type == 'data':
        percentages = np.arange(0.1, 1.1, 0.1)
        sizes = (len(X_train) * percentages).astype(int)
        train_rmse, val_rmse = [], []
        # Shuffle training data
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train.iloc[idx]
        for s in sizes:
            model_i = clone(model)
            model_i.fit(X_shuffled[:s], y_shuffled[:s])
            y_train_pred = model_i.predict(X_shuffled[:s])
            y_val_pred = model_i.predict(X_val)
            train_rmse.append(root_mean_squared_error(y_shuffled[:s], y_train_pred))
            val_rmse.append(root_mean_squared_error(y_val, y_val_pred))
        # plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(percentages, train_rmse, linestyle="-", color="b", linewidth=2, label="Training RMSE")
        ax.plot(percentages, val_rmse, linestyle="--", color="r", linewidth=2, label="Validation RMSE")
        ax.set_title(f"{type(model).__name__} Data-Size Learning Curve")
        ax.set_xlabel('Training Set Fraction')
        ax.set_ylabel('RMSE')
        ax.legend()
        ax.grid(True, linestyle='--')

    # -----------------------------------------------
    # Iteration-based learning curve
    elif curve_type == 'iteration':
        model.fit(X_train, y_train)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(model.loss_curve_, color='blue', linewidth=2)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title(f"{type(model).__name__} Iteration Learning Curve")
        ax.grid(True, linestyle='--')


# # Repeated evaluation on fixed test set
def repeated_test_evaluation(
    best_model,
    X_train,
    y_train,
    X_test,
    y_test,
    n_runs=5,
    random_seed_start=0
):
    """
    Repeatedly train and evaluate a model with different random seeds.

    The test set remains fixed to assess performance stability.

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

        # Clone model to avoid parameter carry-over
        model = clone(best_model)

        # Set random state if supported
        if "random_state" in model.get_params():
            model.set_params(random_state=seed)

        # Measure training time (TRAIN SET)
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - train_start

        # Measure inference time (TEST SET)
        infer_start = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - infer_start

        # Performance metrics (TEST SET)
        metrics = calculate_metrics(y_test, y_pred)

        # Store results
        results["RMSE"].append(metrics['RMSE'])
        results["MAE"].append(metrics['MAE'])
        results["R2"].append(metrics['R2'])
        results["Training Time"].append(train_time)
        results["Inference Time"].append(infer_time)

    return results


# Main Workflow
def main():
    """
    Load data, evaluate final models, and visualize performance.
    """
    # data load
    data_train = load_steel_data("normalized_train_data.csv")
    data_test = load_steel_data("normalized_test_data.csv")
    data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)
    # Data preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = process_data(data)
    X_train_final = np.vstack((X_train, X_val))
    y_train_final = np.hstack((y_train, y_val))
    # Load trained final models
    rf_final = joblib.load("results/models/randomforest_final.joblib")
    svr_final = joblib.load("results/models/svr_final.joblib")
    mlp_final = joblib.load("results/models/mlp_final.joblib")
    gpr_final = joblib.load("results/models/gpr_final.joblib")

    final_models = {
            "RandomForest": rf_final,
            "SVR": svr_final,
            "MLP": mlp_final,
            "GPR": gpr_final,
        }
    
    # Prediction and residual plots
    for name, model in final_models.items():
        plot_predictions_vs_actual(y_test, X_test, model)
        save_figure(f"{name.lower()}_predictions_vs_actual.png", "figures/model_performances")
        plot_residuals(y_test, X_test, model)
        save_figure(f"{name.lower()}_residuals.png", "figures/model_performances")

    # Repeated evaluation on test set 
    final_results = {
        name: repeated_test_evaluation(
            model,
            X_train_final,
            y_train_final,
            X_test,
            y_test,
        )
        for name, model in final_models.items()
    }
    
    # Model comparison plot
    plot_model_comparison(final_results)
    save_figure("Model Performance Comparison.png", "figures/model_performances")
    

if __name__ == '__main__':
    main()

    


