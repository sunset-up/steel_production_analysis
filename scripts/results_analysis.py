import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

#4.1 performance metrics
def calculate_metrics(y_true, y_pred):
    rmse = f'{root_mean_squared_error(y_true, y_pred):.4f}'
    mae = f'{mean_absolute_error(y_true, y_pred):.4f}'
    r2 = f'{r2_score(y_true, y_pred):.4f}'
    results ={
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }
    return results


def create_performance_table(results):
    """Create comparative performance table"""  
    results.to_csv("results/comparative_performance.csv", index=False)


# 4.2 visualize
def plot_model_comparison(results):
    """Create bar plots with error bars"""
    pass


def plot_predictions_vs_actual(y_true, y_pred, model_name):
    """Scatter plot of predictions vs actual values"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, y_pred, c="b", alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val],[min_val, max_val], linestyle="--", c="g", label="ideal line")
    ax.set_title(model_name)
    ax.legend()


def plot_residuals(y_true, y_pred, model_name):
    """Plot residual analysis"""
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, residuals)
    ax.axhline(0, linestyle="--", color="r")
    ax.set_title(model_name+"——residual")


def plot_learning_curve(X_train, y_train, X_val, y_val, model):
    percentages = np.arange(0, 1.1, 0.1)
    sizes = (len(X_train) * percentages).astype(int)
    train_rmse, val_rmse = [], []
    for s in sizes[1:]:
        model.fit(X_train[:s], y_train[:s])
        y_train_pred = model.predict(X_train[:s])
        y_val_pred = model.predict(X_val)
        train_rmse.append(root_mean_squared_error(y_train[:s], y_train_pred))
        val_rmse.append(root_mean_squared_error(y_val, y_val_pred))
    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(percentages[1:], train_rmse, linestyle="-", color="b", linewidth=2, label="Training RMSE")
    ax.plot(percentages[1:], val_rmse, linestyle="--", color="r", linewidth=2, label="Validation RMSE")
    ax.set_title("learning curve")
    ax.legend()

    


