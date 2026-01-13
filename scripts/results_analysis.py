import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

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
    metrics = ['RMSE', 'MAE', 'R2']
    metric_names = {
        "RMSE": "RMSE",
        "MAE": "MAE",
        "R2": r"$R^2$",
    }
    models = list(results.keys())
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 8))
    bar_width = 0.15
    for i, metric in enumerate(metrics):
        means = []
        stds = []
        for model in models:
            values = results[model][metric]
            means.append(np.mean(values))
            stds.append(np.std(values))

        ax.bar(
            x + i * bar_width,
            means,
            width=bar_width,
            yerr=stds,
            capsize=4,
            label=metric_names[metric],
            alpha=0.85
        )
    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2, models, rotation=30)
    ax.set_ylabel("Metric Value")
    ax.set_title("Model Performance Comparison with Error Bars")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
            
            
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


def plot_learning_curve(X_train, y_train, X_val, y_val, model, curve_type="auto"):
    if curve_type == 'auto':
        if hasattr(model, 'loss_curve_'):
            curve_type = 'iteration'
        else:
            curve_type = 'data'
    if curve_type == 'data':
        percentages = np.arange(0.1, 1.1, 0.1)
        sizes = (len(X_train) * percentages).astype(int)
        train_rmse, val_rmse = [], []
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
        ax.set_title("Data-Size Learning Curve")
        ax.set_xlabel('Training Set Fraction')
        ax.set_ylabel('RMSE')
        ax.legend()
        ax.grid(True, linestyle='--')
    
    elif curve_type == 'iteration':
        model.fit(X_train, y_train)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(model.loss_curve_, color='blue', linewidth=2)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title('Iteration Learning Curve')
        ax.grid(True, linestyle='--')


if __name__ == '__main__':
    pass

    


