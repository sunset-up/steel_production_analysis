import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
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


def evaluate_models_cv(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
):
    """
    使用 K 折交叉验证统一评估多个回归模型

    Parameters
    ----------
    models : dict
        {"ModelName": model_instance}
    X : np.ndarray
        特征矩阵
    y : np.ndarray
        目标变量
    n_splits : int
        K 折交叉验证的折数
    shuffle : bool
        是否打乱数据
    random_state : int
        随机种子

    Returns
    -------
    results : dict
        {
          "ModelName": {
              "RMSE": [...],
              "MAE": [...],
              "R2": [...],
              "train_time": [...],
              "infer_time": [...]
          }
        }
    """

    results = {}

    kf = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    for model_name, model in models.items():
        results[model_name] = {
            "RMSE": [],
            "MAE": [],
            "R2": [],
            "train_time": [],
            "infer_time": []
        }

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ---------- 训练 ----------
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train

            # ---------- 推理 ----------
            start_infer = time.time()
            y_pred = model.predict(X_test)
            infer_time = time.time() - start_infer

            # ---------- 指标 ----------
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # ---------- 记录 ----------
            results[model_name]["RMSE"].append(rmse)
            results[model_name]["MAE"].append(mae)
            results[model_name]["R2"].append(r2)
            results[model_name]["train_time"].append(train_time)
            results[model_name]["infer_time"].append(infer_time)

    return results

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

    


