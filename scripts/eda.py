from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loading import load_steel_data
from data_preprocessing import split_set


# plot the matrix correlation heatmap
def plot_correlation_matrix(corr_matrix):
    """
    Plot a heatmap of the correlation matrix.

    Parameters
    ----------
    corr_matrix : array-like or pandas.DataFrame
        Precomputed correlation matrix to visualize.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")


# plot feature distributions
def plot_feature_distributions(X, bins=20):
    """
    Plot histograms of feature distributions.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix where each column represents a feature.
    bins : int, default=20
        Number of bins for the histograms.
    """
    if isinstance(X, pd.DataFrame):
        data = X.values
        features_name = X.columns.tolist()
    else:
        data = np.asarray(X)
        features_name = [f'Feature {i}' for i in range(data.shape[1])]
    n_features = data.shape[1]
    cols = int(np.ceil(np.sqrt(n_features)))
    rows = int(np.ceil(n_features/cols))
    width = cols * 5
    height = rows  * 4
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(width, height), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    # plot
    for i in range(n_features):
        axes[i].hist(data[:, i], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(features_name[i])
        axes[i].grid(True, linestyle='--')
    # delete redundant subgraphs 
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])
    # plt.tight_layout()


# plot target variable distribution
def plot_target_distributions(y):
    """
    Plot the distribution of the target variable.

    Parameters
    ----------
    y : array-like
        Target values to visualize.
    """
    plt.figure(figsize=(10, 8))
    plt.hist(y, bins=20, color="skyblue", edgecolor="black")
    plt.title("Target Distributions")
    plt.grid(True)


# plot box plots
def plot_box(X):
    """
    Plot box plots for each feature to visualize distributions and outliers.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix where each column represents a feature.
    """
    if isinstance(X, np.ndarray):
        n_features = X.shape[1]
        features_name = [f'Feature{i}' for i in range(1, n_features+1)]
        X = pd.DataFrame(X, columns=features_name)
    plt.figure(figsize=(10, 8))
    X.boxplot(showfliers=True, rot=45, fontsize=14)
    plt.title("Box plot of features", fontsize=16)
    plt.ylabel("Value")
    plt.xlabel("Features")
    plt.grid(True, axis="y", linestyle='--', alpha=0.7)


# draw pair plots
def plot_pair(X, features=None):
    """
    Plot pairwise relationships between features.

    Parameters
    ----------
    X : pandas.DataFrame
        Input dataset containing features.
    features : list of str, optional
        Subset of feature names to include in the pair plot.
    """
    sns.pairplot(X if features is None else X[features], diag_kind="hist", height=2.5, aspect=1.2)
    # plt.subplots_adjust(top=0.9)
    plt.suptitle("Pair Plot")


# save figures
def save_figure(
    filename: str,
    subdir: str = "figures/eda",
    dpi: int = 600,
    close: bool = True,
):
    """
    Save current matplotlib figure safely.

    Parameters
    ----------
    filename : str
        File name, e.g. "correlation_matrix.png"
    subdir : str
        Subdirectory relative to project root
    dpi : int
        Figure resolution
    tight_layout : bool
        Whether to apply tight_layout before saving
    close : bool
        Whether to close the figure after saving
    """

    # get the root directory
    base_dir = Path(__file__).resolve().parent.parent

    # target directory
    save_dir = base_dir / subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    # save path
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=dpi)

    if close:
        plt.close()

    print(f"{filename} saved to: {save_path}")

#  Data analysis plots
def main():
    # Data load
    data_train = load_steel_data("normalized_train_data.csv")
    data_test = load_steel_data("normalized_test_data.csv")
    data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)
    # split data set
    X_train_split, _, _, y_train, _, _ =split_set(data)
    # plot
    train_set = pd.concat([X_train_split, y_train], axis=1)
    corr_matrix = train_set.corr()
    # plot the matrix correlation heatmap
    plot_correlation_matrix(corr_matrix)
    save_figure("correlation_matrix.png")

    # feature distributions
    plot_feature_distributions(X_train_split)
    save_figure("feature_distributions.png")

    # target variable distribution
    plot_target_distributions(y_train)
    save_figure("target_distributions.png")

    # box plots
    plot_box(X_train_split)
    save_figure("box_plots.png")

    # pair plots
    # Select the most correlated features with the output based on the correlation matrix heatmap
    features = ["input1", "input2", "input3", "input4",  "output"]
    plot_pair(train_set, features=features)
    save_figure("pair_plot.png")

if __name__ == '__main__':
    main()

