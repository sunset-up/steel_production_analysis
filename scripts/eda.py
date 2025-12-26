import seaborn as sns
import matplotlib.pyplot as plt


# plot the matrix correlation heatmap
def plot_correlation_matrix(corr_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")


# plot feature distributions
def plot_feature_distributions(X):
    n_features = X.shape[1]
    n_rows = (n_features + 3) // 4
    X.hist(bins=20, figsize=(20, 16), layout=(5, 5))
    plt.suptitle("Feature Distributions", fontsize=25)
    plt.tight_layout()


# plot target variable distribution
def plot_target_distributions(y):
    plt.figure(figsize=(10, 8))
    plt.hist(y, bins=20, color="skyblue", edgecolor="black")
    plt.title("Target Distributions")
    plt.grid(True)


# plot box plots
def plot_box(X):
    plt.figure(figsize=(20, 16))
    X.boxplot(showfliers=True, rot=45, fontsize=16)
    # set the line width
    ax = plt.gca()
    for item in ax.artists:
        item.set_linewidth(3)  # 设置箱体的线宽
    for line in ax.lines:
        line.set_linewidth(3)
    plt.title("Box plot of features", fontsize=25)


# draw pair plots
def plot_pair(X, features=None):
    plt.figure(figsize=(10, 8))
    sns.pairplot(X if features is None else X[features], diag_kind="hist")
    plt.title("Pair Plot")


