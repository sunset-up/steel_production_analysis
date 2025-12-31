import seaborn as sns
import matplotlib.pyplot as plt


# plot the matrix correlation heatmap
def plot_correlation_matrix(corr_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")


# plot feature distributions
def plot_feature_distributions(X):
    X.hist(bins=20, figsize=(30, 25), layout=(4, 6))
    plt.suptitle("Feature Distributions", fontsize=20)
    # plt.tight_layout()


# plot target variable distribution
def plot_target_distributions(y):
    plt.figure(figsize=(10, 8))
    plt.hist(y, bins=20, color="skyblue", edgecolor="black")
    plt.title("Target Distributions")
    plt.grid(True)


# plot box plots
def plot_box(X):
    plt.figure(figsize=(10, 8))
    X.boxplot(showfliers=True, rot=45, fontsize=14)
    plt.title("Box plot of features", fontsize=16)


# draw pair plots
def plot_pair(X, features=None):
    sns.pairplot(X if features is None else X[features], diag_kind="hist", height=2.5, aspect=1.2)
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Pair Plot")


