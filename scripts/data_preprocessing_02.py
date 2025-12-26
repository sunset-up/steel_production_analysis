import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# define removing duplicates function
def remove_duplicates(df):
    '''
    remove duplicate rows from dataset 
    '''
    df_cleaned = df.drop_duplicates()
    return df_cleaned

# handle missing values
class HandleMissingValues(BaseEstimator, TransformerMixin):
    '''
    Identify and impute missing values
    
    :param df: dataframe
    :param fillna: fill missing values with specific value
    '''
    def __init__(self, replace="mean"):
        self.fill_with = replace
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.fill_with=="mean":
            df_handled = X.fillna(X.mean(numeric_only=True))
        elif self.fill_with=="median":
            df_handled = X.fillna(X.median(numeric_only=True))
 
        return df_handled
    

# IQR methods
class DetectOutliers(BaseEstimator, TransformerMixin):
    '''
    Detect outliers using IQR method
    :param df: 说明
    :param columns: 说明
    '''
    def __init__(self, columns, replace="mean"):
        self.columns = columns
        self.replace = replace

    def fit(self, X, y=None):
        return self        
    
    def transform(self, X):

        Q1 = X[self.columns].quantile(0.25)
        Q3 = X[self.columns].quantile(0.75)
        IQR = Q3 - Q1
        # define the upper and lower bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # marking outliers
        X['isoutliers'] = (X[self.columns] < lower_bound) | (X[self.columns] > upper_bound)
    # replace outliers
        if self.replace=="mean":
            mean_value = X.loc[~X['isoutliers'], self.columns].mean()
            X.loc[X['isoutliers'], self.columns] = mean_value
        elif self.replace=="median":
            median_value=X.loc[~X['isoutliers'], self.columns].median()
            X.loc[X['isoutliers'], self.columns] = median_value
        return X
        
# handle the categorical variables
class EncodeCategoricalVariables(BaseEstimator, TransformerMixin):
    """
    Convert categorical columns to numerical
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df_encoded = pd.get_dummies(X)

        return df_encoded


# 检查数据一致性
pass


# creat data preprocessing pipeline
def data_preprocessing_pipeline(replace="mean"):
    pipeline = Pipeline("handle_missing_values", HandleMissingValues(replace),
                         "detect_outliers", DetectOutliers(replace),
                         "encode_categorical_variables", EncodeCategoricalVariables,

    )
    return pipeline

def process_data(df, replace="mean"):
    # remove deplicates
    df_cleaned = remove_duplicates(df)

    pipeline = data_preprocessing_pipeline()
    df_processed = pipeline.fit_transform(df_cleaned)
    return df_processed 