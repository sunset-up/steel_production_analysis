import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# define removing duplicates function
def remove_duplicates(df):
    '''
    remove duplicate rows from dataset 
    '''
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    # removed_count = len(df) - len(df_cleaned)
    # print(f"被删除的重复行有{removed_count}行，占比{(removed_count*100)/len(df)}%")
    return df_cleaned


# handle missing values
class HandleMissingValues(BaseEstimator, TransformerMixin):
    '''
    Identify and impute missing values
    
    :param df: dataframe
    :param fillna: fill missing values with specific value
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_handled = X.fillna(X.median(numeric_only=True, skipna=True))
    
        return df_handled


# IQR methods
class DetectOutliers(BaseEstimator, TransformerMixin):
    '''
    Detect outliers using IQR method
    :param df: 说明
    :param columns: 说明
    '''
    def __init__(self):
        # store the computation
        self.stats = {}

    def fit(self, X, y=None):
        # select the numeric columns
        numeric_col = X.select_dtypes(include=[np.number]).columns.tolist()
        self.process_col = numeric_col
        # compute
        for col in self.process_col:
            replace_value = X[col].median()
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            # define the upper and lower bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # store
            self.stats[col] = {
                "lower": lower_bound,
                "upper": upper_bound,
                "replace": replace_value
            }

        return self        

    def transform(self, X):
        X_target = X.copy()

        for col in self.process_col:
            stats = self.stats[col]
            lower = stats["lower"]
            upper = stats["upper"]
            replace_value = stats["replace"]
    
            # marking outliers
            mask = (X_target[col] < lower) | (X_target[col] > upper)
            # outlier_count = mask.sum()  # count the outliers

            # # print the quantity of the replacement
            # if outlier_count > 0:
            #     print(f"Column '{col}' has {outlier_count} outliers replaced by {replace_value}.")
            # replace outliers
            if mask.any():
                X_target.loc[mask, col] = replace_value
        return X_target
        
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
def data_preprocessing_pipeline():
    pipeline = Pipeline(
        [("handle_missing_values", HandleMissingValues()),
         ("detect_outliers", DetectOutliers()),
         ("encode_categorical_variables", EncodeCategoricalVariables()),

    ])
    return pipeline

def process_data(df):
    # remove deplicates
    df_cleaned = remove_duplicates(df)

    pipeline = data_preprocessing_pipeline()
    df_processed = pipeline.fit_transform(df_cleaned)
    return df_processed 