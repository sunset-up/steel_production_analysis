import numpy as np
import pandas as pd

# define data loading function
def load_steel_data(file_path):
    '''
    load steel production dataset
    returns: pandas DataFrame
    '''
    data = pd.read_csv(file_path)
    # print(f'data load successfully:{data.shape[0]} rows and {data.shape[1]} columns')
    # print(f'columns:{data.columns.tolist()}')
    return data