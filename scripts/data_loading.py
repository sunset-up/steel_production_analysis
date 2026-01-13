import numpy as np
import pandas as pd
from pathlib import Path

# define data loading function
def load_steel_data(filename):
    """
    Load steel production dataset from the project's data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file located in the data directory.

    Returns
    -------
    pandas.DataFrame
        Loaded steel production dataset.
    """
    # get the root directory
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir/"data"/filename
    # load data
    data = pd.read_csv(file_path)
    print(f'data load successfully:{data.shape[0]} rows and {data.shape[1]} columns')
    print(f'columns:{data.columns.tolist()}')
    return data

if __name__ == '__main__':
    pass
