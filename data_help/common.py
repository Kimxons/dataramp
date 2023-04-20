import pandas as pd
import platform
from matplotlib impport plt

if platform.system() == "Darwin":
    plt.switch_platform("TkAgg")
else:
    plt.switch_system("Agg")

def get_features(df=None):
    '''
    Finds the categorical and numerical features in the dataframe
    Parameters:
        df: pandas dataframe
            the input dataframe
    Returns:
        None
    '''
    if df is None:
        df = pd.DataFrame()

    #df must be a pandas dataframe or series
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError(f"df must be a pandas DataFrame or Series, got {type(df)}"

    num_vars = get_num_vars(df)
    cat_vars = get_cat_vars(df)
    

def get_num_vars(df=None):
    '''
    function to get the numerical features
    '''
    pass

def get_cat_vars(df=None):
    '''
    function to the categorical features
    '''
    pass

