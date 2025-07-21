# ─────────────────────────────────────────────────────────────
# Path Configuration
# ─────────────────────────────────────────────────────────────
import os
import sys

# Add the 'src' directory to the Python path
SRC_PATH = os.path.abspath(os.path.join("..", "src"))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ─────────────────────────────────────────────────────────────
# Local Imports (from src)
# ─────────────────────────────────────────────────────────────
from imports import *

# ─────────────────────────────────────────────────────────────────────
# Function for changing the nomenclature of the cavities
# ─────────────────────────────────────────────────────────────────────

def adjust_names(name):
    """
    Adjust cavity and system names.

    For cavities that have '_v2' at the end of their value,
    this function removes the last three characters to align with
    the common pristine cavity name.

    Parameters:
    ----------
    name : str
        The name of the cavity or system.

    Returns:
    -------
    str
        The adjusted name.
    """
    if name.endswith("_v2"):
        return name[:-3]  # Remove the last three characters
    else:
        return name

# ─────────────────────────────────────────────────────────────────────
# Function for detecting outliers 
# ─────────────────────────────────────────────────────────────────────

def detect_outliers(df, group_col, value_col):
    """
    Calculate the Interquartile Range (IQR) and identify outliers in a DataFrame.

    Outliers are defined as values outside of 1.5 times the IQR from the quartiles.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    group_col : str
        The column name to group by.
    value_col : str
        The column name containing the values to check for outliers.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the outliers.
    """
    outliers_list = []
    grouped = df.groupby(group_col)
    for name, group in grouped:
        Q1 = group[value_col].quantile(0.25)
        Q3 = group[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = group[
            (group[value_col] < lower_bound) | (group[value_col] > upper_bound)
        ]
        outliers_list.append(outliers)
    return pd.concat(outliers_list)


