"""This is the module that contains utility function for the spatailtsp
"""

def csv_to_df(csv_file):
    """Converts a CSV file to a pandas DataFrame.

    Args : csv_file(str) : The path to the CSV file.

    Returns:
        pandas.DataFrame: The pandas Dataframe.
    """
    import pandas as pd

    return pd.read_csv(csv_file)

# def 