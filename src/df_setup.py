import pandas as pd
from typing import Tuple, Union

def get_column_names(file_path: str) -> Tuple[Union[pd.DataFrame, None], str]:
    """Reads a CSV file and returns the DataFrame and column names as a string."""
    try:
        df = pd.read_csv(file_path)
        column_names_list = df.columns.tolist()
        column_names_str = ', '.join(column_names_list)
        return df, column_names_str
    except Exception as e:
        return None, f"Error: {str(e)}"