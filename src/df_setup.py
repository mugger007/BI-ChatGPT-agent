import pandas as pandas

def get_column_names(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Get the column names as a list
        column_names_list = df.columns.tolist()

        # Convert the list of column names to a comma-separated string
        column_names_str = ', '.join(column_names_list)

        # Return the DataFrame and the column names string
        return df, column_names_str

    except Exception as e:
        # Handle exceptions such as file not found or invalid CSV format
        return None, f"Error: {str(e)}"