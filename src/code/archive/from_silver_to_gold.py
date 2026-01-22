import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

DATABASE = "../data/database/annotations.sqlite"

# Deprecated function - use filter_gold_labels.
def read_silver(table = "silver") -> pd.DataFrame:
    """
    Reads data from the 'silver' table in the annotations SQLite database,
    excluding records where the class is equal to 1 (No material).

    Args:
        table (str, optional): The name of the table to read from. Defaults to "silver".

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the specified table,
                      excluding rows where the 'class' column is equal to 1.
    """
    with sqlite3.connect(DATABASE) as conn:
        # Class 1 = "No material"
        query = f"SELECT * FROM {table} WHERE class != 1"
        return pd.read_sql_query(query, conn)
    
def filter_gold_labels(table = "silver") -> pd.DataFrame:
    """
    Filters the 'silver' table to select the most confident label for each file.

    This function selects one representative record per file based on the
    most frequent class label, excluding records where the class is equal to 1 (No material).
    It uses rowid and group by to achieve this.

    Args:
        table (str, optional): The name of the table to read from. Defaults to "silver".

    Returns:
        pd.DataFrame: A Pandas DataFrame
    """
    sql_query = f"""
        SELECT name, file, class, `x-center`, `y-center`, width, height
        FROM {table}
        WHERE rowid IN (
            SELECT MIN(rowid)
            FROM {table}
            GROUP BY file
            HAVING COUNT(*) = 1

            UNION ALL

            SELECT MIN(rowid)
            FROM {table}
            WHERE file IN (
                SELECT file
                FROM {table}
                GROUP BY file
                HAVING COUNT(*) > 1
            )
            GROUP BY file
            HAVING class = (
                SELECT class
                FROM {table} AS t2
                WHERE t2.file = {table}.file
                GROUP BY class
                ORDER BY COUNT(*) DESC
                LIMIT 1
            )
        );
        """
    with sqlite3.connect(DATABASE) as conn:
        df = pd.read_sql_query(sql_query, conn)
        df = df[df['class'] != 1]
    return df


def save_to_gold(df:pd.DataFrame, table="gold") -> bool:
    """
    Saves a Pandas DataFrame to the 'gold' table in the annotations SQLite database.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        table (str, optional): The name of the table to save to. Defaults to "gold".

    Returns:
        bool: True if the DataFrame was successfully saved to the table, False otherwise.
    """
    with sqlite3.connect(DATABASE) as conn:
        df.to_sql(name=table, con=conn, if_exists='replace', index=False)
        return True
    
def split_dataset(df:pd.DataFrame, random_seed = 42, validation_size = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a Pandas DataFrame into training and validation sets using train_test_split from scikit-learn.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        validation_size (float, optional): Proportion of the data to use for validation. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the training and validation DataFrames.
    """
    return train_test_split(df, test_size=validation_size, random_state=random_seed)

def save_split_data(training_df:pd.DataFrame, validation_df:pd.DataFrame, traingin_table="training", validation_table="validation")-> bool:
    """
    Saves the training and validation DataFrames to separate tables in the annotations SQLite database.

    Args:
        training_df (pd.DataFrame): The DataFrame for the training set.
        validation_df (pd.DataFrame): The DataFrame for the validation set.
        traingin_table (str, optional): The name of the table to save the training data to. Defaults to "training".
        validation_table (str, optional): The name of the table to save the validation data to. Defaults to "validation".

    Returns:
        bool: True if both DataFrames were successfully saved to their respective tables, False otherwise.
    """
    with sqlite3.connect(DATABASE) as conn:
        training_df.to_sql(name=traingin_table, con=conn, if_exists='replace', index=False)
        validation_df.to_sql(name=validation_table, con=conn, if_exists='replace', index=False)
        return True
    
if __name__ == "__main__":
    print("Reading gold data:")
    gold_df = filter_gold_labels()
    print(f"Read {len(gold_df)} lines")
    gold_saved = save_to_gold(gold_df)
    if gold_saved:print("Gold was saved to database")
    training_df, validation_df = split_dataset(gold_df)
    print(f"Dataset is split. Training size {len(training_df)} and validation size {len(validation_df)}")
    training_saved = save_split_data(training_df=training_df, validation_df=validation_df)
    if training_saved:print("Traingin dataset was saved to database")
