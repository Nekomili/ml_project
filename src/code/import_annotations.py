import os
import io
import pandas as pd
import sqlite3

PATH = "../data"


def read_files(path:str) -> pd.DataFrame:
    """Reads data from files in a specified directory and returns it as a Pandas DataFrame.

    The function iterates through each file within the given path, parses its content
    assuming a specific format (class, x-center, y-center, width, height), and creates a dictionary representing each row.
    These dictionaries are then collected into a list, which is finally converted into a Pandas DataFrame.

    Args:
        path (str): The path to the directory containing the data files.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the parsed data from all files in the specified directory.
                      If a file is empty, it creates a row with default values (class=1, x-center=0, etc.).
    """
    name = path.split("_")[-1]
    labels = os.listdir(path)
    list_of_dicts = []

    for label in labels:
        # <object-class> <x-center> <y-center> <width> <height>
        with io.open(f"{path}/{label}", mode='r', encoding='utf-8') as file:
            lines = file.readlines()
            if lines:
                for line in lines:
                    data = line.strip().split(" ")
                    row = {
                        "name":name,
                        "file":label,
                        "class":int(data[0]),
                        "x-center":float(data[1]),
                        "y-center":float(data[2]),
                        "width":float(data[3]),
                        "height":float(data[4])
                    }
                    list_of_dicts.append(row)
            else:
                row = {
                    "name":name,
                    "file":label,
                    "class":int(1),
                    "x-center":float(0),
                    "y-center":float(0),
                    "width":float(0),
                    "height":float(0)
                }
                list_of_dicts.append(row)
    return pd.DataFrame(list_of_dicts)

def save_data(df:pd.DataFrame, table="bronze") -> bool:
    """Saves the DataFrame to an SQLite database.

    This function connects to a specified SQLite database file, creates a table named 'silver' if it doesn't exist, and then writes the data from the input DataFrame into that table. If the table already exists, it replaces its contents.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing the data to be saved.

    Returns:
        bool: True if the data was successfully saved to the database, False otherwise.
    """
    db_file = PATH + "/database/annotations.sqlite"
    table_name = 'silver'
    with sqlite3.connect(db_file) as conn:
        df.to_sql(name=table, con=conn, if_exists='append', index=False)
        return True

def filter_df(df:pd.DataFrame) -> pd.DataFrame:
    """Filters the DataFrame to exclude files listed in image-rejects.txt."""
    rejected = []
    with io.open(PATH + "/" +"image-rejects.txt", mode="r", encoding="utf-8") as file:
        for line in file:
            rejected.append(line.strip().split(".")[0]+".txt")
    rejected_set = set(rejected)

    mask = ~df['file'].isin(rejected_set)
    return df[mask]

def calculate_and_replace_median(group:pd.DataFrame)-> pd.DataFrame:
    """Calculates the median of each specified column in a Pandas DataFrame and replaces the original values with the calculated medians.

    Args:
        group: A Pandas DataFrame containing numerical columns for which the median will be calculated.

    Returns:
        A Pandas DataFrame with the original values in the specified columns replaced by their medians.
    """
    calculated_columns = ["x-center", "y-center", "width", "height"]
    for column in calculated_columns:
        tmp = group[column].median()
        group[column] = tmp
    return group

def write_medians(table="bronze")-> None:
    """Calculates and saves medians for specific classes in an annotations SQLite database.

    This function reads annotations from a SQLite database, calculates the median for each file 
    within the 'bronze' table, and saves the resulting medians to a new 'silver' table.
    It also saves the original bronze and class=1 rows.

    Args:
        table (str, optional): The name of the annotations table to read. Defaults to "bronze".
    
    Returns:
        None
    """
    db_file = PATH + "/database/annotations.sqlite"
    with sqlite3.connect(db_file) as conn:
        bronze_df = pd.read_sql_query(f"SELECT * FROM {table} WHERE class != 1", conn)
        class_1_df = pd.read_sql_query(f"SELECT * FROM {table} WHERE class == 1", conn)
    silver_df = bronze_df.groupby('file', group_keys=False)[bronze_df.columns].apply(calculate_and_replace_median)
    save_data(df=silver_df, table="silver")
    save_data(df=class_1_df, table="silver")
    print(f"Medians calculated for {len(silver_df) + len(class_1_df)} rows and they were saved to database")



if __name__ == "__main__":
    all_content = os.listdir(PATH)
    label_dirs = [item for item in all_content if "labels_" in item]
    print(f"Found {len(label_dirs)} directories with labels")
    print("Starting import...")
    all_df = pd.concat([read_files(f"{PATH}/{label}") for label in label_dirs])
    print(f"Read {len(all_df)} rows")
    filtered_df = filter_df(all_df)
    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Saving to database...")
    save_data(filtered_df, table="bronze")
    print(f"Saved successfully")
    write_medians()