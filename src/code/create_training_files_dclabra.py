import os
import sqlite3
import pandas as pd
from typing import Tuple

# "/home/jovyan/shared/dataset/" DCLABRA Shared picture folder

def read_data_from_gold(database = "../data/database/annotations.sqlite", training_table = "training", validation_table = "validation")->Tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(database) as conn:
        training_df = pd.read_sql_query(f"SELECT * FROM {training_table};", conn)
        validation_df = pd.read_sql_query(f"SELECT * FROM {validation_table};", conn)
        return training_df, validation_df

def write_training_files(training_df:pd.DataFrame, output_dir = "../YOLO/datasets/materials", img_path = "/home/jovyan/shared/dataset/"):
    train_label_dir = os.path.join(output_dir, "labels", "train")
    train_image_dir = os.path.join(output_dir, "images", "train")
    if not os.path.exists(train_label_dir) or not os.path.exists(train_image_dir):
        print(train_label_dir)
        os.makedirs(train_label_dir)
        os.makedirs(train_image_dir)
    for filename, group_df in training_df.groupby('file'):
        txt_filename = filename
        txt_filepath = os.path.join(train_label_dir, txt_filename)
        img_filename = filename.split('.')[0]+".jpg"
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            for _, row in group_df.iterrows():
                class_id = int(row['class'])
                x_center = row['x-center']
                y_center = row['y-center']
                width = row['width']
                height = row['height']

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        os.symlink(os.path.join(img_path, img_filename), os.path.join(train_image_dir, img_filename))

def write_validation_data(validation_df:pd.DataFrame, output_dir = "../YOLO/datasets/materials", img_path = "/home/jovyan/shared/dataset/"):
    validation_label_dir = os.path.join(output_dir, "labels", "validation")
    validation_image_dir = os.path.join(output_dir, "images", "validation")
    if not os.path.exists(validation_label_dir) or not os.path.exists(validation_image_dir):
        print(validation_label_dir)
        os.makedirs(validation_label_dir)
        os.makedirs(validation_image_dir)
    for filename, group_df in validation_df.groupby('file'):
        txt_filename = filename
        txt_filepath = os.path.join(validation_label_dir, txt_filename)
        img_filename = filename.split('.')[0]+".jpg"
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            for _, row in group_df.iterrows():
                class_id = int(row['class'])
                x_center = row['x-center']
                y_center = row['y-center']
                width = row['width']
                height = row['height']

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        os.symlink(os.path.join(img_path, img_filename), os.path.join(validation_image_dir, img_filename))

def filter_files(tables = {"tra":"training", "val":"validation", "base":"silver"}):
    """
    Filters a list of files, removing those present in the specified training and validation tables from a SQLite database.
    
    This function iterates through a SQLite database to retrieve files from the specified training and validation tables, then filters a provided list of files, excluding those found in the database tables.
    
    Args:
        tables (dict, optional): A dictionary mapping table names to their corresponding SQLite table names. Defaults to {"tra":"training", "val":"validation", "base":"silver"}.
    """

    with sqlite3.connect("../data/database/annotations.sqlite") as conn:
        cursor = conn.cursor()

        cursor.execute(f"SELECT file FROM {tables['tra']}")
        files_training = set([node[0] for node in cursor.fetchall()])
        cursor.execute(f"SELECT file FROM {tables['val']}")
        files_validation = set([node[0] for node in cursor.fetchall()])

        cursor.execute(f"SELECT DISTINCT file FROM {tables['base']}")
        all_files = [node[0] for node in cursor.fetchall()]
        return [node for node in all_files if node not in files_training and node not in files_validation]
    

def create_bg_training_symlinks(bg_files:list[str], output_dir = "../YOLO/datasets/materials", img_path = "/home/jovyan/shared/dataset/"):
    """
    Creates symbolic links from background images to the training image directory.

    This function iterates through a list of background image filenames,
    creates symbolic links pointing to the corresponding image files within
    the specified image directory.  It ensures the training image directory exists.

    Args:
        bg_files: A list of background image filenames (including extension).
        output_dir: The root directory for the training dataset. Defaults to "../YOLO/datasets/materials".
        img_path: The path to the directory containing the original image files. Defaults to "/home/jovyan/shared/dataset/".
    """
    train_image_dir = os.path.join(output_dir, "images", "train")
    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)
    for filename in bg_files:
        img_filename = filename.split('.')[0]+".jpg"
        os.symlink(os.path.join(img_path, img_filename), os.path.join(train_image_dir, img_filename))




if __name__ == "__main__":
    training_df, validation_df = read_data_from_gold()
    write_training_files(training_df=training_df)
    write_validation_data(validation_df=validation_df)
    bg_files = filter_files()
    create_bg_training_symlinks(bg_files=bg_files)