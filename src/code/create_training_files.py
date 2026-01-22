import os
import io
import sqlite3
import pandas as pd
from shutil import copyfile
from typing import Tuple

def read_data_from_gold(database = "../data/database/annotations.sqlite", training_table = "training", validation_table = "validation")->Tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(database) as conn:
        training_df = pd.read_sql_query(f"SELECT * FROM {training_table};", conn)
        validation_df = pd.read_sql_query(f"SELECT * FROM {validation_table};", conn)
        return training_df, validation_df
    
def write_training_files(training_df:pd.DataFrame, output_dir = "../YOLO/datasets/materials", img_path = "../data/example_images"):
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
        copyfile(os.path.join(img_path,img_filename), os.path.join(train_image_dir, img_filename))

def write_validation_data(validation_df:pd.DataFrame, output_dir = "../YOLO/datasets/materials", img_path = "../data/example_images"):
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
        copyfile(os.path.join(img_path,img_filename), os.path.join(validation_image_dir, img_filename))

if __name__ == "__main__":
    training_df, validation_df = read_data_from_gold()
    write_training_files(training_df=training_df)
    write_validation_data(validation_df=validation_df)
    print("Write successful")

