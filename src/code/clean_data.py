import os

# db file removal

db_file = "../data/database/annotations.sqlite"

if os.path.isfile(db_file):
    try:
        os.remove(db_file)
        print(f"annotations.sqlite deleted.")
    except exception as e:
        print(f"Couldn't delete database: {e}")
else:
    print(f"Database cannot be found or already deleted")


# training data removal, images and annotations 

dirs_to_clean = [
    "../YOLO/datasets/materials/images/train",
    "../YOLO/datasets/materials/images/validation",
    "../YOLO/datasets/materials/labels/train",
    "../YOLO/datasets/materials/labels/validation"
]

for dir in dirs_to_clean:
    if not os.path.exists(dir):
        print(f"Directory not found: {dir}")
        continue
    
    delete_action = False
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                delete_action = True
        except Exception as e:
            print(f"Cannot remove file(s): {file_path}: {e}")
    print(f"Files deleted from {dir}" if delete_action else f"No files to delete in {dir}")