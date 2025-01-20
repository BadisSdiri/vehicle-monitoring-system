import os
import random
import shutil

train_folder = "datasets/driver_behavior/train/"
test_folder = "datasets/driver_behavior/test/"
test_split_ratio = 0.2  
os.makedirs(test_folder, exist_ok=True)
for class_name in os.listdir(train_folder):
    class_train_path = os.path.join(train_folder, class_name)
    class_test_path = os.path.join(test_folder, class_name)
    os.makedirs(class_test_path, exist_ok=True)
    files = os.listdir(class_train_path)
    num_test_files = int(len(files) * test_split_ratio)
    test_files = random.sample(files, num_test_files)
    for file_name in test_files:
        src_path = os.path.join(class_train_path, file_name)
        dst_path = os.path.join(class_test_path, file_name)
        shutil.move(src_path, dst_path)
    print(f"Moved {num_test_files} files from '{class_name}' to the test set.")

print("\nTest set created successfully!")
