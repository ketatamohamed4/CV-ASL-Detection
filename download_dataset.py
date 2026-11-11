import kagglehub

# Download latest version
path = kagglehub.dataset_download("grassknoted/asl-alphabet") 

print("Path to dataset files:", path)
import os # os is used to traverse the directory structure of the downloaded dataset

for root, dirs, files in os.walk(
    path
):  # root is the current directory, dirs is a list of subdirectories, files is a list of files in the current directory,, all are considered a thruple
    if "A" in dirs and "B" in dirs: # if the current directory contains subdirectories named "A" and "B", then we have found the root of the dataset
        print("Use this as DATASET_DIR:", root)
        break
    #
