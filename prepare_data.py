# prepare_data.py reads landmarks.csv,
# cleans up the numbers (wrist as center, divided by hand size)
# and splits them into training data and test data.
#  It saves prepared_data.npz.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("landmarks.csv") # read the landmarks.csv file into a pandas dataframe

print("Samples per letter:")
print(df["label"].value_counts()) # print the number of samples for each letter in the dataset

labels = df["label"].values
# each row: 63 numbers -> reshape into 21 points of (x, y, z)
coords = df.drop(columns=["label"]).values.reshape(-1, 21, 3) # (-1 means infer the number of rows, 21 is the number of landmarks, 3 is the number of coordinates (x, y, z))

# 1) wrist (point 0) becomes the origin
coords = coords - coords[:, 0:1, :] # subtract the wrist position from all landmarks

# 2) divide by the hand size (largest distance from the wrist, in x and y)
scale = np.linalg.norm(coords[:, :, :2], axis=2).max(axis=1) # calculate the maximum distance from the wrist in the x and y directions for each image
coords = coords / scale[:, None, None] # divide all coordinates by the scale to normalize the hand size

X = coords.reshape(-1, 63) # reshape the coordinates into a 2D array for machine learning

# 80% for training, 20% for testing; stratify keeps the letter proportions equal
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)
mirrored = X_train.reshape(-1, 21, 3).copy()
mirrored[:, :, 0] *= -1  # flip x = mirror image
mirrored = mirrored.reshape(-1, 63)

X_train = np.concatenate([X_train, mirrored])
y_train = np.concatenate([y_train, y_train])

np.savez(
    "prepared_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Saved prepared_data.npz")
