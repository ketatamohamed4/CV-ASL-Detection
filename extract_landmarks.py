# extract_landmarks.py goes through the images of the letters I chose,
#  finds the hand in each one with MediaPipe, 
# and saves the 21 points and the letter in landmarks.csv.
import random
import cv2
import os
import csv # csv is used to write the landmarks to a csv file
import mediapipe as mp


DATASET_DIR = r"C:\Users\LOQ\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1\asl_alphabet_train\asl_alphabet_train"

LETTERS = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y",]
MAX_PER_LETTER = 1500  # max per letter is the maximum images to be used for each letter, it can be bigger but for testing we only need 500

mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(
    static_image_mode=True,  
    max_num_hands=1,
    min_detection_confidence=0.5,
)

with open("landmarks.csv", "w", newline="") as f: # open a new csv file called landmarks.csv in write mode, newline="" is used to avoid blank lines between rows
    writer = csv.writer(f) # create a csv writer object to write to the file

    # first row = column names: label, x0, y0, z0, x1, y1, z1, ...
    header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] # create a list of column names, label is the first column, then x0, y0, z0 for the first landmark, x1, y1, z1 for the second landmark, and so on up to x20, y20, z20 for the 21st landmark  
    writer.writerow(header)

    for letter in LETTERS:
        folder = os.path.join(DATASET_DIR, letter) # create the path to the folder for the current letter, e.g. C:\Users\LOQ\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1\asl_alphabet_train\asl_alphabet_train\A for letter A
        all_files = os.listdir(folder)
        files = random.sample(all_files, min(MAX_PER_LETTER, len(all_files))) # get the list of files in the folder and limit it to MAX_PER_LETTER, e.g. 500 images for letter A
        detected = 0 # counter for the number of images where a hand was detected

        for name in files:
            img = cv2.imread(os.path.join(folder, name))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                points = results.multi_hand_landmarks[0].landmark
                row = [letter] # create a new row for the csv file, starting with the letter label
                for p in points:
                    row += [p.x, p.y, p.z]
                writer.writerow(row)
                detected += 1

        print(f"{letter}: hand found in {detected}/{len(files)} images")

print("Done. File saved as landmarks.csv")
