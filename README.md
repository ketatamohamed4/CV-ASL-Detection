# Sign Language Letter Launcher

This project reads a sign language letter from my webcam and opens an application depending on the letter. For example, signing S opens Spotify.

## How it works

The webcam gives a picture many times per second. For each picture, MediaPipe finds my hand and gives me 21 points on it (the wrist, the knuckles, the fingertips). I don't work with the picture itself, only with those points, because they describe the shape of the hand without the background, the lighting or the skin color.

The points are then cleaned up so that the position of the hand in the picture and its distance from the camera don't matter. The wrist becomes the center, and everything is divided by the size of the hand. After that, only the shape of the hand is left.

A trained model looks at these numbers and guesses which letter it is. To avoid opening apps by accident, the program only accepts a letter if it is seen in most of the last 10 frames, and I have to hold it for 1.5 seconds. After an app opens, there is a 5 second pause before another one can open.

## Why a public dataset

My professor said not to train the model on pictures I took myself, because a good dataset needs variety (different people, hands, lighting and backgrounds) and I can't produce that alone. So the model is trained on the ASL Alphabet dataset from Kaggle.

## The files

The scripts don't call each other. Each one saves a file that the next one reads, so they have to be run in order when training.

1. download_dataset.py downloads the dataset from Kaggle and prints the folder where the images are. I paste that path into extract_landmarks.py.
2. extract_landmarks.py goes through the images of the letters I chose, finds the hand in each one with MediaPipe, and saves the 21 points and the letter in landmarks.csv.
3. prepare_data.py reads landmarks.csv, cleans up the numbers (wrist as center, divided by hand size) and splits them into training data and test data. It saves prepared_data.npz.
4. train_model.py trains the model on the training data, checks it on the test data, and saves the model as sign_model.joblib.
5. app_launcher.py is the main program. It loads sign_model.joblib, reads the webcam, finds the letter and opens the app.

The other files were for learning or testing. capture_test.py only shows the webcam. hand_detection.py only shows the hand points on the video. live_predict.py shows the model's guess and its confidence, without opening anything.

One thing to remember: the cleaning of the numbers is written in two places, in prepare_data.py and in the normalize function of app_launcher.py. They must stay identical. If I change one, I must change the other, or the model will get different numbers live than the ones it learned from.

## Running it

Open the project folder in VS Code, open a terminal, and type:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    venv\Scripts\activate
    python app_launcher.py

The line should start with (venv) after the second command. Press q in the video window to quit.

At the top of app_launcher.py, DRY_RUN = True means the program only prints what it would open. Set it to False to really open the apps. The letter and app pairs are in the ACTIONS dictionary, in the same file.

To retrain after changing the letters or the data, run these in order:

    python extract_landmarks.py
    python prepare_data.py
    python train_model.py

## What I found

The model was trained on 6 letters (A, B, C, L, S, Y) and got 99.5% on the test data. That number is too optimistic because the test data comes from the same dataset as the training data. On my own webcam the model is much less sure of itself, so I had to lower the confidence threshold (CONF_MIN) from 0.7 to 0.2 for it to work. Also, a sign that is not one of the six letters still gets read as one of them, because the model has no other answer to give.

The C only worked when the video was mirrored before being given to MediaPipe, which means the dataset pictures are in selfie view like a mirror.

## Ideas to improve it

Measure how often it launches by mistake in DRY_RUN mode, raise CONF_MIN as high as still works, try removing the z coordinate, and retrain with most of the alphabet so that unknown signs land on letters that don't do anything.

## Setup

Windows 11, Python 3.11 in a virtual environment. Libraries: opencv-python, mediapipe version 0.10.14 (newer versions removed the part I use), numpy, scikit-learn, pandas, kagglehub.
