# Sign Language Letter Launcher

This project reads a sign language letter from my webcam and opens an application depending on the letter. For example, signing S opens Spotify.

It runs on Windows, macOS and Linux without changing any code.

## How it works

The webcam gives a picture many times per second. For each picture, MediaPipe finds my hand and gives me 21 points on it (the wrist, the knuckles, the fingertips). I don't work with the picture itself, only with those points, because they describe the shape of the hand without the background, the lighting or the skin color.

The points are then cleaned up so that the position of the hand in the picture and its distance from the camera don't matter. The wrist becomes the center, and everything is divided by the size of the hand. After that, only the shape of the hand is left.

A trained model looks at these numbers and guesses which letter it is. To avoid opening apps by accident, the program only accepts a letter if it is seen in most of the last 10 frames, and I have to hold it for 1.5 seconds. After an app opens, there is a 5 second pause before another one can open.

The video is always mirrored before being given to MediaPipe. There is nothing to choose here: the model was trained on the dataset pictures and on their mirrored versions, so it already knows both orientations.

## Cross-platform support

The program detects the operating system with `sys.platform` and picks the right way to open things, so nobody has to edit the code to run it on their machine.

- The webcam uses DirectShow on Windows and V4L2 on Linux.
- Links (like YouTube) are opened with Python's `webbrowser` module, which works everywhere.
- Apps are opened differently on each system: a command on Windows, `open -a` on macOS, and `gtk-launch` on Linux.

`gtk-launch` starts an app from its `.desktop` ID, which every app registers no matter how it was installed (Flatpak, RPM, Snap, AUR). That is why the Linux entries look like `com.spotify.Client` and `org.kde.kcalc` instead of `spotify` and `kcalc`.

The `ACTIONS` dictionary in `app_launcher.py` holds one block per system:

| Letter | Windows | macOS | Linux |
| ------ | ------- | ----- | ----- |
| S | `spotify` | `Spotify` | `com.spotify.Client` |
| C | `calc.exe` | `Calculator` | `org.kde.kcalc` |
| Y | YouTube link | YouTube link | YouTube link |

If an app does not open on Linux, its `.desktop` ID is probably different. To find it:

```bash
ls /usr/share/applications/ /var/lib/flatpak/exports/share/applications/ ~/.local/share/applications/ 2>/dev/null | grep -i spotify
```

Take the name it prints, remove `.desktop`, and put that in the Linux block of `ACTIONS`.

## Why a public dataset

My professor said not to train the model on pictures I took myself, because a good dataset needs variety (different people, hands, lighting and backgrounds) and I can't produce that alone. So the model is trained on the ASL Alphabet dataset from Kaggle.

## The files

The scripts don't call each other. Each one saves a file that the next one reads, so they have to be run in order when training.

- `download_dataset.py` downloads the dataset from Kaggle and prints the folder where the images are. I paste that path into `extract_landmarks.py`.
- `extract_landmarks.py` goes through the images of the letters I chose, finds the hand in each one with MediaPipe, and saves the 21 points and the letter in `landmarks.csv`.
- `prepare_data.py` reads `landmarks.csv`, cleans up the numbers (wrist as center, divided by hand size) and splits them into training data and test data. It saves `prepared_data.npz`.
- `train_model.py` trains the model on the training data, checks it on the test data, and saves the model as `sign_model.joblib`.
- `app_launcher.py` is the main program. It loads `sign_model.joblib`, reads the webcam, finds the letter and opens the app.

The other files were for learning or testing. `capture_test.py` only shows the webcam. `hand_detection.py` only shows the hand points on the video. `live_predict.py` shows the model's guess and its confidence, without opening anything.

One thing to remember: the cleaning of the numbers is written in two places, in `prepare_data.py` and in the `normalize` function of `app_launcher.py`. They must stay identical. If I change one, I must change the other, or the model will get different numbers live than the ones it learned from.

## Setup

Python 3.11 in a virtual environment. Libraries: `opencv-python`, `mediapipe` 0.10.14, `numpy`, `scikit-learn`, `pandas`, `kagglehub`, `joblib`.

MediaPipe 0.10.14 is the version I use because newer releases removed `mp.solutions`, which is the part this project is built on. That version has no package for Python 3.12 or 3.13, so on a recent distro like Fedora you have to install Python 3.11 first and build the virtual environment with it:

```bash
sudo dnf install python3.11
python3.11 -m venv venv
source venv/bin/activate
pip install mediapipe==0.10.14 opencv-python numpy scikit-learn pandas kagglehub joblib
```

On Windows, Python 3.11 can be installed from python.org, then:

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
pip install mediapipe==0.10.14 opencv-python numpy scikit-learn pandas kagglehub joblib
```

## Running it

Open the project folder in VS Code and open a terminal.

On Windows:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
python app_launcher.py
```

On Linux or macOS:

```bash
source venv/bin/activate
python app_launcher.py
```

The line should start with `(venv)` after the activation command. Press `q` in the video window to quit.

At the top of `app_launcher.py`, `DRY_RUN = True` means the program only prints what it would open. Set it to `False` to really open the apps. It is a good idea to leave it on `True` the first time, to check that the letters are read correctly before anything starts opening.

To retrain after changing the letters or the data, run these in order:

```bash
python extract_landmarks.py
python prepare_data.py
python train_model.py
```

## What I found

The model was trained on 6 letters (A, B, C, L, S, Y) and got 99.5% on the test data. That number is too optimistic because the test data comes from the same dataset as the training data. On my own webcam the model is much less sure of itself, so I had to lower the confidence threshold (`CONF_MIN`) for it to work. Also, a sign that is not one of the six letters still gets read as one of them, because the model has no other answer to give.

Since the training data contains both the original pictures and their mirrored versions, the C works like the other letters and there is no orientation to adjust.

## Ideas to improve it

Measure how often it launches by mistake in `DRY_RUN` mode, raise `CONF_MIN` as high as still works, try removing the z coordinate, and retrain with most of the alphabet so that unknown signs land on letters that don't do anything.

## Status

Tested on Windows 11 and on Fedora with KDE Plasma. The macOS branch is written but not tested yet.
