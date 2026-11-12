# app_launcher.py is the main program.
# It loads sign_model.joblib, reads the webcam, finds the letter and opens the app.
import os
import sys
import subprocess
import webbrowser
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import cv2
import time
import joblib
import numpy as np
from collections import deque, Counter
import mediapipe as mp

# ---------- SETTINGS YOU CAN CHANGE ----------
# Define actions per operating system. 
# The script auto-detects the OS, so end-users don't need to change anything!
ACTIONS = {
    "win32": {  # Windows
        "S": "spotify",         # Assumes Spotify is in PATH. If not, use full path: "C:\\...\\Spotify.exe"
        "C": "calc.exe",        # Windows Calculator
        "Y": "https://www.youtube.com"
    },
    "darwin": {  # macOS
        "S": "Spotify",
        "C": "Calculator",
        "Y": "https://www.youtube.com"
    },
    "linux": {  # Linux (Fedora/KDE)
        "S": "com.spotify.Client",  # Flatpak Spotify (standard on Fedora)
        "C": "org.kde.kcalc",       # KDE Calculator (works via gtk-launch)
        "Y": "https://www.youtube.com"
    }
}

DRY_RUN = False  # Set to True to test safely, False to really open the apps
HOLD_SECONDS = 1.5  # how long you must hold the letter
COOLDOWN_SECONDS = 5  # pause after each launch
CONF_MIN = 0.4  # minimum confidence to accept a guess
# ---------------------------------------------

# Make model path location-independent
MODEL_PATH = Path(__file__).parent / "sign_model.joblib"
model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

def normalize(landmarks):
    coords = np.array([[p.x, p.y, p.z] for p in landmarks])
    coords = coords - coords[0]
    scale = np.linalg.norm(coords[:, :2], axis=1).max()
    coords = coords / scale
    return coords.flatten().reshape(1, -1)

def launch(letter):
    current_os = sys.platform
    
    # Safety check for unsupported OS
    if current_os not in ACTIONS:
        print(f"Error: OS '{current_os}' is not configured in ACTIONS.")
        return
        
    target = ACTIONS[current_os][letter]
    
    if DRY_RUN:
        print(f"[DRY RUN] OS: {current_os} | Letter '{letter}' -> would open: {target}")
        return

    print(f"Opening: {target} (OS: {current_os})")
    
    # 1. Universal URL handling (works on Windows, macOS, and Linux)
    if target.startswith("http://") or target.startswith("https://"):
        webbrowser.open(target)
        return

    # 2. OS-specific app launching
    if current_os == "win32":
        # Windows: shell=True allows running commands like calc.exe or spotify
        subprocess.Popen(target, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    elif current_os == "darwin":
        # macOS: 'open -a' launches applications by name
        subprocess.Popen(["open", "-a", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        
    elif current_os.startswith("linux"):
        # Linux: gtk-launch uses .desktop files (agnostic to Flatpak/RPM/Snap)
        subprocess.Popen(["gtk-launch", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

# Cross-platform webcam backend
backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
cap = cv2.VideoCapture(0, backend)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

history = deque(maxlen=10)
hold_letter = None
hold_start = 0
last_launch = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirrored input
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    guess_text = "no hand"
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        probs = model.predict_proba(normalize(hand.landmark))[0]
        
        best = probs.argmax()
        guess_text = f"guess: {model.classes_[best]} ({probs[best]:.2f})"
        history.append(model.classes_[best] if probs[best] >= CONF_MIN else None)
    else:
        history.append(None)

    # stable letter = at least 8 of the last 10 frames
    stable = None
    valid = [h for h in history if h is not None]
    if valid:
        top, count = Counter(valid).most_common(1)[0]
        if count >= 8:
            stable = top

    # hold timer
    now = time.time()
    if stable != hold_letter:
        hold_letter = stable
        hold_start = now

    status = ""
    # Check if the stable letter is in the current OS's actions
    if stable in ACTIONS.get(sys.platform, {}):
        if now - last_launch < COOLDOWN_SECONDS:
            status = "cooldown..."
        elif now - hold_start >= HOLD_SECONDS:
            launch(stable)
            last_launch = now
            status = "launched!"
        else:
            status = f"hold {now - hold_start:.1f}/{HOLD_SECONDS}s"

        cv2.putText(
            frame,
            f"Letter: {stable if stable else '-'}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 200, 255),
            3,
        )
    
    cv2.putText(frame, status, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        frame, guess_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
    )
    if DRY_RUN:
        cv2.putText(
            frame, "DRY RUN", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

    cv2.imshow("Sign Language Launcher", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()