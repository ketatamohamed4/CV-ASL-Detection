# Same camera loop, plus prediction, a confidence value and a stability rule
import cv2
import time
import joblib
import numpy as np
from collections import deque, Counter
import mediapipe as mp

model = joblib.load("sign_model.joblib")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


def normalize(landmarks):
    """Same steps as prepare_data.py: wrist to origin, then divide by hand size."""
    coords = np.array([[p.x, p.y, p.z] for p in landmarks])
    coords = coords - coords[0]
    scale = np.linalg.norm(coords[:, :2], axis=1).max()
    coords = coords / scale
    return coords.flatten().reshape(1, -1)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

flip_input = False  # press 'f' to switch
history = deque(maxlen=10)  # the last 10 guesses
CONF_MIN = 0.7  # minimum confidence to accept a guess
prev_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # which image do we give to MediaPipe?
    proc = cv2.flip(frame, 1) if flip_input else frame
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    guess_text = "no hand"
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(proc, hand, mp_hands.HAND_CONNECTIONS)

        probs = model.predict_proba(normalize(hand.landmark))[0]
        best = probs.argmax()
        letter = model.classes_[best]
        conf = probs[best]

        guess_text = f"guess: {letter} ({conf:.2f})"
        history.append(letter if conf >= CONF_MIN else None)
    else:
        history.append(None)

    # a letter is "stable" if it fills at least 8 of the last 10 frames
    stable = None
    valid = [h for h in history if h is not None]
    if valid:
        top, count = Counter(valid).most_common(1)[0]
        if count >= 8:
            stable = top

    # always show the user a mirrored picture
    display = proc if flip_input else cv2.flip(proc, 1)

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    mode = "mirrored input" if flip_input else "raw input"
    cv2.putText(
        display,
        f"FPS: {int(fps)}   mode: {mode}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        display, guess_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
    )
    cv2.putText(
        display,
        f"STABLE: {stable if stable else '-'}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 200, 255),
        3,
    )

    cv2.imshow("Live Sign Detector", display)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("f"):
        flip_input = not flip_input
        history.clear()

cap.release()
cv2.destroyAllWindows()
