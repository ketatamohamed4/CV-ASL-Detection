#Goal: draw the 21 hand landmarks on the live video.
import cv2
import time
import sys
import mediapipe as mp

mp_hands = mp.solutions.hands # Initialize the hand detection model
mp_draw = mp.solutions.drawing_utils # Initialize the drawing utilities

hands = mp_hands.Hands(
    static_image_mode=False,  # False = video (tracks the hand between frames)
    max_num_hands=1,  # we only need one hand
    min_detection_confidence=0.7,  # how sure MediaPipe must be to say "this is a hand"
    min_tracking_confidence=0.5, # how sure MediaPipe must be to say "this is the same hand as before"
)


backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
cap = cv2.VideoCapture(0, backend)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

prev_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # MediaPipe expects RGB colors, but OpenCV gives BGR, so we convert
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb) # Process the frame and detect hands

    # If a hand was found, draw its 21 points and connections
    if results.multi_hand_landmarks: # Check if any hands were detected
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw the hand landmarks and connections on the frame, frame is the image, hand_landmarks is the detected hand landmarks, mp_hands.HAND_CONNECTIONS is the predefined connections between the landmarks

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    cv2.putText(
        frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
