# Goal: a live window with mirrored video and an FPS counter,
# closing with q. Result: about 30 FPS (the webcam limit).
import cv2
import time # we will use it to calculate the fps

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # .videocapture opens the camera, 0 is the default camera, CAP_DSHOW is a flag for Windows to use DirectShow

if not cap.isOpened(): #check if the camera opened successfully
    raise RuntimeError("Could not open webcam")

prev_time = 0 # bch tkoun reference time loula wbaadha nzidou wahdokhra bch nehsbou lfark binethom

while True:
    success, frame = cap.read()      # grab one picture from the camera, success telkaha boolean true ken l'image est bien capturée, frame c'est l'image capturée
    if not success:
        break

    frame = cv2.flip(frame, 1)       # mirror effect, 0 is vertical flip, 1 is horizontal flip, -1 is both

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # write the FPS on the picture
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), # position of the text, (10, 40) means 10 pixels from the left and 40 pixels from the top, 0,0 is the top-left corner of the image
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #font style 1 is the font size, (0, 255, 0) is the color in BGR format (green), 2 is the thickness of the text

    cv2.imshow("Sign Language Detector", frame) # display the frame in a window named "Sign Language Detector"

    if cv2.waitKey(1) == ord('q'):   # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
