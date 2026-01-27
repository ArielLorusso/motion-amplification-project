
import cv2 as cv
import numpy as np
from collections import deque




# trackbars limit Parameters
MAX_DELAY = 10  
MAX_AMP   = 10  

# separated windows for video and trackbars
controls_disp = "Controls"
cv.namedWindow(controls_disp, cv.WINDOW_NORMAL)
window_disp   = "Motion Analysis"
cv.namedWindow(window_disp  , cv.WINDOW_NORMAL)
def nothing(x):
    pass
cv.createTrackbar("Frame delay",   controls_disp, 1, 60, nothing)
cv.createTrackbar("Amplification", controls_disp, 1, 10, nothing)


video_path = "./circle-sqare.mp4"
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Error opening video")
# Read first frame
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")
print (ret)


# Buffer of past frames
frame_buffer = deque(maxlen=MAX_DELAY + 1)
frame_buffer.append(frame)

background = frame.astype(np.float32)
motion_energy = np.zeros_like(background)


while True:                                 #* 🔁 video Loop 
    ret, frame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)          
        # break         # no loop
        continue

    frame_buffer.append(frame)

    delay = cv.getTrackbarPos("Frame delay",   controls_disp)
    amp   = cv.getTrackbarPos("Amplification", controls_disp)

    # Only compute diff if we have enough history
    if delay < len(frame_buffer):
        old_frame = frame_buffer[-(delay + 1)]      # update buffer

        diff = amp * np.abs(                        # calculate difference
                    frame.astype(np.int16) -
                    old_frame.astype(np.int16)
        )
        # Clip = limit fom 0 to 255  ( 8 bit per chanel) 
        diff = np.clip(diff, 0, 255).astype(np.uint8)
    else:
        # show Black frame untill bufer has enought data
        diff = np.zeros_like(frame)

    cv.imshow(window_disp, diff)

    if cv.waitKey(30) == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()


# Todo   Limit buffer to fix size (delete old ferames)
# Todo   Limit implement FFT  
# Todo   perform segmentation
# Todo   subsample the video frame acording to trackbar settings