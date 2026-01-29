import cv2 as cv
import numpy as np
from collections import deque  # frame buffer datastruct
from scipy import fft


import cProfile
import pstats



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
cv.createTrackbar("Frame delay",   controls_disp, 1, MAX_DELAY, nothing)
cv.createTrackbar("Amplification", controls_disp, 1, MAX_AMP,   nothing)


with cProfile.Profile() as profile:

    video_path = "./circle-sqare.mp4"
    cap = cv.VideoCapture(video_path)                   # Open video
    if not cap.isOpened():
        raise RuntimeError("Error opening video")
    sucess, frame = cap.read()                          # Read first frame
    if not sucess:
        raise RuntimeError("Could not read first frame")

    frame_buffer = deque(maxlen=MAX_DELAY + 1)          # deque =  Double-Ended Queue  ( Add and remove elements from both ends )
    frame_buffer.append(frame)                          # append first frame to buffer

    #============================================================================================

    while True:                                 #* 🔁 video Loop 
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)   # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html         
            # index of the frame to be decoded/captured next = 0  LOOPS THE VIDEO
            # break         # no loop
            continue     # infinite loop

        # current_frame_number = cap.get(cv.CAP_PROP_POS_FRAMES)
        # print ( current_frame_number )
        frame_buffer.append(frame)
        # print ( len(frame_buffer)    )

        delay = cv.getTrackbarPos("Frame delay",   controls_disp)
        amp   = cv.getTrackbarPos("Amplification", controls_disp)

        # Only compute diff if we have enough history
        if delay < len(frame_buffer):

            print (-(delay ) )
            old_frame = frame_buffer[-(delay + 1)]      # take old frame from buffer
            '''
            deque  index
            [-1] is the last item added.
            [-2] is the second to last item added.
            [0]  is the first item (the oldest in a deque).
            '''

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

        if cv.waitKey(160) == 27:  # ESC==27    wait(15)==60_fps
            break

    cap.release()
    cv.destroyAllWindows()

    p_result = pstats.Stats (profile)
    p_result.sort_stats(pstats.SortKey.TIME)
    p_result.print_stats()

# Todo   implement FFT
# Todo   perform segmentation
# Todo   subsample the video frame acording to trackbar settings