#  Tue 20 Jan 2026 23:01:38 

#  The purpose of this script is generating simple video samples
#  This will be used to test the motion_amplification.py script 
#  in a clear and consistent maner


import cv2 as cv   # Import OpenCv library
import numpy as np
#  pip list | grep cv     <-- see my version  of OpenCV
'''
opencv-contrib-python    4.8.1.78
opencv-python            4.12.0.88
'''

# time Parameters
fps = 60
seconds = 2
frames = seconds * fps

# Resolution  Parameters
video_W,  video_H = 640, 480
w_center    = video_W // 2   # use // to perform integer operation  / is float
h_center    = video_H // 2 

# Shape parameters
rectangle_width, rectangle_height = 60, 60
radius = 30
color1 = (  0, 255, 255) # YELLOW
color2 = (255,   0,   0) # RED

fourcc = cv.VideoWriter_fourcc(*"mp4v") 
# fourcc  = video codec  https://fourcc.org/codecs.php
out   = cv.VideoWriter("circle-sqare.mp4", fourcc, fps, (video_W, video_H))
# out   = video generated 

radius = 30
y = video_H // 2

for i in range(frames):     # i = frame number : from 0 to frames -1
    img = np.zeros((video_H,video_W, 3), dtype=np.uint8)  # Refresh image to black

    # x position moves left -> right
    t = i / (frames - 1)    # update time
                            # t = 0  : first frame
                            # t = 1  : last  frame
    
#    print (f"i = {i } \t t = {t}")                   # debug purposes, everithing OK
    x = int(radius + t * (video_W - 2 * radius))      # update position

    cv.circle(img, (x, y), radius, color1, -1)        # render (harcoded for quick test)
    cv.rectangle(img, (w_center,h_center),
                      (video_W-x,y//2),color=color2, thickness=-1 )

    out.write(img)

out.release()


# This code is a proof of concept
# it just makes a circle and square to move in oposite horizontal directions

#todo   refactor : make a function to draw square given its center position
#todo   combine  : make a way to add horizontal + vertical movement before render
#todo   implement sinusoidal motion
#todo   implement motion_reduction to subpixel level if posible

