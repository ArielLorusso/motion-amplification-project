#  Tue 20 Jan 2026 23:01:38  Start
#  Tue 21 Jan 2026 12:01:38  1st commit
#  
#  The purpose of this script is generating simple video samples
#  This will be used to test the motion_amplification.py script 
#  in a clear and consistent maner
#
#
#
#


import cv2 as cv     # Import OpenCv library
import numpy as np   # generate image and positions    
import skimage.draw  # draw with float subpixel (openCV only uses int values)    
#  pip list | grep cv     <-- see my version  of OpenCV
'''
opencv-contrib-python    4.8.1.78
opencv-python            4.12.0.88
numpy                    1.26.4
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

#* create video object named out
fourcc = cv.VideoWriter_fourcc(*"mp4v")  #  other options : DIVX, XVID, MJPG, X264, WMV1, WMV2
# fourcc  = video codec  https://fourcc.org/codecs.php
# https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html
out   = cv.VideoWriter("circle-sqare.mp4", fourcc, fps, (video_W, video_H))
# out   = video generated 

# def centered_rectangle (img, center_x,center_y,width,height, color) :
#    p1 = (center_x - width//2, center_y - height//2)
#    p2 = (center_x + width//2, center_y + height//2)
#    img[skimage.draw.rectangle(circle_mov[i], radius=radius), shape=img.shape] = color1
#    cv.rectangle(img, p1, p2, color, thickness=-1 )        #//  OLD   no subpixel
#    return img


def centered_rectangle(img, pos, width, height, color):
    row, col = pos  # floats
    
    hh = height / 2
    hw = width / 2
    
    # OpenCV uses (x, y) format, not (row, col)
    # and needs integer coordinates with fractional shift
    shift = 4  # 2^4 = 16 subpixel precision
    scale = 2 ** shift
    
    pts = np.array([
        [col - hw, row - hh],
        [col + hw, row - hh],
        [col + hw, row + hh],
        [col - hw, row + hh]
    ]) * scale
    
    pts = pts.astype(np.int32)
    
    cv.fillPoly(img, [pts], color, lineType=cv.LINE_AA, shift=shift)  


def initial_position(frames,x,y):
    pos = np.tile([x,y], (frames,1))

    return pos

def linear_movement (pos,x,y,x2,y2,f_start,f_end,frames):
    dx = ( x2 - x ) / (f_end-f_start-1)    # X increment each frame
    dy = ( y2 - y ) / (f_end-f_start-1)    # frames-1 is necesary to display last position 
    for i in range(f_start,f_end):  
        x += dx
        y += dy
        pos [i,0] = y  # Note: [0] is row, [1] is col
        pos [i,1] = x 
    for i in range(f_end,frames):  
        pos [i,0] = y  
        pos [i,1] = x 

    return pos


y = video_H / 2
rectang_mov = initial_position(frames,y,100.0)
rectang_mov = linear_movement( rectang_mov, 1.0, y, 12.0,y,  0,60,frames)
rectang_mov = linear_movement( rectang_mov, 12.0, y, 1.0,y,  60,frames,frames)

circle_mov  = initial_position(frames, y,100.0)
circle_mov  = linear_movement( circle_mov, 12.0,y, 1.0, y,   0,60,frames)
circle_mov  = linear_movement( circle_mov, 1.0, y, 12.0,y,   60,frames,frames)

radius = 30

for i in range(frames):     # i = frame number : from 0 to frames -1
    img = np.zeros((video_H,video_W, 3), dtype=np.uint8)  # Refresh image to (RGB) black  

    img[skimage.draw.disk(circle_mov[i], radius=radius)] = color1  # draw disk = mask 
    centered_rectangle(img, rectang_mov[i] , width=30, height=60, color=color2)

    out.write(img)

out.release()



# This code is a proof of concept
# it just makes a circle and square to move in oposite horizontal directions

#todo   refactor : make a function to draw square given its center position
#todo   combine  : make a way to add horizontal + vertical movement before render
#todo   implement sinusoidal motion
#todo   implement motion_reduction to subpixel level if posible

