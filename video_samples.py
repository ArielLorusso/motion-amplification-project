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
import math

#  pip list | grep cv     <-- see my version  of OpenCV
'''
opencv-contrib-python    4.8.1.78
opencv-python            4.12.0.88
numpy                    1.26.4
'''

# time Parameters
fps = 60
seconds = 10
frames = seconds * fps

# Resolution  Parameters
video_W,  video_H = 640, 480
w_center    = video_W // 2   # use // to perform integer operation  / is float
h_center    = video_H // 2 

# Shape parameters
rectangle_width, rectangle_height = 6, 6
radius = 30
color1 = (  0, 255, 255) # YELLOW
color2 = (255,   0,   0) # RED

#* create video object named out
fourcc = cv.VideoWriter_fourcc(*"mp4v")  #  other options : DIVX, XVID, MJPG, X264, WMV1, WMV2
# fourcc  = video codec  https://fourcc.org/codecs.php
# https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html
out   = cv.VideoWriter("circle-sqare.mp4", fourcc, fps, (video_W, video_H))
# out   = video generated 

"""
def rotation_movement( f_start, f_end, frames):


    # Define the rotation angle
    angle = 45  # Rotate by 45 degrees
    # Define the scaling factor
    scale = 1.0  # No scaling
    # Get the rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale) 
    # Perform the rotation
    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
    # https://opencv.org/blog/image-rotation-and-translation-using-opencv/#:~:text=1-,cv2.getRotationMatrix2D

"""
def rectangle(img, pos, sca , color):
    width, height = sca
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

def ellipse(img, pos, sca, color):
    """
    pos: (y, x) position  (float)
    sca: (a, b) float
    """
    y, x = pos
    a, b = sca # a is vertical radius, b is horizontal radius
    
    # 1. Define bounding box (add padding for the AA ramp)
    pad = max(a, b) + 2
    top, bottom = int(y - pad), int(y + pad)
    left, right = int(x - pad), int(x + pad)
    
    # Clip to image boundaries
    top, bottom = max(0, top ), min(img.shape[0], bottom)
    left, right = max(0, left), min(img.shape[1], right)
    
    # 2. Create grid
    yy, xx = np.ogrid[top:bottom, left:right]
    
    # 3. Calculate Normalized Distance (d^2)
    # Inside the ellipse, value < 1. Outside, value > 1.
    dist_sq = ((yy - y)**2 / a**2) + ((xx - x)**2 / b**2)
    dist = np.sqrt(dist_sq)
    
    # 4. Anti-aliasing Mask
    # We create a smooth transition at the boundary (dist = 1.0)
    # The 'width' of the smoothing is roughly 1/min(a,b) pixels
    smoothing = 1.0 / min(a, b)
    mask = np.clip((1.0 - dist) / smoothing + 0.5, 0, 1)
    
    # 5. Blend
    mask = mask[:, :, np.newaxis]
    img_crop = img[top:bottom, left:right]
    img[top:bottom, left:right] = (1 - mask) * img_crop + mask * color


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


def sinusoidal_movement(pos, x_amp, y_amp, x_freq, y_freq, f_start, f_end, frames, x_phase=0, y_phase=0):
    """
    Generates sinusoidal x and y position oscillation.
    position is a numpy array  [ [x,y],[x,y]... ] 
    ( y axis arguments have same behaviour as x ones )
    this function adds to the original position for easy composed motion

    arguments:
        pos      : [ [x,y],[x,y]... ]  position array to modify
        x_amp    : amplitude of x axis oscillation in pixels
        x_freq   : number of complete cycles during the movement period
        f_start  : start frame
        f_end    : end frame
        frames   : total frames (for holding position after f_end)
        x_phase  : initial phase offset in radians (0 to 360)
   """
    # phace is converted to degree  (if deleted radiasn are used instead)
    x_phase = x_phase / 180 * math.pi  
    y_phase = y_phase / 180 * math.pi  
    
    for i in range(f_start, f_end):
        # calculate fime in seconds given a frame i
        t = (i - f_start) / fps
        
        # Calculate sinusoidal position
        # 2π * frequency * t gives the angle,  used for oscilation
        x_angle = 2 * math.pi * x_freq * t + x_phase
        x =  x_amp * math.sin(x_angle)
        # same for y axis
        y_angle = 2 * math.pi * y_freq * t + y_phase
        y = y_amp * math.sin(y_angle)
        # update position vector
        pos[i, 0] += y     # row
        pos[i, 1] += x     # column
    
    # Hold the last position for remaining frames
    if f_end < frames:
        last_x, last_y = pos[f_end - 1]
        for i in range(f_end, frames):
            pos[i, 0] = last_y
            pos[i, 1] = last_x
    
    return pos

def sinusoidal_rotation(rot, amp, freq, phase, f_start, f_end, frames):

    """
    Generate sinusoidal oscillation in x-axis
    
    arguments:
        pos      : [ [x,y],[x,y]... ]  position array to modify

        x_amp    : amplitude of x axis oscillation in pixels
        x_freq   : number of complete cycles during the movement period
        f_start  : start frame
        f_end    : end frame
        frames   : total frames (for holding position after f_end)
        x_phase  : initial phase offset in radians (0 to 360)
    """
    if pos == None :
        pos = np.tile([phase], (frames,1))

    for i in range(f_start, f_end):
        # calculate fime given the frame
        t = (i - f_start) / fps
        
        # Calculate sinusoidal position
        # 2π * frequency * t gives the angle for the desired number of cycles
        phase = 2 * math.pi * freq * t 
        angle =  amp * math.sin(angle)

        pos[i, 1] += x     # column
    
    # Hold the last position for remaining frames
    if f_end < frames:
        last_x, last_y = pos[f_end - 1]
        for i in range(f_end, frames):
            pos[i, 0] = last_y
            pos[i, 1] = last_x
    
    return pos

y = video_H / 2
x = video_W / 2

def linear_animation ():
    rectang_mov = initial_position(frames,y,100.0)
    rectang_mov = linear_movement( rectang_mov, 1.0, y, 3.0,y,  0,60,frames)
    rectang_mov = linear_movement( rectang_mov, 3.0, y, 1.0,y,  60,frames,frames)

    circle_mov  =  initial_position(frames,y,100.0)
    circle_mov  = linear_movement( circle_mov, 3.0,y, 1.0, y,   0,60,frames)
    circle_mov  = linear_movement( circle_mov, 1.0, y, 3.0,y,   60,frames,frames)
    return rectang_mov,circle_mov 

rectang_mov,circle_mov = linear_animation()


def sine_position_animation ():
    rectang_mov = initial_position(frames, y, x)
    # rectang_mov = sinusoidal_movement(rectang_mov,x_amp=350.5,  x_freq=.80,  x_phase=0, 
    #                                               y_amp=200.5,  y_freq=.30 , y_phase=90,    
    #                                               f_start=0, f_end=frames, frames=frames)   #* Moire paretn test
    rectang_mov = sinusoidal_movement(rectang_mov,x_amp=2.5,  x_freq=16,  x_phase=0, 
                                                  y_amp=4.5,  y_freq=8 ,  y_phase=90,    f_start=0, f_end=frames, frames=frames)
    rectang_mov = sinusoidal_movement(rectang_mov,x_amp=100.5, x_freq=0.5, x_phase=0, 
                                                  y_amp=100.5,  y_freq=0.5, y_phase=110 , f_start=0, f_end=frames, frames=frames)
    #return rectang_mov,circle_mov
    return rectang_mov

#######################*  DRAWING LOOP  *############################################################333

rectang_pos = sine_position_animation ()     # same position and scale share dimention [x,y][i]
rectang_sca = sine_position_animation () /10 # so thers no need for making new functions, 

for i in range(frames):     # i = frame number : from 0 to frames -1
    img = np.zeros((video_H,video_W, 3), dtype=np.uint8)  # Refresh image to (RGB) black  
    # By moving previous line otside the for loop we get frame acumulation (usefull to debug)

#    centered_rectangle(img, rectang_pos[i] , width=6, height=6, color=color2)
    rectangle(img, rectang_pos[i] , rectang_sca[i], color=color2)  
    ellipse  (img, circle_mov[i], (radius*2,  radius/2), color=color1)
    out.write(img)

out.release()

# This script is good for some test but improvements can be done to it

#todo   implement scale /ratation motion
#todo   implement color hue rotation (time)
#todo   implement color gradients    (space)


