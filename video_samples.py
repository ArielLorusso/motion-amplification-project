
#  The purpose of this script is generating simple video samples
#  This will be used to test the motion_amplification.py script 
#  in a clear and consistent maner


import cv2 as cv     # Import OpenCv library
import numpy as np   # generate image and positions    
import math

#  pip list | grep cv     <-- see my version  of OpenCV
'''
opencv-contrib-python    4.8.1.78
opencv-python            4.12.0.88
numpy                    1.26.4
'''

# time Parameters
fps = 60
seconds = 4
frames = seconds * fps

# Resolution  Parameters
video_W,  video_H = 640, 480
w_center    = video_W // 2   # use // to perform integer operation  / is float
h_center    = video_H // 2 

# Shape parameters
rectangle_width, rectangle_height = 6, 6
radius = 30
color1 = (  0, 255, 255) # YELLOW  (BGR)
color2 = (255, 255,   0) # CYAN
#color2 = (  0,   0, 255) # RED  (BGR)

#* create video object named out
fourcc = cv.VideoWriter_fourcc(*"mp4v")  #  other options : DIVX, XVID, MJPG, X264, WMV1, WMV2
out   = cv.VideoWriter("circle-sqare.mp4", fourcc, fps, (video_W, video_H))
# fourcc  = video codec  https://fourcc.org/codecs.php
# https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html
# out   = video generated 


def rectangle(img, pos, sca, rot, color):
    """
    Draw rotated rectangle with subpixel precision
    
    Arguments:
        img: Image array
        pos: (row, col) position as floats
        sca: (width, height) tuple
        rot: rotation angle in degrees
        color: BGR color tuple
    """
    width, height = sca
    row, col = pos
    
    hh = height / 2
    hw = width / 2
    
    # Create rectangle corners centered at origin
    pts = np.array([
        [-hw, -hh],  # top-left
        [+hw, -hh],  # top-right
        [+hw, +hh],  # bottom-right
        [-hw, +hh]   # bottom-left
    ], dtype=np.float32)
    
    # Apply rotation if needed
    if rot != 0:
        angle_rad = rot * math.pi / 180
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Rotation matrix
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # Rotate all points
        pts = pts @ rotation_matrix.T
    
    
    # 1. Increase sub-pixel resolution for smoother position but sharp pixels
    # A shift of 4 or 8 bits is standard for high-precision sub-pixel drawing.
    shift = 4
    scale = 1 << shift  # 2^4 = 16
    # Translate to final position (OpenCV uses x, y which is col, row)
    pts[:, 0] += col  # x-coordinate (column)
    pts[:, 1] += row  # y-coordinate (row)

    # 2. Ensure your points are scaled correctly
    pts_scaled = np.round(pts * scale).astype(np.int32)

    # 3. Use cv.LINE_8 to avoid the "soft" look of anti-aliasing (LINE_AA)
    cv.fillConvexPoly(img, pts_scaled, color, lineType=cv.LINE_AA, shift=shift)

    # ALTERNATIVES 
#   cv.drawContours(img, [pts_scaled], contourIdx=-1, color=color thickness=-1, lineType=cv.LINE_8)    
#   cv.rectangle(img,pt1=pt1, pt2=pt2, color=(255,0,0))
    # img = cv.warpAffine(img, rotation_matrix, (width, height))
    # https://opencv.org/blog/image-rotation-and-translation-using-opencv/#:~:text=1-,cv2.getRotationMatrix2D


def ellipse(img, pos, sca, rot, color):
    """
    Draw rotated ellipse with subpixel precision
    
    Arguments:
        img: Image array
        pos: (row, col) center position
        sca: (width, height) - full diameter, not radius
        rot: rotation angle in degrees
        color: BGR color tuple
    """
    row, col = pos
    width, height = sca
    
    # OpenCV ellipse uses subpixel coordinates with shift
    shift = 4
    scale = 1 << shift
    
    center = (int(col * scale), int(row * scale))
    axes   = (int(width / 2 * scale), int(height / 2 * scale))  # Semi-axes
    
    cv.ellipse(img, center, axes,
               angle=rot,
               startAngle=0, endAngle=360,
               color=color, thickness=-1,
               lineType=cv.LINE_AA, shift=shift)

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
    Generate sinusoidal oscillation 
    
    arguments:
        pos      : [ [x,y],[x,y]... ]  position array to modify

        x_amp    : amplitude of angular change
        x_freq   : number of complete cycles during the movement period
        f_start  : start frame
        f_end    : end frame
        frames   : total frames (for holding position after f_end)
        x_phase  : initial phase offset in radians (0 to 360)
    """
    if rot == None :
        rot = np.full(frames, phase, dtype=np.float32)

    for i in range(f_start, f_end):
        # calculate fime given the frame
        t = (i - f_start) / fps
        
        # Calculate sinusoidal position
        # 2π * frequency * t gives the angle for the desired number of cycles
        phase = 2 * math.pi * freq * t      # variable phase is repurpused
        angle_offset =  amp * math.sin(phase)

        rot[i] = phase + angle_offset  # Set angle (center + oscillation)
    
    # Hold the last rotation for remaining frames
    if f_end < frames:
        last = rot[f_end - 1]
        for i in range(f_end, frames):
            rot[i] = last

    
    return rot

y = video_H / 2
x = video_W / 2

#######################*  ANIMATION FUNCTIONS  *############################################################333

# ! Reimplement as :     shape_ani = (shape_pos, shape_sca, shape_rot)


def linear_animation ():
    rectang_pos = initial_position(frames,y,100.0)
 #   rectang_pos = linear_movement( rectang_pos, 1.0, y, 3.0,y,  0,60,frames)
 #   rectang_pos = linear_movement( rectang_pos, 3.0, y, 1.0,y,  60,frames,frames)

    circle_pos  = initial_position(frames,y,300.0)
 #   circle_pos  = linear_movement( circle_pos, 3.0,y, 1.0, y,   0,60,frames)
 #   circle_pos  = linear_movement( circle_pos, 1.0, y, 3.0,y,   60,frames,frames)

    
    return rectang_pos,circle_pos 

rectang_pos, circle_pos = linear_animation()

'''
def sine_position_animation ():
    rectang_mov = initial_position(frames, y, x)
    # rectang_mov = sinusoidal_movement(rectang_mov,x_amp=350.5,  x_freq=.80,  x_phase=0, 
    #                                               y_amp=200.5,  y_freq=.30 , y_phase=90,    
    #                                               f_start=0, f_end=frames, frames=frames)   #* Moire paretn test
    rectang_pos = sinusoidal_movement(rectang_mov,x_amp=2.5,  x_freq=16,  x_phase=0, 
                                                  y_amp=4.5,  y_freq=8 ,  y_phase=90,    f_start=0, f_end=frames, frames=frames)
    rectang_pos = sinusoidal_movement(rectang_mov,x_amp=100.5, x_freq=0.5, x_phase=0, 
                                                  y_amp=100.5,  y_freq=0.5, y_phase=110 , f_start=0, f_end=frames, frames=frames)
    #return rectang_mov,circle_mov
    return rectang_pos
'''

angle = sinusoidal_rotation(None, amp=15, freq=1, phase=45, 
                           f_start=0, f_end=240, frames=frames)  # from 0 to 90    (45-45)  to (45+45)

# rectang_pos = sine_position_animation ()     # same position and scale share dimention [x,y][i]
# rectang_sca = sine_position_animation () /10 # so thers no need for making new functions, 

#######################*  DRAWING LOOP  *############################################################333

for i in range(frames):     # i = frame number : from 0 to frames -1
    img = np.zeros((video_H,video_W, 3), dtype=np.uint8)  # Refresh image to (RGB) black  
    # By moving previous line otside the for loop we get frame acumulation (usefull to debug)

#    centered_rectangle(img, rectang_pos[i] , width=6, height=6, color=color2)
    rectangle(img, rectang_pos[i] , (180,90) ,angle[i],   color=color2)  
    ellipse  (img, circle_pos[i]  , (90,180) ,angle[i],   color=color1)
    out.write(img)

out.release()

# This script is good for some test but improvements can be done to it

#todo   Add :     scale /ratation motion
#todo   Add :     color hue rotation (time)
#todo   Add :     color gradients    (space)

#todo   Add : timing profilet
#todo   Reimplement position with cv2.warpAffine  and compare performance
        # translation_matrix = np.float32([
        #     [1, 0, tx],
        #     [0, 1, ty]
        # ])
        # translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
