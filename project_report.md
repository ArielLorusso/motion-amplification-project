# Project Report
	
# The inspiration : 

    I've seen many examples of Video motion amplification and the capabilities it has.
	“Amplifying Invisible Motion”  by Steve Mould  https://www.youtube.com/watch?v=kuMUquaeE6g
    He provided some example with 3D printer vibration, the purring of a cat,
    the htoat movement while humming and the resonation motion of building caused by wind.
    all very inspiring examples


# The Project
    to make a software capable of amplification motion bi a selected range of frequencies by the user
    This has lots of applications :
    Industry : vibration diagnosis, allow making changer for  equipment to last longer, be safer, and quieter.
    seismography : detecting movements in buildings and clouds could be used to set alarms and save people 
    Medicine : detecting someones breathing / hart rate 
    can be a life saver for people that need constant monitoring but cant stay at the hospital.
    it can also be used to detect apnea or other problems while sleeping.
    I plan on making a Python script, mainly in OpenCV and using various techniques

# The stack
## Why Python ?
	My idea is to use Python. I'v use both C++ and python for image projects in the past.
    I value how easy python is to write and how fast libraries can be tried by just a pip install 
    C++ has better performance by as a language but by using C compiled libraries like numpy it gets almost as fast.
    Other libraries i can try are torch, Cython, numba and taichi since i already have them installed.
    Once I have the script working I could try making a C++ version of it. other approach is to ma 
    I never made my own C python library, is not out of question, but ill try it after everything is working.

## Why OpenCV ?
    OpenCV has most tools I need, and its Graph API (G-API) has good performance.  
    I can use OpenCV to read the frames of the video and store multiple of them in a buffer to perform time operations.
    It also has user control trough track bars that came really handy
    I may use other libraries as PIL (Useful for opening some codecs) or SKimage (has lots of nice filters)
    Numpy (useful to do my own kernel and images) but I'll try to rely on OpenCV asn the main library.
    
# The Techniques
## Difference :
    The first approach could be to make the difference between one frame and the previous.
    This has some Limitations.

### Limitations
    It does differentiate where a change occurred in each pixel by itself but not including the context.     
    for example if we provide a video of a hand we will see just the perimeter being detected as motion.
    since the pixels inside far from the edge dont have much change (skin color is mostly the same everywhere) 
    its movement will be near 0 or black. Edges will be perceived as having abrupt changes near 1 or white


## Analisis
    I could generate my own input videos instead of using random downloaded videos.
    this will make easiest to diagnose what our script is really making and where is laking.
    it also allows the control resolution and framerate wich will come handy.
    It also avoids undesired effects of real footage like noise and rolling shutter than can make the analisis confusing
    examples I want to make are :
        Solid Shape moving : horizontal, vertical, circular motion, Grow/Shrink, bending.
    I could use different methods for the time response. Linear and sinusoidal as a starting point.

## FFT:
    I plan on using FFT (Fast Fourier Transform) to analyze and amplify the frequencies in the video.
    This may require IFFT (Inverse Fourier transform) to get back to "time" domain after amplification.
    
    The last time I used FFT was in an image
    I converted the 2D space of a grayscale image to its frequency domain. 
    This has Amplitude and phase for each pixel and the Amplitude is displayed as a result image.
    here there's no "time" the original time domain are the pixel positions of the image
    so from now on I will call it "input space" to avoid confusion.

    Video is different, we have time in addition to the 2D space as the domain.
    I can start by doing a isolated pixel by pixel analisis like in the difference approach 

## Interpolation:
    To amplify motion I must amplify the difference in position.
    Some kinds of motions (Grow/Shrink in particular) might make me create pixels with empty space in between
    some sort of spatial interpolation will be necesary for the image to be posible at all.    
    
## Profiling & Optimizing
    I plan on profiling the execution of each iteration to keep track of computational const
    https://docs.python.org/3/library/profile.html
    If needed I'll use linux tools like top, htop, grep, nvidia-smi ... etc  to se CPU/GPU use
    /proc/meminfo /proc/PID/status  /proc/cpuinfo  could be also good to analyze
    

# My Motivation : 

    1) To feel fulfilled by using my knowledge to provide solutions for me and others
   
    2) To be versatile, knowing which methods and algorithms fits better to each problem.
   
    3) To have fun by trying different ideas and comparing their pros and cons.
		

# Week by week schedule (subject to changes)
    1st week :  Video generation for test
    2st week :  Samples + Difference
    3nd week :  FFT  temporal space mix 
    4rd week :  Tracking & interpolation
    5rd week :  Research & review


◣ Tue 20 Jan 2026 19:44:59  Time by the end of this document