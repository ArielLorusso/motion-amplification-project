import cv2 as cv
import numpy as np
from collections import deque
from scipy import fft  # better than numpy.fft


# todo    Fix slow even when unprocessing  (mode 0)


#*█████████   Create Trackbars   ████████████████████████████████████████

# Trackbar limit Parameters
MAX_DELAY = 30            # Increased for FFT analysis
MAX_AMP   = 10
MAX_TEMPORAL_WINDOW = 64  # For per-pixel temporal FFT

# Separated windows for trackbar controls
controls_disp = "Controls"
cv.namedWindow(controls_disp, cv.WINDOW_NORMAL)
window_disp   = "Motion Analysis"
cv.namedWindow(window_disp,   cv.WINDOW_NORMAL)
def nothing(x):
    pass
# Trackbars        Name                window,     MIN, MAX,  function  
cv.createTrackbar("Mode",              controls_disp, 0, 4, nothing)  
cv.createTrackbar("Frame delay",       controls_disp, 1, MAX_DELAY, nothing)
cv.createTrackbar("Amplification",     controls_disp, 1, MAX_AMP, nothing)
# 0: Diff, 1: Temporal FFT, 2: Spatiotemporal FFT, 3: Phase-based motion
cv.createTrackbar("Freq_min",          controls_disp, 5, 30, nothing)  # For FFT filtering
cv.createTrackbar("Freq Band",         controls_disp, 5, 30, nothing)  # For FFT filtering
cv.createTrackbar("Record",            controls_disp, 0,  1, nothing)   # 0: Off, 1: Recording

#*█████████   Video Processing Class   ████████████████████████████████████████

class VideoFFTAnalyzer:
    """Handles FFT-based video analysis"""
    
    def __init__(self, max_temporal_window=64):
        self.max_temporal_window = max_temporal_window
        self.temporal_buffer = None  # Will store (height, width, time) array
        self.frame_count = 0
        
    def add_frame(self, frame_gray):
        """Add frame to temporal buffer"""
        if self.temporal_buffer is None:
            h, w = frame_gray.shape
            self.temporal_buffer = np.zeros((h, w, self.max_temporal_window), dtype=np.float32)
        
        # Roll buffer and add new frame
        self.temporal_buffer = np.roll(self.temporal_buffer, -1, axis=2)
        self.temporal_buffer[:, :, -1] = frame_gray.astype(np.float32)
        self.frame_count = min(self.frame_count + 1, self.max_temporal_window)
    
    def per_pixel_temporal_fft(self, amplification=10, freq_min=0.5, freq_max=10, fps=30):
        """
        A) Per-pixel temporal FFT
        Computes 1D FFT along time axis for each pixel independently
        
        Args:
            freq_min, freq_max: Frequency band to amplify (Hz)
            fps: Video frame rate
        
        Returns:
            Filtered motion image
        """
        if self.frame_count < 8:
            return np.zeros(self.temporal_buffer.shape[:2], dtype=np.uint8)
        
        # Get actual data (not zero-padded part)
        valid_data = self.temporal_buffer[:, :, -self.frame_count:]
        
        # 1D FFT along time axis (axis=2)
        fft_result = fft.fft(valid_data, axis=2)
        
        # Create frequency axis for making mask
        freqs = fft.fftfreq(self.frame_count, d=1.0/fps)
        
        # Create bandpass filter
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        # Apply filter (zero out frequencies outside band)
        fft_filtered = fft_result.copy()
        fft_filtered *= amplification       # 
        fft_filtered[:, :, ~freq_mask] = 0  # frequencyes outside range = 0
        
        # Inverse FFT to get filtered temporal signal
        filtered_signal = fft.ifft(fft_filtered, axis=2).real
        
        # Compute motion energy (variance over time)
        motion_energy   = np.std(filtered_signal, axis=2)
        
        # Normalize to 0-255
        motion_energy   = np.clip(motion_energy * 10, 0, 255).astype(np.uint8)
        
        return motion_energy
    
    def spatiotemporal_fft_3d(self, subsample=4):
        """
        B) Full 3D spatiotemporal FFT
        Treats video as 3D volume and computes 3D FFT
        
        Args:
            subsample: Spatial downsampling factor (for speed)
        
        Returns:
            3D FFT magnitude visualization
        """
        if self.frame_count < 8:
            return np.zeros(self.temporal_buffer.shape[:2], dtype=np.uint8)
        
        # Get valid data and subsample spatially
        valid_data = self.temporal_buffer[:, :, -self.frame_count:]
        h, w, t = valid_data.shape
        
        # Subsample for computational efficiency
        data_small = valid_data[::subsample, ::subsample, :]
                    #  Slicing syntax: start:stop:step
                    #  we subsample every 4 row,col

        # 3D FFT (x, y, t)
        fft_3d = fft.fftn(data_small)
        fft_3d_shifted = fft.fftshift(fft_3d)
        
        # Magnitude spectrum
        magnitude = np.abs(fft_3d_shifted)
        
        # Visualize by summing along temporal axis (show spatial frequencies)
        spatial_freq_map = np.sum(magnitude, axis=2)
        
        # Log scale for better visualization
        spatial_freq_map = np.log1p(spatial_freq_map)
        
        # Normalize and resize back to original size
        spatial_freq_map = (spatial_freq_map / spatial_freq_map.max() * 255).astype(np.uint8)
        spatial_freq_map = cv.resize(spatial_freq_map, (w, h))
        
        return spatial_freq_map
    
    def phase_based_motion_amplification(self, amplification=10, freq_min=0.5, freq_max=10):
        """
        Phase-based motion amplification (inspired by MIT's motion magnification)
        Uses phase information to amplify subtle motion
        
        Args:
            amplification: Motion amplification factor
            freq_band: (low, high) frequency band in Hz
        """
        if self.frame_count < 16:
            return np.zeros(self.temporal_buffer.shape[:2], dtype=np.uint8)
        
        valid_data = self.temporal_buffer[:, :, -self.frame_count:]
        
        # 1D FFT per pixel
        fft_result = fft.fft(valid_data, axis=2)
        
        # Extract phase
        phase = np.angle(fft_result)
        magnitude = np.abs(fft_result)
        
        # Amplify phase in frequency band
        freqs = fft.fftfreq(self.frame_count, d=1.0/30)
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        phase_amplified = phase.copy()
        phase_amplified[:, :, freq_mask] *= amplification
        
        # Reconstruct signal with amplified phase
        fft_amplified = magnitude * np.exp(1j * phase_amplified)
        signal_amplified = fft.ifft(fft_amplified, axis=2).real
        
        # Show current amplified frame
        current_frame = signal_amplified[:, :, -1]
        current_frame = np.clip(current_frame, 0, 255).astype(np.uint8)
        
        return current_frame

    def phase_3d_amplification(self, subsample=4, amplification=10, freq_min=0.5, freq_max=10, fps=30):
        """
        Simpler version without fftshift
        """
        if self.frame_count < 8:
            return np.zeros(self.temporal_buffer.shape[:2], dtype=np.uint8)
        
        valid_data = self.temporal_buffer[:, :, -self.frame_count:]
        h, w, t = valid_data.shape
        
        data_small = valid_data[::subsample, ::subsample, :]
        
        # 3D FFT
        fft_3d = fft.fftn(data_small)
        # No fftshift needed!
        
        # Extract phase
        phase = np.angle(fft_3d)
        magnitude = np.abs(fft_3d)
        
        # Temporal frequency mask
        freqs = fft.fftfreq(data_small.shape[2], d=1.0/fps)
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        # Amplify phase
        phase_amplified = phase.copy()
        phase_amplified[:, :, freq_mask] *= amplification
        
        # Reconstruct
        fft_amplified = magnitude * np.exp(1j * phase_amplified)
        
        # Inverse FFT (no axis, no ifftshift needed)
        signal_amplified = fft.ifftn(fft_amplified).real
        
        # Get current frame and resize
        current_frame = signal_amplified[:, :, -1]
        current_frame = cv.resize(current_frame, (w, h))
        current_frame = np.clip(current_frame, 0, 255).astype(np.uint8)
        
        return current_frame

#*█████████   Initialize Video Capture   ████████████████████████████████████████
# Initialize video
video_path = "./circle-sqare.mp4"
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Error opening video")
# Get video properties
fps = cap.get(cv.CAP_PROP_FPS) or 30
video_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(f"Video properties: {video_width}x{video_height} @ {fps} FPS")
success, frame = cap.read()
if not success:
    raise RuntimeError("Could not read first frame")
# Initialize video writer variables
video_writer    = None
is_recording    = False
output_filename = None

#*█████████   Recording video functions   ████████████████████████████████████████
def get_output_filename(mode):
    """Generate unique output filename based on mode and timestamp"""
    import datetime
    mode_names = ["diff", "temporal_fft", "spatiotemporal_fft", "phase_amp","3D_phace"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output_{mode_names[mode]}_{timestamp}.mp4"

def initialize_video_writer(filename, fps, width, height):
    """Initialize video writer with proper codec and parameters"""
    # Try different codecs in order of preference
    codecs = [
        ('mp4v', '.mp4'),  # MPEG-4
        ('avc1', '.mp4'),  # H.264
        ('XVID', '.avi'),  # Xvid
        ('MJPG', '.avi'),  # Motion JPEG
    ]
    
    for codec, ext in codecs:
        fourcc = cv.VideoWriter_fourcc(*codec)
        if not filename.endswith(ext):
            filename = filename.rsplit('.', 1)[0] + ext
        
        writer = cv.VideoWriter(filename, fourcc, fps, (width, height))
        
        if writer.isOpened():
            print(f"✓ Video writer initialized: {filename} ({codec} codec)")
            return writer, filename
        else:
            writer.release()
    
    raise RuntimeError("Failed to initialize video writer with any codec")

#*█████████   Inic Video Buffer and Process class  ████████████████████████████████████████
# Initialize buffers
frame_buffer = deque(maxlen=MAX_DELAY + 1)
frame_buffer.append(frame)

# Initialize FFT analyzer
fft_analyzer = VideoFFTAnalyzer(max_temporal_window=MAX_TEMPORAL_WINDOW)
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
fft_analyzer.add_frame(frame_gray)

#*█████████   Display Modes   ████████████████████████████████████████

print("\n=== Controls ===")
print("Mode 0: Show unprocess video")
print("Mode 1: Frame Difference ")
print("Mode 2: Per-Pixel Temporal FFT")
print("Mode 3: 3D Spatiotemporal FFT")
print("Mode 4: Phase-Based Motion Amplification")
print("\nRecord Trackbar:")
print("  0 = Off (not recording)")
print("  1 = On (recording to file)")
print("\nESC: Exit")
print("================\n")

#============================================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_buffer.append(frame)
    
    # Convert to grayscale for FFT analysis
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fft_analyzer.add_frame(frame_gray)
    
    #*█████████   Get trackbar values  ████████████████████████████████████████
    mode  = cv.getTrackbarPos("Mode", controls_disp)
    delay = cv.getTrackbarPos("Frame delay", controls_disp)
    amp   = cv.getTrackbarPos("Amplification", controls_disp)
    freq_min  = cv.getTrackbarPos("Freq_min", controls_disp)
    freq_band = cv.getTrackbarPos("Freq Band", controls_disp)
    freq_max  = freq_min + freq_band
    record = cv.getTrackbarPos("Record", controls_disp)
    # Handle recording state changes

    #*█████████   RECORD              ████████████████████████████████████████
    if record == 1 and not is_recording:
        # Start recording
        output_filename = get_output_filename(mode)
        try:
            video_writer, output_filename = initialize_video_writer(
                output_filename, fps, video_width, video_height
            )
            is_recording = True
            print(f"🔴 Recording started: {output_filename}")
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            cv.setTrackbarPos("Record", controls_disp, 0)
            
    elif record == 0 and is_recording:
        # Stop recording
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            print(f"⏹️  Recording stopped: {output_filename}")
            print(f"✓ Video saved successfully")
        is_recording = False

    # Write frame to video file if recording
    if is_recording and video_writer is not None:
        # Ensure frame is correct size and 3-channel BGR
        if display.shape[:2] != (video_height, video_width):
            display_resized = cv.resize(display, (video_width, video_height))
        else:
            display_resized = display
            
        if len(display_resized.shape) == 2:  # Grayscale
            display_resized = cv.cvtColor(display_resized, cv.COLOR_GRAY2BGR)
        
        video_writer.write(display_resized)
    
    
    #*█████████   VODEO   PROCESS     ████████████████████████████████████████
   
    # Mode selection
    if   mode == 0:                   # dont process video
        display = frame     

    if   mode == 1:
        # Original frame difference
        if delay < len(frame_buffer):
            old_frame = frame_buffer[-(delay + 1)]
            diff = amp * np.abs(
                frame.astype(np.int16) - old_frame.astype(np.int16)
            )
            result = np.clip(diff, 0, 255).astype(np.uint8)
        else:
            result = np.zeros_like(frame)
        
        display = result
    
    elif mode == 2:
        # A) Per-pixel temporal FFT
        motion_map = fft_analyzer.per_pixel_temporal_fft(
            freq_min=freq_min, 
            freq_max=freq_max, 
            fps=fps
        )
        display = cv.applyColorMap(motion_map, cv.COLORMAP_JET)
    
    elif mode == 3:
        # B) 3D Spatiotemporal FFT
        freq_map = fft_analyzer.spatiotemporal_fft_3d(subsample=4)
        display = cv.applyColorMap(freq_map, cv.COLORMAP_HOT)
    
    elif mode == 4:
        # Phase-based motion amplification
        freq_high = freq_max
        amplified = fft_analyzer.phase_based_motion_amplification(
            amplification=amp,
            freq_min=freq_min,
            freq_max=freq_max
        )
        display = cv.cvtColor(amplified, cv.COLOR_GRAY2BGR)

    elif mode == 5:
        # Phase-based motion amplification
        freq_high = freq_max
        amplified = fft_analyzer.phase_3d_amplification(
            subsample=4,
            amplification=amp,
            freq_min=freq_min,
            freq_max=freq_max
        )
        display = cv.cvtColor(amplified, cv.COLOR_GRAY2BGR)
    
    else:
        display = frame
    
    #*█████████     Display VODEO       ████████████████████████████████████████

    cv.imshow(window_disp, display)
    if cv.waitKey(13) == 27:  # ESC
        break                 # exit video Loop

# Cleanup
if video_writer is not None:
    video_writer.release()
    print(f"✓ Video saved on exit: {output_filename}")

cap.release()
cv.destroyAllWindows()

print("\n=== Cleanup Complete ===")