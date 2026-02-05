import cv2 as cv
import numpy as np
from collections import deque
from scipy import fft  # better than numpy.fft
import time


#*█████████   Create Trackbars   ████████████████████████████████████████

# Trackbar limit Parameters
MAX_DELAY = 30            # Increased for FFT analysis
MAX_AMP   = 100
MAX_TEMPORAL_WINDOW = 64  # For per-pixel temporal FFT

# Separated windows for trackbar controls
controls_disp = "Controls"
cv.namedWindow(controls_disp, cv.WINDOW_NORMAL)
window_disp   = "Motion Analysis"
cv.namedWindow(window_disp,   cv.WINDOW_NORMAL)
def nothing(x):
    pass
# Trackbars        Name                window,     MIN, MAX,  function  
cv.createTrackbar("Mode",              controls_disp, 0, 7, nothing)  
cv.createTrackbar("Frame delay",       controls_disp, 1, MAX_DELAY, nothing)
cv.createTrackbar("Amplification",     controls_disp, 1, MAX_AMP, nothing)
cv.createTrackbar("Freq_min",          controls_disp, 1, 60, nothing)
cv.createTrackbar("Freq Band",         controls_disp, 1, 60, nothing)
cv.createTrackbar("Record",            controls_disp, 0,  1, nothing)

#*█████████   Video Processing Class   ████████████████████████████████████████

class VideoFFTAnalyzer:
    """Handles FFT-based video analysis"""
    
    def __init__(self, max_temporal_window=64):
        self.max_temporal_window = max_temporal_window
        self.temporal_buffer = None  # Will store (height, width, time) array
        self.buffer_idx  = 0  # Current write position (circular)
        self.frame_count = 0
        
    def add_frame(self, frame_gray):
        """Add frame to temporal buffer"""
        if self.temporal_buffer is None:
            h, w = frame_gray.shape
            self.temporal_buffer = np.zeros(
                (h, w, self.max_temporal_window), 
                dtype=np.float32
                )
        
        # write to circular index
        self.temporal_buffer[:, :, self.buffer_idx] = frame_gray.astype(np.float32)
        # Update circular index
        self.buffer_idx  = (self.buffer_idx + 1) % self.max_temporal_window
        self.frame_count = min(self.frame_count + 1, self.max_temporal_window)

    def get_valid_data_in_order(self):
        """Get frames in correct temporal order"""
        if self.frame_count < self.max_temporal_window:
            return self.temporal_buffer[:, :, :self.frame_count]
        else:
            return np.concatenate([
                self.temporal_buffer[:, :,  self.buffer_idx:],
                self.temporal_buffer[:, :, :self.buffer_idx]
            ], axis=2)

    def fft_1d(self, amplification=10, freq_min=0.5, freq_max=10, fps=30, subsample=4):
        """
        1D FFT - Amplify full FFT result
        Returns: Single 2D frame (grayscale)
        """
        if self.frame_count < 8:
            h, w = self.temporal_buffer.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        valid_data = self.get_valid_data_in_order()
        h, w, t = valid_data.shape
        data_small = valid_data[::subsample, ::subsample, :]
        
        # 1D FFT along time
        fft_result = fft.fft(data_small, axis=2)
        
        # Frequency mask
        freqs = fft.fftfreq(self.frame_count, d=1.0/fps)
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        # Amplify and filter
        fft_result *= amplification
        fft_result[:, :, ~freq_mask] = 0
        
        # Inverse FFT
        filtered_signal = fft.ifft(fft_result, axis=2).real
        
        # ✓ FIX: Get current frame (last in temporal sequence)
        current_frame = filtered_signal[:, :, -1]
        current_frame = cv.resize(current_frame, (w, h))
        
        return np.clip(current_frame, 0, 255).astype(np.uint8)

    def fft_1d_mag(self, amplification=10, freq_min=0.5, freq_max=10, fps=30, subsample=4):
        """
        1D FFT - Amplify magnitude only
        Returns: Single 2D frame (grayscale)
        """
        if self.frame_count < 8:
            h, w = self.temporal_buffer.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        valid_data = self.get_valid_data_in_order()
        h, w, t = valid_data.shape
        data_small = valid_data[::subsample, ::subsample, :]
        
        fft_result = fft.fft(data_small, axis=2)
        
        # Separate magnitude and phase
        phase = np.angle(fft_result)
        magnitude = np.abs(fft_result)
        
        # Frequency mask
        freqs = fft.fftfreq(self.frame_count, d=1.0/fps)
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        # Amplify magnitude, zero outside band
        magnitude *= amplification
        magnitude[:, :, ~freq_mask] = 0
        
        # Reconstruct
        fft_filtered = magnitude * np.exp(1j * phase)
        filtered_signal = fft.ifft(fft_filtered, axis=2).real
        
        # ✓ FIX: Get current frame
        current_frame = filtered_signal[:, :, -1]
        current_frame = cv.resize(current_frame, (w, h))
        
        return np.clip(current_frame, 0, 255).astype(np.uint8)
    
    def fft_1d_phase(self, amplification=10, freq_min=0.5, freq_max=10, fps=30, subsample=4):
        """
        1D FFT - Amplify phase only
        Returns: Single 2D frame (grayscale)
        """
        if self.frame_count < 8:
            h, w = self.temporal_buffer.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        valid_data = self.get_valid_data_in_order()
        h, w, t = valid_data.shape
        data_small = valid_data[::subsample, ::subsample, :]
        
        fft_result = fft.fft(data_small, axis=2)
        
        # Separate magnitude and phase
        phase = np.angle(fft_result)
        magnitude = np.abs(fft_result)
        
        # Frequency mask
        freqs = fft.fftfreq(self.frame_count, d=1.0/fps)
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        # Amplify phase, zero outside band
        phase_amplified = phase.copy()
        phase_amplified[:, :, freq_mask] *= amplification
        phase_amplified[:, :, ~freq_mask] = 0
        
        # Reconstruct
        fft_filtered = magnitude * np.exp(1j * phase_amplified)
        filtered_signal = fft.ifft(fft_filtered, axis=2).real
        
        # ✓ FIX: Get current frame
        current_frame = filtered_signal[:, :, -1]
        current_frame = cv.resize(current_frame, (w, h))
        
        return np.clip(current_frame, 0, 255).astype(np.uint8)
    
    def fft_3d(self, amplification=10, freq_min=0.5, freq_max=10, fps=30, subsample=4):
        """
        3D FFT - Amplify full FFT result
        Returns: Single 2D frame (grayscale)
        """
        if self.frame_count < 8:
            h, w = self.temporal_buffer.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        valid_data = self.get_valid_data_in_order()
        h, w, t = valid_data.shape
        data_small = valid_data[::subsample, ::subsample, :]
        
        # 3D FFT
        fft_3d = fft.fftn(data_small)
        
        # Temporal frequency mask
        freqs = fft.fftfreq(data_small.shape[2], d=1.0/fps)
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        # Amplify in frequency band
        fft_amplified = fft_3d.copy()
        fft_amplified[:, :, freq_mask] *= amplification
        fft_amplified[:, :, ~freq_mask] = 0
        
        # Inverse FFT
        signal_amplified = fft.ifftn(fft_amplified).real
        
        # Get current frame and resize
        current_frame = signal_amplified[:, :, -1]
        current_frame = cv.resize(current_frame, (w, h))
        
        return np.clip(current_frame, 0, 255).astype(np.uint8)

    def fft_3d_mag(self, amplification=10, freq_min=0.5, freq_max=10, fps=30, subsample=4):
        """
        3D FFT - Amplify magnitude only
        Returns: Single 2D frame (grayscale)
        """
        if self.frame_count < 8:
            h, w = self.temporal_buffer.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        valid_data = self.get_valid_data_in_order()
        h, w, t = valid_data.shape
        data_small = valid_data[::subsample, ::subsample, :]
        
        fft_3d = fft.fftn(data_small)
        
        # Separate magnitude and phase
        phase = np.angle(fft_3d)
        magnitude = np.abs(fft_3d)
        
        # Temporal frequency mask
        freqs = fft.fftfreq(data_small.shape[2], d=1.0/fps)
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        # Amplify magnitude
        mag_amplified = magnitude.copy()
        mag_amplified[:, :, freq_mask] *= amplification
        mag_amplified[:, :, ~freq_mask] = 0
        
        # Reconstruct
        fft_amplified = mag_amplified * np.exp(1j * phase)
        signal_amplified = fft.ifftn(fft_amplified).real
        
        # Get current frame
        current_frame = signal_amplified[:, :, -1]
        current_frame = cv.resize(current_frame, (w, h))
        
        return np.clip(current_frame, 0, 255).astype(np.uint8)

    def fft_3d_phase(self, amplification=10, freq_min=0.5, freq_max=10, fps=30, subsample=4):
        """
        3D FFT - Amplify phase only
        Returns: Single 2D frame (grayscale)
        """
        if self.frame_count < 8:
            h, w = self.temporal_buffer.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        valid_data = self.get_valid_data_in_order()
        h, w, t = valid_data.shape
        data_small = valid_data[::subsample, ::subsample, :]
        
        fft_3d = fft.fftn(data_small)
        
        # Separate magnitude and phase
        phase = np.angle(fft_3d)
        magnitude = np.abs(fft_3d)
        
        # Temporal frequency mask
        freqs = fft.fftfreq(data_small.shape[2], d=1.0/fps)
        freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        
        # Amplify phase
        phase_amplified = phase.copy()
        phase_amplified[:, :, freq_mask] *= amplification
        phase_amplified[:, :, ~freq_mask] = 0
        
        # Reconstruct
        fft_amplified = magnitude * np.exp(1j * phase_amplified)
        signal_amplified = fft.ifftn(fft_amplified).real
        
        # Get current frame
        current_frame = signal_amplified[:, :, -1]
        current_frame = cv.resize(current_frame, (w, h))
        
        return np.clip(current_frame, 0, 255).astype(np.uint8)

#*█████████   Initialize Video Capture   ████████████████████████████████████████
video_path = "./circle-sqare.mp4"
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Error opening video")

fps = cap.get(cv.CAP_PROP_FPS) or 30
video_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(f"Video properties: {video_width}x{video_height} @ {fps} FPS")

success, frame = cap.read()
if not success:
    raise RuntimeError("Could not read first frame")

video_writer    = None
is_recording    = False
output_filename = None

#*█████████   Recording video functions   ████████████████████████████████████████
def get_output_filename(mode):
    """Generate unique output filename"""
    import datetime
    mode_names = ["original", "diff", "fft1d", "fft1d_mag", "fft1d_phase", 
                  "fft3d", "fft3d_mag", "fft3d_phase"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output_{mode_names[mode]}_{timestamp}.mp4"

def initialize_video_writer(filename, fps, width, height):
    """Initialize video writer"""
    codecs = [
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi'),
    ]
    
    for codec, ext in codecs:
        fourcc = cv.VideoWriter_fourcc(*codec)
        if not filename.endswith(ext):
            filename = filename.rsplit('.', 1)[0] + ext
        
        writer = cv.VideoWriter(filename, fourcc, fps, (width, height))
        
        if writer.isOpened():
            print(f"✓ Video writer initialized: {filename} ({codec})")
            return writer, filename
        else:
            writer.release()
    
    raise RuntimeError("Failed to initialize video writer")

#*█████████   Initialize Buffers   ████████████████████████████████████████
frame_buffer = deque(maxlen=MAX_DELAY + 1)
frame_buffer.append(frame)

fft_analyzer = VideoFFTAnalyzer(max_temporal_window=MAX_TEMPORAL_WINDOW)
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
fft_analyzer.add_frame(frame_gray)

#*█████████   Display Modes   ████████████████████████████████████████

print("\n=== Controls ===")
print("Mode 0: Original (unprocessed)")
print("Mode 1: Frame Difference")
print("Mode 2: FFT_1D")
print("Mode 3: FFT_1D_mag")
print("Mode 4: FFT_1D_phase")
print("Mode 5: FFT_3D")
print("Mode 6: FFT_3D_mag")
print("Mode 7: FFT_3D_phase")
print("\nRecord: 0=Off, 1=On")
print("ESC: Exit\n")

#*█████████   MAIN LOOP   ████████████████████████████████████████

while True:
    t_start = time.perf_counter()
    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_buffer.append(frame)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fft_analyzer.add_frame(frame_gray)
    
    #*█████████   Get Trackbar Values   ████████████████████████████████████████
    mode      = cv.getTrackbarPos("Mode", controls_disp)
    delay     = cv.getTrackbarPos("Frame delay", controls_disp)
    amp       = cv.getTrackbarPos("Amplification", controls_disp) / 10
    freq_min  = cv.getTrackbarPos("Freq_min", controls_disp)
    freq_band = cv.getTrackbarPos("Freq Band", controls_disp)
    freq_max  = freq_min + freq_band
    record    = cv.getTrackbarPos("Record", controls_disp)

    #*█████████   Recording State   ████████████████████████████████████████
    if record == 1 and not is_recording:
        output_filename = get_output_filename(mode)
        try:
            video_writer, output_filename = initialize_video_writer(
                output_filename, fps, video_width, video_height
            )
            is_recording = True
            print(f"🔴 Recording: {output_filename}")
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            cv.setTrackbarPos("Record", controls_disp, 0)
            
    elif record == 0 and is_recording:
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            print(f"⏹️  Stopped: {output_filename}")
        is_recording = False
    
    #*█████████   Process Video   ████████████████████████████████████████
   
    if mode == 0:
        display = frame
        
    elif mode == 1:
        if delay < len(frame_buffer):
            old_frame = frame_buffer[-(delay + 1)]
            diff = amp * np.abs(frame.astype(np.int16) - old_frame.astype(np.int16))
            result = np.clip(diff, 0, 255).astype(np.uint8)
        else:
            result = np.zeros_like(frame)
        display = result
    
    elif mode == 2:
        motion_map = fft_analyzer.fft_1d(
            amplification=amp,
            freq_min=freq_min, 
            freq_max=freq_max, 
            fps=fps,
            subsample=4
        )
        display = cv.applyColorMap(motion_map, cv.COLORMAP_INFERNO)
    
    elif mode == 3:
        motion_map = fft_analyzer.fft_1d_mag(
            amplification=amp,
            freq_min=freq_min, 
            freq_max=freq_max, 
            fps=fps,
            subsample=4
        )
        display = cv.applyColorMap(motion_map, cv.COLORMAP_INFERNO)
    
    elif mode == 4:
        motion_map = fft_analyzer.fft_1d_phase(
            amplification=amp,
            freq_min=freq_min, 
            freq_max=freq_max, 
            fps=fps,
            subsample=4
        )
        display = cv.applyColorMap(motion_map, cv.COLORMAP_VIRIDIS)
    
    elif mode == 5:
        amplified = fft_analyzer.fft_3d(
            amplification=amp,
            freq_min=freq_min,
            freq_max=freq_max,
            fps=fps,
            subsample=4
        )
        display = cv.applyColorMap(amplified, cv.COLORMAP_JET)

    elif mode == 6:
        amplified = fft_analyzer.fft_3d_mag(
            amplification=amp,
            freq_min=freq_min,
            freq_max=freq_max,
            fps=fps,
            subsample=4
        )
        display = cv.applyColorMap(amplified, cv.COLORMAP_HOT)

    elif mode == 7:
        amplified = fft_analyzer.fft_3d_phase(
            amplification=amp,
            freq_min=freq_min,
            freq_max=freq_max,
            fps=fps,
            subsample=4
        )
        display = cv.applyColorMap(amplified, cv.COLORMAP_RAINBOW)
    
    else:
        display = frame

    # Add labels
    mode_names = ["Original", "Diff", "FFT1D", "FFT1D_mag", "FFT1D_phase", 
                  "FFT3D", "FFT3D_mag", "FFT3D_phase"]
    cv.putText(display, f"Mode: {mode_names[mode]}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if is_recording:
        cv.circle(display, (display.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv.putText(display, "REC", (display.shape[1] - 80, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #*█████████   Record Frame   ████████████████████████████████████████
    if is_recording and video_writer is not None:
        if display.shape[:2] != (video_height, video_width):
            display_resized = cv.resize(display, (video_width, video_height))
        else:
            display_resized = display
            
        if len(display_resized.shape) == 2:
            display_resized = cv.cvtColor(display_resized, cv.COLOR_GRAY2BGR)
        
        video_writer.write(display_resized)

    #*█████████   Display   ████████████████████████████████████████
    cv.imshow(window_disp, display)
    if cv.waitKey(1) == 27:
        break

    t_end = time.perf_counter()
    print(f"FPS: {1/(t_end-t_start):.1f}", end='\r', flush=True)

# Cleanup
if video_writer is not None:
    video_writer.release()
    print(f"\n✓ Video saved: {output_filename}")

cap.release()
cv.destroyAllWindows()
print("=== Done ===")
