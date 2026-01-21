"""
Configuration file for drone path prediction system.

Contains all tunable parameters for video processing, optical flow,
and visualization settings.
"""

# Video Processing Settings
VIDEO_PATH = "video1.MP4"
FRAME_SKIP = 2  # Process every Nth frame (higher = faster but less accurate)
MAX_FRAMES = None  # None = process all frames, or set a limit for testing

# Optical Flow Settings - Lucas-Kanade (Sparse)
LUCAS_KANADE_PARAMS = {
    'winSize': (21, 21),  # Size of search window at each pyramid level
    'maxLevel': 3,  # Number of pyramid levels
    'criteria': (3, 10, 0.03)  # Termination criteria (type, max_iter, epsilon)
}

# Feature Detection Settings - Shi-Tomasi
FEATURE_PARAMS = {
    'maxCorners': 200,  # Maximum number of corners to detect
    'qualityLevel': 0.01,  # Minimal quality of corners
    'minDistance': 10,  # Minimum distance between corners
    'blockSize': 7  # Size of averaging block
}

# Optical Flow Settings - Farneback (Dense)
FARNEBACK_PARAMS = {
    'pyr_scale': 0.5,  # Pyramid scale
    'levels': 3,  # Number of pyramid layers
    'winsize': 15,  # Averaging window size
    'iterations': 3,  # Number of iterations at each pyramid level
    'poly_n': 5,  # Size of pixel neighborhood
    'poly_sigma': 1.2,  # Standard deviation of Gaussian for polynomial expansion
    'flags': 0
}

# Path Estimation Settings
SMOOTHING_WINDOW = 5  # Window size for moving average smoothing
OUTLIER_THRESHOLD = 3.0  # Standard deviations for outlier removal
SCALE_FACTOR = 1.0  # Meters per pixel (adjust based on drone altitude/camera)

# Map Visualization Settings
# IMPORTANT: Set this to your actual drone starting location!
# Examples:
#   Kyiv, Ukraine: [50.4501, 30.5234]
#   Lviv, Ukraine: [49.8397, 24.0297]
#   San Francisco: [37.7749, -122.4194]
MAP_CENTER = [50.4501, 30.5234]  # Update to actual location - [latitude, longitude]
MAP_ZOOM_START = 15
PATH_COLOR = 'blue'
PATH_WEIGHT = 3
PATH_OPACITY = 0.8

# Output Settings
OUTPUT_DIR = "output"
SAVE_MOTION_VECTORS = True
SAVE_DEBUG_FRAMES = False  # Save frames with flow visualization
DEBUG_FRAMES_INTERVAL = 30  # Save every Nth debug frame
