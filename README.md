# 🚁 Drone Flight Path Prediction System

A comprehensive system for extracting and visualizing drone flight paths from video footage using optical flow algorithms and computer vision techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Approach](#technical-approach)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Output](#output)
- [Algorithms Explained](#algorithms-explained)
- [Challenges & Solutions](#challenges--solutions)
- [Future Improvements](#future-improvements)
- [Requirements](#requirements)

---

## 🎯 Overview

This project analyzes video footage from a drone to extract its flight path and visualize it on an interactive map. Using optical flow algorithms, the system tracks motion between consecutive frames to estimate the drone's trajectory in real-time.

### Key Objectives

1. **Extract Motion**: Apply optical flow algorithms to track pixel movement across frames
2. **Predict Path**: Convert motion vectors into a continuous flight trajectory
3. **Visualize Results**: Generate interactive maps and plots showing the drone's path
4. **Compare Methods**: Implement both classical and deep learning approaches

---

## ✨ Features

- **Multiple Optical Flow Methods**
  - Sparse Optical Flow (Lucas-Kanade with Shi-Tomasi corner detection)
  - Dense Optical Flow (Farneback algorithm)
  - (Optional) Deep Learning-based flow (RAFT model)

- **Robust Path Estimation**
  - Outlier detection and removal using statistical filtering
  - Trajectory smoothing with moving average filters
  - Pixel-to-coordinate conversion with scale calibration

- **Rich Visualizations**
  - Interactive HTML maps with markers (using Folium)
  - Static trajectory plots with displacement analysis
  - Motion analysis graphs (velocity over time, speed distribution)

- **Comprehensive Output**
  - GeoJSON export for GIS applications
  - JSON export of motion vectors and statistics
  - Detailed configuration logging

---

## 🔬 Technical Approach

### 1. **Optical Flow Analysis**

The system uses optical flow to track motion between consecutive video frames:

**Sparse Optical Flow (Lucas-Kanade)**:
- Detects strong feature points (corners) using Shi-Tomasi corner detector
- Tracks these points across frames using pyramidal Lucas-Kanade method
- Calculates median displacement to get robust motion estimate
- Re-detects features when tracking quality degrades

**Dense Optical Flow (Farneback)**:
- Computes motion vectors for all pixels in the frame
- Uses polynomial expansion to model pixel neighborhoods
- Calculates weighted average flow based on magnitude
- More computationally intensive but captures full scene motion

### 2. **Path Reconstruction**

Motion vectors are converted to a continuous trajectory:

1. **Outlier Removal**: Z-score filtering removes anomalous motion vectors
2. **Smoothing**: Moving average filter reduces noise and jitter
3. **Accumulation**: Cumulative sum of displacements creates trajectory
4. **Coordinate Conversion**: Pixel displacements mapped to lat/lon coordinates

### 3. **Visualization**

Multiple visualization types provide comprehensive path analysis:

- **Interactive Map**: Web-based map with start/end markers and path overlay
- **Trajectory Plot**: 2D visualization showing geographic and relative displacement
- **Motion Analysis**: Time-series plots of velocity components and speed

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
cd path-prediction
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies**:
- `opencv-python`: Video processing and optical flow
- `numpy`: Numerical computations
- `folium`: Interactive map generation
- `matplotlib`: Static plotting
- `scipy`: Signal processing and filtering
- `tqdm`: Progress bars

**Optional (for deep learning)**:
```bash
pip install torch torchvision
```

### Step 3: Verify Installation

```bash
python main.py --help
```

---

## 🚀 Usage

### Basic Usage

Process a video using sparse optical flow (default):

```bash
python main.py
```

This will:
1. Process `video1.MP4` in the current directory
2. Extract motion using Lucas-Kanade optical flow
3. Generate visualizations in the `output/` directory

### Advanced Usage

**Use dense optical flow**:
```bash
python main.py --method dense
```

**Specify custom video file**:
```bash
python main.py --video /path/to/your/video.mp4
```

**Custom output directory**:
```bash
python main.py --output results/my_analysis
```

**Combine options**:
```bash
python main.py --method dense --video drone_flight.mp4 --output results/
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--method` | Optical flow method (`sparse` or `dense`) | `sparse` |
| `--video` | Path to input video file | `video1.MP4` |
| `--output` | Output directory for results | `output` |

---

## 📁 Project Structure

```
path-prediction/
├── video1.MP4                    # Input video file
├── requirements.txt              # Python dependencies
├── config.py                     # Configuration parameters
├── main.py                       # Main execution pipeline
├── README.md                     # This file
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── optical_flow.py          # Optical flow tracking
│   ├── path_estimator.py        # Path reconstruction
│   ├── map_visualizer.py        # Visualization generation
│   └── deep_learning_flow.py    # (Optional) DL-based flow
└── output/                       # Generated results
    ├── path_map.html            # Interactive map (open in browser)
    ├── trajectory_plot.png      # Static trajectory visualization
    ├── motion_analysis.png      # Motion statistics plots
    ├── path.geojson             # GeoJSON path export
    ├── motion_data.json         # Motion vectors and coordinates
    └── config_used.json         # Configuration snapshot
```

---

## ⚙️ Configuration

Edit `config.py` to customize processing parameters:

### Video Processing
```python
FRAME_SKIP = 2              # Process every Nth frame (higher = faster, less accurate)
MAX_FRAMES = None           # Limit processing (None = process all)
```

### Optical Flow Settings
```python
# Lucas-Kanade parameters
LUCAS_KANADE_PARAMS = {
    'winSize': (21, 21),    # Search window size
    'maxLevel': 3,          # Pyramid levels
}

# Feature detection parameters
FEATURE_PARAMS = {
    'maxCorners': 200,      # Maximum corners to detect
    'qualityLevel': 0.01,   # Corner quality threshold
    'minDistance': 10,      # Minimum distance between corners
}
```

### Path Estimation
```python
SMOOTHING_WINDOW = 5        # Moving average window size
OUTLIER_THRESHOLD = 3.0     # Z-score threshold for outliers
SCALE_FACTOR = 1.0          # Meters per pixel (adjust for drone altitude)
```

### Map Visualization
```python
MAP_CENTER = [37.7749, -122.4194]  # Default map center [lat, lon]
MAP_ZOOM_START = 15                 # Initial zoom level
PATH_COLOR = 'blue'                 # Path line color
```

---

## 📊 Output

After running the pipeline, you'll find several output files:

### 1. Interactive Map (`path_map.html`)

Open in any web browser to view:
- Complete flight path as a colored polyline
- Start marker (green) and end marker (red)
- Intermediate waypoints
- Pan, zoom, and click for details

### 2. Trajectory Plot (`trajectory_plot.png`)

Two-panel visualization:
- **Top**: Geographic coordinates (lat/lon) with direction arrows
- **Bottom**: Relative displacement from start in meters

### 3. Motion Analysis (`motion_analysis.png`)

Four-panel analysis:
- **X Velocity**: Horizontal motion over time
- **Y Velocity**: Vertical motion over time
- **Speed**: Motion magnitude over time with mean line
- **Distribution**: Scatter plot of motion vectors

### 4. GeoJSON Export (`path.geojson`)

Standard GIS format containing:
- LineString geometry of the path
- Statistics and metadata
- Can be imported into QGIS, Google Earth, etc.

### 5. Motion Data (`motion_data.json`)

Raw data export including:
- Motion vectors for each frame
- Timestamps
- Coordinate points
- Path statistics

---

## 🧮 Algorithms Explained

### Lucas-Kanade Optical Flow

**How it works**:
1. Detect good features to track (corners, edges) using Shi-Tomasi algorithm
2. For each feature, search for its location in the next frame
3. Use pyramidal implementation for multi-scale tracking
4. Calculate displacement vectors

**Advantages**:
- Fast and efficient
- Works well for distinct features
- Robust to camera rotation

**Limitations**:
- Requires detectable features
- Can lose tracking in uniform regions

### Farneback Optical Flow

**How it works**:
1. Approximate pixel neighborhoods with quadratic polynomials
2. Calculate motion from polynomial coefficients
3. Use pyramid for coarse-to-fine estimation
4. Generate dense flow field for entire image

**Advantages**:
- Captures motion everywhere in the frame
- No feature detection required
- Better for scenes with subtle motion

**Limitations**:
- More computationally expensive
- Sensitive to lighting changes

### Path Reconstruction Algorithm

```
For each frame:
    1. Calculate optical flow → motion vector (dx, dy)
    2. Filter outliers using Z-score method
    3. Apply moving average smoothing
    4. Accumulate displacement: position += motion_vector
    5. Convert pixels to geographic coordinates
```

---

## 🎯 Challenges & Solutions

### Challenge 1: Camera Rotation vs Translation

**Problem**: Optical flow captures both camera rotation and translation, but we only want translation for path tracking.

**Solution**: 
- Use robust statistics (median instead of mean) to reduce rotation effects
- Track multiple features and analyze displacement consensus
- Future: Implement homography decomposition to separate rotation

### Challenge 2: Scale Ambiguity

**Problem**: Without knowing drone altitude or camera properties, pixel displacement has no absolute scale.

**Solution**:
- Provide configurable `SCALE_FACTOR` parameter
- Generate relative paths that can be scaled later
- Export in normalized coordinates for flexibility

### Challenge 3: Drift Accumulation

**Problem**: Small errors in motion vectors accumulate over time, causing path drift.

**Solution**:
- Outlier removal using statistical filtering
- Trajectory smoothing with moving average
- Re-detection of features when tracking quality drops

### Challenge 4: Noisy Motion Vectors

**Problem**: Lighting changes, shadows, and small objects cause noisy flow estimates.

**Solution**:
- Z-score based outlier detection and interpolation
- Weighted averaging based on flow magnitude
- Temporal smoothing across frames

---

## 🚀 Future Improvements

1. **Enhanced Calibration**
   - Automatic scale estimation using known landmarks
   - Camera intrinsic parameter calibration
   - GPS metadata extraction if available

2. **Advanced Motion Models**
   - IMU sensor fusion for better accuracy
   - Kalman filtering for state estimation
   - SLAM (Simultaneous Localization and Mapping)

3. **Deep Learning Enhancements**
   - Fine-tune RAFT on drone footage
   - Depth estimation from monocular video
   - Semantic segmentation to identify moving objects

4. **Real-Time Processing**
   - Optimize for live video streams
   - GPU acceleration for deep learning models
   - Streaming visualization updates

5. **3D Reconstruction**
   - Structure from Motion (SfM)
   - 3D trajectory visualization
   - Altitude estimation

---

## 📦 Requirements

### System Requirements

- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for dependencies + space for video files

### Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Video processing and optical flow |
| numpy | ≥1.24.0 | Numerical computations |
| folium | ≥0.14.0 | Interactive map generation |
| matplotlib | ≥3.7.0 | Static plotting |
| scipy | ≥1.10.0 | Signal processing |
| tqdm | ≥4.65.0 | Progress bars |

### Optional Dependencies (Deep Learning)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | RAFT optical flow model |

---

## 📝 Notes

- **Video Format**: Supports all formats readable by OpenCV (MP4, AVI, MOV, etc.)
- **Coordinate System**: Assumes origin at `MAP_CENTER` in `config.py`
- **Performance**: Processing time depends on video resolution and length
  - Typical: 2-5 minutes for a 1-minute 1080p video
  - Use `FRAME_SKIP` to trade accuracy for speed

---

## 🤝 Contributing

Suggestions for improvements:
1. Implement camera calibration module
2. Add support for stereo vision
3. Integrate with actual GPS data when available
4. Optimize performance with multiprocessing

---

## 📄 License

This project is provided for educational and research purposes.

---

## 👥 Authors

Drone Path Prediction Team

---

## 📞 Support

For issues or questions:
1. Check the configuration in `config.py`
2. Verify video file is accessible
3. Ensure all dependencies are installed
4. Review console output for error messages

---

## 🎓 References

- **Lucas-Kanade Method**: Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique
- **Farneback Algorithm**: Farnebäck, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion
- **RAFT**: Teed, Z., & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
- **OpenCV Documentation**: https://docs.opencv.org/

---

**✨ Happy Flight Path Tracking! ✨**
