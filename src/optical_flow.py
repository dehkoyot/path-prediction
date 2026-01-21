"""
Optical Flow Module

This module implements optical flow algorithms for tracking motion in video frames.
Supports both sparse (Lucas-Kanade) and dense (Farneback) optical flow methods.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import config


class OpticalFlowTracker:
    """
    Tracks motion in video using optical flow algorithms.
    
    This class provides methods for extracting frames, detecting features,
    and computing optical flow using both sparse and dense methods.
    """
    
    def __init__(self, video_path: str, method: str = 'sparse'):
        """
        Initialize the optical flow tracker.
        
        Args:
            video_path: Path to the input video file
            method: Flow method to use ('sparse' or 'dense')
        """
        self.video_path = video_path
        self.method = method
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {self.frame_width}x{self.frame_height}, "
              f"{self.fps:.2f} fps, {self.frame_count} frames")
    
    def detect_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect good features to track using Shi-Tomasi corner detection.
        
        Args:
            frame: Grayscale image frame
            
        Returns:
            Array of detected feature points with shape (N, 1, 2)
        """
        corners = cv2.goodFeaturesToTrack(
            frame,
            mask=None,
            **config.FEATURE_PARAMS
        )
        return corners
    
    def compute_sparse_flow(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Compute sparse optical flow using Lucas-Kanade method.
        
        This method tracks specific feature points across consecutive frames.
        
        Returns:
            Tuple of (motion_vectors, timestamps) where:
                - motion_vectors: List of displacement vectors for each frame
                - timestamps: List of timestamps for each frame
        """
        motion_vectors = []
        timestamps = []
        
        # Read first frame
        ret, old_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = self.detect_features(old_gray)
        
        if p0 is None or len(p0) == 0:
            raise ValueError("No features detected in first frame")
        
        frame_idx = 0
        total_frames = self.frame_count // config.FRAME_SKIP
        
        with tqdm(total=total_frames, desc="Computing sparse optical flow") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Skip frames according to config
                if frame_idx % config.FRAME_SKIP != 0:
                    continue
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None,
                    **config.LUCAS_KANADE_PARAMS
                )
                
                if p1 is not None and st is not None:
                    # Select good points
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    if len(good_new) > 0:
                        # Calculate displacement vectors
                        displacements = good_new - good_old
                        
                        # Calculate median displacement (robust to outliers)
                        median_displacement = np.median(displacements, axis=0)
                        motion_vectors.append(median_displacement)
                        
                        # Calculate timestamp
                        timestamp = frame_idx / self.fps
                        timestamps.append(timestamp)
                        
                        # Update previous frame and points
                        old_gray = frame_gray.copy()
                        
                        # Re-detect features if too few remain
                        if len(good_new) < config.FEATURE_PARAMS['maxCorners'] * 0.3:
                            p0 = self.detect_features(old_gray)
                            if p0 is None or len(p0) == 0:
                                print(f"Warning: No features detected at frame {frame_idx}")
                                p0 = good_new.reshape(-1, 1, 2)
                        else:
                            p0 = good_new.reshape(-1, 1, 2)
                    else:
                        # No good points, try to re-detect
                        p0 = self.detect_features(old_gray)
                        if p0 is None or len(p0) == 0:
                            print(f"Warning: Lost all features at frame {frame_idx}")
                            break
                
                pbar.update(1)
                
                # Check if we've reached max frames
                if config.MAX_FRAMES and len(motion_vectors) >= config.MAX_FRAMES:
                    break
        
        self.cap.release()
        print(f"Extracted {len(motion_vectors)} motion vectors")
        
        return motion_vectors, timestamps
    
    def compute_dense_flow(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Compute dense optical flow using Farneback method.
        
        This method computes flow for all pixels in the frame.
        
        Returns:
            Tuple of (motion_vectors, timestamps) where:
                - motion_vectors: List of average flow vectors for each frame
                - timestamps: List of timestamps for each frame
        """
        motion_vectors = []
        timestamps = []
        
        # Read first frame
        ret, old_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        frame_idx = 0
        total_frames = self.frame_count // config.FRAME_SKIP
        
        with tqdm(total=total_frames, desc="Computing dense optical flow") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Skip frames according to config
                if frame_idx % config.FRAME_SKIP != 0:
                    continue
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    old_gray, frame_gray, None,
                    **config.FARNEBACK_PARAMS
                )
                
                # Calculate magnitude and angle of flow
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Filter out low-magnitude flows (likely noise)
                threshold = np.percentile(magnitude, 50)
                mask = magnitude > threshold
                
                if np.any(mask):
                    # Calculate weighted average flow
                    avg_flow_x = np.average(flow[..., 0][mask], weights=magnitude[mask])
                    avg_flow_y = np.average(flow[..., 1][mask], weights=magnitude[mask])
                    
                    motion_vectors.append(np.array([avg_flow_x, avg_flow_y]))
                    
                    # Calculate timestamp
                    timestamp = frame_idx / self.fps
                    timestamps.append(timestamp)
                
                old_gray = frame_gray
                
                pbar.update(1)
                
                # Check if we've reached max frames
                if config.MAX_FRAMES and len(motion_vectors) >= config.MAX_FRAMES:
                    break
        
        self.cap.release()
        print(f"Extracted {len(motion_vectors)} motion vectors")
        
        return motion_vectors, timestamps
    
    def compute_flow(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Compute optical flow based on the selected method.
        
        Returns:
            Tuple of (motion_vectors, timestamps)
        """
        if self.method == 'sparse':
            return self.compute_sparse_flow()
        elif self.method == 'dense':
            return self.compute_dense_flow()
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'sparse' or 'dense'.")
    
    def __del__(self):
        """Release video capture on destruction."""
        if hasattr(self, 'cap'):
            self.cap.release()


def visualize_flow_on_frame(frame: np.ndarray, flow: np.ndarray, 
                           step: int = 16) -> np.ndarray:
    """
    Visualize optical flow vectors on a frame.
    
    Args:
        frame: Input frame
        flow: Dense optical flow field
        step: Step size for drawing flow vectors
        
    Returns:
        Frame with flow vectors drawn
    """
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    
    # Create lines for visualization
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    
    vis = frame.copy()
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    
    return vis
