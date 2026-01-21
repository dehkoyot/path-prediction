"""
Deep Learning-Based Optical Flow Module (Optional)

This module provides an alternative motion estimation approach using
deep learning models. It serves as a comparison to classical optical flow methods.

Note: This requires additional dependencies (PyTorch) to be installed.
Uncomment the relevant lines in requirements.txt to enable this module.

Supported Models:
    - RAFT (Recurrent All-Pairs Field Transforms)
    - FlowNet2
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from tqdm import tqdm
import warnings

# Check if PyTorch is available
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. Deep learning optical flow is disabled. "
        "Install with: pip install torch torchvision"
    )


class DeepFlowTracker:
    """
    Deep learning-based optical flow tracker.
    
    This class uses pre-trained neural networks for motion estimation,
    providing potentially more accurate results than classical methods,
    especially for complex scenes.
    """
    
    def __init__(self, video_path: str, model: str = 'raft', device: str = 'cpu'):
        """
        Initialize the deep learning flow tracker.
        
        Args:
            video_path: Path to input video file
            model: Model to use ('raft' or 'flownet2')
            device: Device to run inference on ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for deep learning optical flow. "
                "Install with: pip install torch torchvision"
            )
        
        self.video_path = video_path
        self.model_name = model
        self.device = torch.device(device)
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Load model
        self.model = self._load_model()
        
        print(f"Deep Learning Flow Tracker initialized")
        print(f"Model: {model}, Device: {device}")
        print(f"Video: {self.frame_width}x{self.frame_height}, {self.fps:.2f} fps")
    
    def _load_model(self):
        """
        Load the pre-trained optical flow model.
        
        Returns:
            Loaded model
        """
        if self.model_name == 'raft':
            return self._load_raft_model()
        elif self.model_name == 'flownet2':
            raise NotImplementedError("FlowNet2 support coming soon")
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _load_raft_model(self):
        """
        Load RAFT model from torchvision.
        
        Returns:
            RAFT model
        """
        try:
            from torchvision.models.optical_flow import raft_large
            model = raft_large(pretrained=True)
            model = model.to(self.device)
            model.eval()
            print("✓ RAFT model loaded successfully")
            return model
        except Exception as e:
            raise ImportError(
                f"Failed to load RAFT model: {e}\n"
                "Make sure torchvision>=0.12.0 is installed."
            )
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for neural network input.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        frame = frame.unsqueeze(0).to(self.device)
        
        return frame
    
    def compute_flow(self, frame_skip: int = 2) -> Tuple[List[np.ndarray], List[float]]:
        """
        Compute optical flow using deep learning model.
        
        Args:
            frame_skip: Process every Nth frame
            
        Returns:
            Tuple of (motion_vectors, timestamps)
        """
        motion_vectors = []
        timestamps = []
        
        # Read first frame
        ret, prev_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        frame_idx = 0
        total_frames = self.frame_count // frame_skip
        
        with torch.no_grad():
            with tqdm(total=total_frames, desc="Computing DL optical flow") as pbar:
                while True:
                    ret, curr_frame = self.cap.read()
                    if not ret:
                        break
                    
                    frame_idx += 1
                    
                    # Skip frames
                    if frame_idx % frame_skip != 0:
                        continue
                    
                    # Preprocess frames
                    prev_tensor = self._preprocess_frame(prev_frame)
                    curr_tensor = self._preprocess_frame(curr_frame)
                    
                    # Compute flow
                    flow = self.model(prev_tensor, curr_tensor)[-1]
                    
                    # Convert to numpy
                    flow_np = flow[0].cpu().numpy().transpose(1, 2, 0)
                    
                    # Calculate average flow (weighted by magnitude)
                    magnitude = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)
                    threshold = np.percentile(magnitude, 50)
                    mask = magnitude > threshold
                    
                    if np.any(mask):
                        avg_flow_x = np.average(flow_np[..., 0][mask], weights=magnitude[mask])
                        avg_flow_y = np.average(flow_np[..., 1][mask], weights=magnitude[mask])
                        
                        motion_vectors.append(np.array([avg_flow_x, avg_flow_y]))
                        
                        timestamp = frame_idx / self.fps
                        timestamps.append(timestamp)
                    
                    prev_frame = curr_frame
                    pbar.update(1)
        
        self.cap.release()
        print(f"Extracted {len(motion_vectors)} motion vectors using {self.model_name}")
        
        return motion_vectors, timestamps
    
    def __del__(self):
        """Release resources on destruction."""
        if hasattr(self, 'cap'):
            self.cap.release()


def compare_methods(video_path: str, output_dir: str):
    """
    Compare classical and deep learning optical flow methods.
    
    This function runs both sparse optical flow and RAFT, then
    generates comparison visualizations.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save comparison results
    """
    import os
    import matplotlib.pyplot as plt
    from optical_flow import OpticalFlowTracker
    from path_estimator import PathEstimator
    
    print("\n" + "="*60)
    print("COMPARING OPTICAL FLOW METHODS")
    print("="*60 + "\n")
    
    # Method 1: Classical (Sparse)
    print("Running Sparse Optical Flow (Lucas-Kanade)...")
    tracker_sparse = OpticalFlowTracker(video_path, method='sparse')
    mv_sparse, ts_sparse = tracker_sparse.compute_flow()
    
    # Method 2: Deep Learning (RAFT)
    if TORCH_AVAILABLE:
        print("\nRunning Deep Learning Optical Flow (RAFT)...")
        try:
            tracker_dl = DeepFlowTracker(video_path, model='raft')
            mv_dl, ts_dl = tracker_dl.compute_flow()
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot paths
            for idx, (mv, ts, label) in enumerate([
                (mv_sparse, ts_sparse, 'Sparse (Lucas-Kanade)'),
                (mv_dl, ts_dl, 'Deep Learning (RAFT)')
            ]):
                estimator = PathEstimator(mv, ts)
                coords = estimator.estimate_path()
                
                lats = [c[0] for c in coords]
                lons = [c[1] for c in coords]
                
                # Plot trajectory
                axes[0, idx].plot(lons, lats, 'b-', linewidth=2, alpha=0.7)
                axes[0, idx].plot(lons[0], lats[0], 'go', markersize=10)
                axes[0, idx].plot(lons[-1], lats[-1], 'ro', markersize=10)
                axes[0, idx].set_title(f'{label} - Path', fontweight='bold')
                axes[0, idx].set_xlabel('Longitude')
                axes[0, idx].set_ylabel('Latitude')
                axes[0, idx].grid(True, alpha=0.3)
                
                # Plot speeds
                speeds = np.linalg.norm(mv, axis=1)
                axes[1, idx].plot(ts, speeds, 'r-', linewidth=1.5)
                axes[1, idx].set_title(f'{label} - Speed', fontweight='bold')
                axes[1, idx].set_xlabel('Time (seconds)')
                axes[1, idx].set_ylabel('Speed (pixels/frame)')
                axes[1, idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, 'method_comparison.png')
            fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n✓ Comparison saved to {comparison_path}")
            
        except Exception as e:
            print(f"\n✗ Deep learning method failed: {e}")
            print("  Continuing with classical method only.")
    else:
        print("\n✗ PyTorch not available. Skipping deep learning comparison.")


# Example usage
if __name__ == '__main__':
    print("Deep Learning Optical Flow Module")
    print("=" * 60)
    
    if TORCH_AVAILABLE:
        print("✓ PyTorch is available")
        print("  This module can be used for deep learning-based optical flow")
    else:
        print("✗ PyTorch is not available")
        print("  Install with: pip install torch torchvision")
        print("  Uncomment the torch lines in requirements.txt")
