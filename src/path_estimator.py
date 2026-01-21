"""
Path Estimation Module

This module converts pixel-space motion vectors into real-world coordinates
and constructs a smooth trajectory representing the drone's flight path.
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.ndimage import uniform_filter1d
from scipy import signal
import json
import config


class PathEstimator:
    """
    Estimates drone flight path from motion vectors.
    
    This class accumulates frame-to-frame displacements, applies smoothing,
    and converts relative movements to coordinate-based trajectories.
    """
    
    def __init__(self, motion_vectors: List[np.ndarray], 
                 timestamps: List[float],
                 origin: Tuple[float, float] = None):
        """
        Initialize the path estimator.
        
        Args:
            motion_vectors: List of 2D displacement vectors from optical flow
            timestamps: Corresponding timestamps for each vector
            origin: Starting coordinates [latitude, longitude]. If None, uses config default.
        """
        self.motion_vectors = np.array(motion_vectors)
        self.timestamps = np.array(timestamps)
        self.origin = origin if origin else config.MAP_CENTER
        self.trajectory = None
        self.coordinates = None
        
        print(f"PathEstimator initialized with {len(motion_vectors)} motion vectors")
    
    def remove_outliers(self, threshold: float = None) -> np.ndarray:
        """
        Remove outlier motion vectors using statistical filtering.
        
        Uses the Z-score method: removes vectors that are more than 
        `threshold` standard deviations from the mean.
        
        Args:
            threshold: Number of standard deviations for outlier detection
            
        Returns:
            Cleaned motion vectors array
        """
        if threshold is None:
            threshold = config.OUTLIER_THRESHOLD
        
        # Calculate magnitudes
        magnitudes = np.linalg.norm(self.motion_vectors, axis=1)
        
        # Calculate Z-scores
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        
        if std_mag == 0:
            return self.motion_vectors
        
        z_scores = np.abs((magnitudes - mean_mag) / std_mag)
        
        # Filter outliers
        mask = z_scores < threshold
        cleaned_vectors = self.motion_vectors.copy()
        
        # Replace outliers with interpolated values
        for i in range(len(cleaned_vectors)):
            if not mask[i]:
                # Find nearest valid neighbors
                left_idx = i - 1
                right_idx = i + 1
                
                while left_idx >= 0 and not mask[left_idx]:
                    left_idx -= 1
                
                while right_idx < len(mask) and not mask[right_idx]:
                    right_idx += 1
                
                # Interpolate
                if left_idx >= 0 and right_idx < len(mask):
                    cleaned_vectors[i] = (cleaned_vectors[left_idx] + cleaned_vectors[right_idx]) / 2
                elif left_idx >= 0:
                    cleaned_vectors[i] = cleaned_vectors[left_idx]
                elif right_idx < len(mask):
                    cleaned_vectors[i] = cleaned_vectors[right_idx]
        
        outliers_removed = np.sum(~mask)
        print(f"Removed {outliers_removed} outliers ({outliers_removed/len(mask)*100:.1f}%)")
        
        return cleaned_vectors
    
    def smooth_trajectory(self, vectors: np.ndarray, 
                         window_size: int = None) -> np.ndarray:
        """
        Apply smoothing to motion vectors using a moving average filter.
        
        Args:
            vectors: Motion vectors to smooth
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed motion vectors
        """
        if window_size is None:
            window_size = config.SMOOTHING_WINDOW
        
        if window_size < 2:
            return vectors
        
        # Apply moving average filter to each dimension
        smoothed_x = uniform_filter1d(vectors[:, 0], size=window_size, mode='nearest')
        smoothed_y = uniform_filter1d(vectors[:, 1], size=window_size, mode='nearest')
        
        smoothed_vectors = np.stack([smoothed_x, smoothed_y], axis=1)
        
        print(f"Applied smoothing with window size {window_size}")
        
        return smoothed_vectors
    
    def accumulate_displacements(self, vectors: np.ndarray) -> np.ndarray:
        """
        Accumulate frame-to-frame displacements to create a cumulative path.
        
        Args:
            vectors: Motion vectors (pixel displacements)
            
        Returns:
            Cumulative trajectory in pixel space
        """
        # Cumulative sum to get trajectory
        trajectory = np.cumsum(vectors, axis=0)
        
        # Prepend origin (0, 0)
        trajectory = np.vstack([np.array([0, 0]), trajectory])
        
        return trajectory
    
    def pixels_to_coordinates(self, trajectory: np.ndarray, 
                            scale_factor: float = None) -> List[Tuple[float, float]]:
        """
        Convert pixel-space trajectory to geographic coordinates.
        
        This is a simplified conversion assuming:
        - X axis (horizontal) corresponds to longitude
        - Y axis (vertical) corresponds to latitude
        - The scale factor determines meters per pixel
        
        Args:
            trajectory: Cumulative trajectory in pixel space
            scale_factor: Meters per pixel (depends on altitude and camera FOV)
            
        Returns:
            List of (latitude, longitude) tuples
        """
        if scale_factor is None:
            scale_factor = config.SCALE_FACTOR
        
        # Convert pixels to meters
        trajectory_meters = trajectory * scale_factor
        
        # Convert meters to degrees (approximate)
        # 1 degree latitude ≈ 111,111 meters
        # 1 degree longitude ≈ 111,111 * cos(latitude) meters
        lat_origin, lon_origin = self.origin
        
        meters_per_degree_lat = 111111.0
        meters_per_degree_lon = 111111.0 * np.cos(np.radians(lat_origin))
        
        # X -> longitude, Y -> latitude (note: Y is typically inverted in images)
        # Inverting Y because image coordinates increase downward
        delta_lat = -trajectory_meters[:, 1] / meters_per_degree_lat
        delta_lon = trajectory_meters[:, 0] / meters_per_degree_lon
        
        # Calculate absolute coordinates
        latitudes = lat_origin + delta_lat
        longitudes = lon_origin + delta_lon
        
        coordinates = list(zip(latitudes, longitudes))
        
        print(f"Converted trajectory to {len(coordinates)} coordinate points")
        print(f"Trajectory spans: "
              f"Lat [{min(latitudes):.6f}, {max(latitudes):.6f}], "
              f"Lon [{min(longitudes):.6f}, {max(longitudes):.6f}]")
        
        return coordinates
    
    def estimate_path(self) -> List[Tuple[float, float]]:
        """
        Complete pipeline to estimate the flight path.
        
        Returns:
            List of (latitude, longitude) coordinates representing the path
        """
        print("\n=== Starting Path Estimation ===")
        
        # Step 1: Remove outliers
        cleaned_vectors = self.remove_outliers()
        
        # Step 2: Smooth trajectory
        smoothed_vectors = self.smooth_trajectory(cleaned_vectors)
        
        # Step 3: Accumulate displacements
        self.trajectory = self.accumulate_displacements(smoothed_vectors)
        
        # Step 4: Convert to coordinates
        self.coordinates = self.pixels_to_coordinates(self.trajectory)
        
        print("=== Path Estimation Complete ===\n")
        
        return self.coordinates
    
    def get_statistics(self) -> Dict:
        """
        Calculate statistics about the estimated path.
        
        Returns:
            Dictionary containing path statistics
        """
        if self.coordinates is None:
            raise ValueError("Path not yet estimated. Call estimate_path() first.")
        
        coords_array = np.array(self.coordinates)
        
        # Calculate distances between consecutive points
        distances = np.sqrt(
            np.sum(np.diff(coords_array, axis=0)**2, axis=1)
        ) * 111111  # Convert degrees to meters (approximate)
        
        total_distance = np.sum(distances)
        max_distance = np.max(distances) if len(distances) > 0 else 0
        avg_speed = total_distance / (self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0
        
        stats = {
            'total_points': len(self.coordinates),
            'total_distance_meters': total_distance,
            'max_displacement_meters': max_distance,
            'average_speed_mps': avg_speed,
            'duration_seconds': self.timestamps[-1] if len(self.timestamps) > 0 else 0,
            'bounds': {
                'lat_min': float(np.min(coords_array[:, 0])),
                'lat_max': float(np.max(coords_array[:, 0])),
                'lon_min': float(np.min(coords_array[:, 1])),
                'lon_max': float(np.max(coords_array[:, 1]))
            }
        }
        
        return stats
    
    def export_to_json(self, filepath: str):
        """
        Export motion vectors and coordinates to JSON file.
        
        Args:
            filepath: Output file path
        """
        data = {
            'motion_vectors': self.motion_vectors.tolist(),
            'timestamps': self.timestamps.tolist(),
            'coordinates': self.coordinates,
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported data to {filepath}")
    
    def export_to_geojson(self, filepath: str):
        """
        Export path as GeoJSON LineString.
        
        Args:
            filepath: Output file path
        """
        if self.coordinates is None:
            raise ValueError("Path not yet estimated. Call estimate_path() first.")
        
        # GeoJSON uses [longitude, latitude] order
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": "Drone Flight Path",
                        "statistics": self.get_statistics()
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[lon, lat] for lat, lon in self.coordinates]
                    }
                }
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported GeoJSON to {filepath}")
