"""
Map Visualization Module

This module provides functionality to visualize drone flight paths on maps
using both interactive HTML maps (Folium) and static plots (Matplotlib).
"""

import folium
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import config
import os


class MapVisualizer:
    """
    Visualizes drone flight paths on maps.
    
    Supports both interactive web-based maps using Folium and
    static trajectory plots using Matplotlib.
    """
    
    def __init__(self, coordinates: List[Tuple[float, float]], 
                 center: Optional[Tuple[float, float]] = None):
        """
        Initialize the map visualizer.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            center: Map center coordinates. If None, uses center of path.
        """
        self.coordinates = coordinates
        
        if center is None:
            # Calculate center from coordinates
            lats = [coord[0] for coord in coordinates]
            lons = [coord[1] for coord in coordinates]
            self.center = (np.mean(lats), np.mean(lons))
        else:
            self.center = center
        
        print(f"MapVisualizer initialized with {len(coordinates)} points")
        print(f"Map center: {self.center}")
    
    def create_interactive_map(self, zoom_start: int = None) -> folium.Map:
        """
        Create an interactive HTML map using Folium.
        
        Args:
            zoom_start: Initial zoom level
            
        Returns:
            Folium Map object
        """
        if zoom_start is None:
            zoom_start = config.MAP_ZOOM_START
        
        # Create base map
        m = folium.Map(
            location=self.center,
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add path as a polyline
        folium.PolyLine(
            locations=self.coordinates,
            color=config.PATH_COLOR,
            weight=config.PATH_WEIGHT,
            opacity=config.PATH_OPACITY,
            popup='Drone Flight Path'
        ).add_to(m)
        
        # Add start marker (green)
        folium.Marker(
            location=self.coordinates[0],
            popup='Start',
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
        
        # Add end marker (red)
        folium.Marker(
            location=self.coordinates[-1],
            popup='End',
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(m)
        
        # Add intermediate markers (every N points)
        marker_interval = max(1, len(self.coordinates) // 10)
        for i in range(marker_interval, len(self.coordinates) - 1, marker_interval):
            folium.CircleMarker(
                location=self.coordinates[i],
                radius=3,
                color=config.PATH_COLOR,
                fill=True,
                fillColor=config.PATH_COLOR,
                fillOpacity=0.6,
                popup=f'Point {i}'
            ).add_to(m)
        
        print("Created interactive map with path and markers")
        
        return m
    
    def save_interactive_map(self, filepath: str):
        """
        Create and save an interactive HTML map.
        
        Args:
            filepath: Output HTML file path
        """
        m = self.create_interactive_map()
        m.save(filepath)
        print(f"Saved interactive map to {filepath}")
    
    def plot_trajectory_2d(self, figsize: Tuple[int, int] = (12, 10),
                          show_grid: bool = True) -> plt.Figure:
        """
        Create a static 2D trajectory plot using Matplotlib.
        
        Args:
            figsize: Figure size (width, height)
            show_grid: Whether to show grid lines
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Extract coordinates
        lats = np.array([coord[0] for coord in self.coordinates])
        lons = np.array([coord[1] for coord in self.coordinates])
        
        # Plot 1: Lat/Lon trajectory
        ax1.plot(lons, lats, 'b-', linewidth=2, alpha=0.7, label='Flight Path')
        ax1.plot(lons[0], lats[0], 'go', markersize=12, label='Start', zorder=5)
        ax1.plot(lons[-1], lats[-1], 'ro', markersize=12, label='End', zorder=5)
        
        # Add direction arrows
        arrow_interval = max(1, len(self.coordinates) // 15)
        for i in range(0, len(self.coordinates) - arrow_interval, arrow_interval):
            dx = lons[i + arrow_interval] - lons[i]
            dy = lats[i + arrow_interval] - lats[i]
            ax1.arrow(lons[i], lats[i], dx * 0.8, dy * 0.8,
                     head_width=0.00003, head_length=0.00005,
                     fc='darkblue', ec='darkblue', alpha=0.6, zorder=4)
        
        ax1.set_xlabel('Longitude (degrees)', fontsize=12)
        ax1.set_ylabel('Latitude (degrees)', fontsize=12)
        ax1.set_title('Drone Flight Path - Geographic Coordinates', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(show_grid, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Plot 2: Relative displacement from start
        rel_lats = (lats - lats[0]) * 111111  # Convert to meters
        rel_lons = (lons - lons[0]) * 111111 * np.cos(np.radians(lats[0]))
        
        ax2.plot(rel_lons, rel_lats, 'b-', linewidth=2, alpha=0.7, label='Flight Path')
        ax2.plot(rel_lons[0], rel_lats[0], 'go', markersize=12, label='Start', zorder=5)
        ax2.plot(rel_lons[-1], rel_lats[-1], 'ro', markersize=12, label='End', zorder=5)
        
        # Add direction arrows
        for i in range(0, len(rel_lons) - arrow_interval, arrow_interval):
            dx = rel_lons[i + arrow_interval] - rel_lons[i]
            dy = rel_lats[i + arrow_interval] - rel_lats[i]
            ax2.arrow(rel_lons[i], rel_lats[i], dx * 0.8, dy * 0.8,
                     head_width=max(abs(rel_lons).max(), abs(rel_lats).max()) * 0.03,
                     head_length=max(abs(rel_lons).max(), abs(rel_lats).max()) * 0.05,
                     fc='darkblue', ec='darkblue', alpha=0.6, zorder=4)
        
        ax2.set_xlabel('East-West Displacement (meters)', fontsize=12)
        ax2.set_ylabel('North-South Displacement (meters)', fontsize=12)
        ax2.set_title('Drone Flight Path - Relative Displacement from Start', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(show_grid, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        print("Created 2D trajectory plot")
        
        return fig
    
    def save_trajectory_plot(self, filepath: str, figsize: Tuple[int, int] = (12, 10)):
        """
        Create and save a static trajectory plot.
        
        Args:
            filepath: Output image file path
            figsize: Figure size (width, height)
        """
        fig = self.plot_trajectory_2d(figsize=figsize)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved trajectory plot to {filepath}")
    
    def plot_motion_analysis(self, motion_vectors: np.ndarray, 
                           timestamps: np.ndarray,
                           filepath: str,
                           figsize: Tuple[int, int] = (14, 10)):
        """
        Create detailed motion analysis plots.
        
        Args:
            motion_vectors: Array of motion vectors
            timestamps: Array of timestamps
            filepath: Output image file path
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Convert to numpy arrays if needed
        motion_vectors = np.array(motion_vectors)
        timestamps = np.array(timestamps)
        
        # Calculate motion statistics
        velocities_x = motion_vectors[:, 0]
        velocities_y = motion_vectors[:, 1]
        speeds = np.linalg.norm(motion_vectors, axis=1)
        
        # Plot 1: X velocity over time
        axes[0, 0].plot(timestamps, velocities_x, 'b-', linewidth=1.5)
        axes[0, 0].set_xlabel('Time (seconds)', fontsize=10)
        axes[0, 0].set_ylabel('X Velocity (pixels/frame)', fontsize=10)
        axes[0, 0].set_title('Horizontal Motion Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Y velocity over time
        axes[0, 1].plot(timestamps, velocities_y, 'g-', linewidth=1.5)
        axes[0, 1].set_xlabel('Time (seconds)', fontsize=10)
        axes[0, 1].set_ylabel('Y Velocity (pixels/frame)', fontsize=10)
        axes[0, 1].set_title('Vertical Motion Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Speed over time
        axes[1, 0].plot(timestamps, speeds, 'r-', linewidth=1.5)
        axes[1, 0].set_xlabel('Time (seconds)', fontsize=10)
        axes[1, 0].set_ylabel('Speed (pixels/frame)', fontsize=10)
        axes[1, 0].set_title('Motion Speed Over Time', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=np.mean(speeds), color='orange', linestyle='--', 
                          label=f'Mean: {np.mean(speeds):.2f}', linewidth=2)
        axes[1, 0].legend()
        
        # Plot 4: Motion vector distribution
        axes[1, 1].scatter(velocities_x, velocities_y, c=timestamps, 
                          cmap='viridis', s=20, alpha=0.6)
        axes[1, 1].set_xlabel('X Velocity (pixels/frame)', fontsize=10)
        axes[1, 1].set_ylabel('Y Velocity (pixels/frame)', fontsize=10)
        axes[1, 1].set_title('Motion Vector Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Time (seconds)', fontsize=10)
        
        plt.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved motion analysis plot to {filepath}")
    
    def create_comprehensive_visualization(self, output_dir: str,
                                          motion_vectors: Optional[np.ndarray] = None,
                                          timestamps: Optional[np.ndarray] = None):
        """
        Create all visualizations and save to output directory.
        
        Args:
            output_dir: Directory to save visualizations
            motion_vectors: Optional motion vectors for analysis plots
            timestamps: Optional timestamps for analysis plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create interactive map
        map_path = os.path.join(output_dir, 'path_map.html')
        self.save_interactive_map(map_path)
        
        # Create trajectory plot
        trajectory_path = os.path.join(output_dir, 'trajectory_plot.png')
        self.save_trajectory_plot(trajectory_path)
        
        # Create motion analysis if data provided
        if motion_vectors is not None and timestamps is not None:
            analysis_path = os.path.join(output_dir, 'motion_analysis.png')
            self.plot_motion_analysis(motion_vectors, timestamps, analysis_path)
        
        print(f"\n=== All visualizations saved to {output_dir} ===")
