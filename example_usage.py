#!/usr/bin/env python3
"""
Example Usage Script

This script demonstrates how to use the drone path prediction modules
programmatically in your own Python code.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.optical_flow import OpticalFlowTracker
from src.path_estimator import PathEstimator
from src.map_visualizer import MapVisualizer


def example_basic_usage():
    """
    Example 1: Basic usage with default settings
    """
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60 + "\n")
    
    # Step 1: Extract motion vectors
    tracker = OpticalFlowTracker("video1.MP4", method='sparse')
    motion_vectors, timestamps = tracker.compute_flow()
    
    # Step 2: Estimate path
    estimator = PathEstimator(motion_vectors, timestamps)
    coordinates = estimator.estimate_path()
    
    # Step 3: Create visualizations
    visualizer = MapVisualizer(coordinates)
    visualizer.save_interactive_map("output/example_map.html")
    visualizer.save_trajectory_plot("output/example_trajectory.png")
    
    # Step 4: Get statistics
    stats = estimator.get_statistics()
    print(f"\nPath Statistics:")
    print(f"  Total Points: {stats['total_points']}")
    print(f"  Total Distance: {stats['total_distance_meters']:.2f} meters")
    print(f"  Average Speed: {stats['average_speed_mps']:.2f} m/s")
    
    print("\n✓ Example 1 complete!")


def example_custom_parameters():
    """
    Example 2: Using custom parameters
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Parameters")
    print("="*60 + "\n")
    
    # Use dense optical flow
    tracker = OpticalFlowTracker("video1.MP4", method='dense')
    motion_vectors, timestamps = tracker.compute_flow()
    
    # Custom origin and smoothing
    estimator = PathEstimator(
        motion_vectors, 
        timestamps,
        origin=[37.7749, -122.4194]  # San Francisco
    )
    
    # Custom outlier and smoothing parameters
    cleaned_vectors = estimator.remove_outliers(threshold=2.5)
    smoothed_vectors = estimator.smooth_trajectory(cleaned_vectors, window_size=7)
    estimator.trajectory = estimator.accumulate_displacements(smoothed_vectors)
    estimator.coordinates = estimator.pixels_to_coordinates(estimator.trajectory, scale_factor=2.0)
    
    # Create visualization with custom center
    visualizer = MapVisualizer(estimator.coordinates, center=[37.7749, -122.4194])
    visualizer.save_interactive_map("output/example_custom_map.html")
    
    print("✓ Example 2 complete!")


def example_export_data():
    """
    Example 3: Exporting data in various formats
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Export")
    print("="*60 + "\n")
    
    # Process video
    tracker = OpticalFlowTracker("video1.MP4", method='sparse')
    motion_vectors, timestamps = tracker.compute_flow()
    
    estimator = PathEstimator(motion_vectors, timestamps)
    coordinates = estimator.estimate_path()
    
    # Export to JSON
    estimator.export_to_json("output/example_data.json")
    print("✓ Exported to JSON")
    
    # Export to GeoJSON
    estimator.export_to_geojson("output/example_path.geojson")
    print("✓ Exported to GeoJSON")
    
    # Manual access to data
    print(f"\nAccessing data programmatically:")
    print(f"  First coordinate: {coordinates[0]}")
    print(f"  Last coordinate: {coordinates[-1]}")
    print(f"  Total points: {len(coordinates)}")
    
    print("\n✓ Example 3 complete!")


def example_motion_analysis():
    """
    Example 4: Detailed motion analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Motion Analysis")
    print("="*60 + "\n")
    
    import numpy as np
    
    # Process video
    tracker = OpticalFlowTracker("video1.MP4", method='sparse')
    motion_vectors, timestamps = tracker.compute_flow()
    
    # Analyze motion
    motion_array = np.array(motion_vectors)
    speeds = np.linalg.norm(motion_array, axis=1)
    
    print("Motion Statistics:")
    print(f"  Mean speed: {np.mean(speeds):.2f} pixels/frame")
    print(f"  Max speed: {np.max(speeds):.2f} pixels/frame")
    print(f"  Min speed: {np.min(speeds):.2f} pixels/frame")
    print(f"  Std deviation: {np.std(speeds):.2f} pixels/frame")
    
    # Create motion analysis visualization
    estimator = PathEstimator(motion_vectors, timestamps)
    coordinates = estimator.estimate_path()
    
    visualizer = MapVisualizer(coordinates)
    visualizer.plot_motion_analysis(
        motion_array,
        np.array(timestamps),
        "output/example_motion_analysis.png"
    )
    
    print("\n✓ Example 4 complete!")


if __name__ == '__main__':
    print("\n🚁 Drone Path Prediction - Example Usage\n")
    
    # Check if video exists
    if not os.path.exists("video1.MP4"):
        print("ERROR: video1.MP4 not found in current directory")
        print("Please ensure the video file exists before running examples.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run examples (uncomment the ones you want to try)
    
    print("Running examples...")
    print("(This may take a few minutes depending on video length)\n")
    
    try:
        # Example 1: Basic usage
        example_basic_usage()
        
        # Uncomment to run additional examples:
        # example_custom_parameters()
        # example_export_data()
        # example_motion_analysis()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nCheck the 'output/' directory for generated files.")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
