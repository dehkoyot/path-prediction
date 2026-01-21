#!/usr/bin/env python3
"""
Drone Path Prediction - Main Pipeline

This script orchestrates the complete pipeline for extracting and visualizing
drone flight paths from video footage using optical flow analysis.

Usage:
    python main.py [--method sparse|dense] [--video VIDEO_PATH]

Author: Drone Path Prediction Team
"""

import os
import sys
import argparse
from datetime import datetime
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.optical_flow import OpticalFlowTracker
from src.path_estimator import PathEstimator
from src.map_visualizer import MapVisualizer
import config


def print_banner():
    """Print a welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║         DRONE FLIGHT PATH PREDICTION SYSTEM              ║
    ║                                                          ║
    ║  Extracting motion from video using Optical Flow        ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(statistics: dict, method: str, video_path: str):
    """
    Print a summary of the analysis.
    
    Args:
        statistics: Path statistics dictionary
        method: Optical flow method used
        video_path: Path to input video
    """
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Video File: {video_path}")
    print(f"Method: {method.upper()} Optical Flow")
    print(f"\nPath Statistics:")
    print(f"  • Total Points: {statistics['total_points']}")
    print(f"  • Total Distance: {statistics['total_distance_meters']:.2f} meters")
    print(f"  • Max Displacement: {statistics['max_displacement_meters']:.2f} meters")
    print(f"  • Average Speed: {statistics['average_speed_mps']:.2f} m/s")
    print(f"  • Duration: {statistics['duration_seconds']:.2f} seconds")
    print(f"\nBounds:")
    print(f"  • Latitude: [{statistics['bounds']['lat_min']:.6f}, {statistics['bounds']['lat_max']:.6f}]")
    print(f"  • Longitude: [{statistics['bounds']['lon_min']:.6f}, {statistics['bounds']['lon_max']:.6f}]")
    print("="*60 + "\n")


def main():
    """Main execution pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract and visualize drone flight path from video'
    )
    parser.add_argument(
        '--method',
        choices=['sparse', 'dense'],
        default='sparse',
        help='Optical flow method to use (default: sparse)'
    )
    parser.add_argument(
        '--video',
        default=config.VIDEO_PATH,
        help='Path to input video file (default: config.VIDEO_PATH)'
    )
    parser.add_argument(
        '--output',
        default=config.OUTPUT_DIR,
        help='Output directory for results (default: config.OUTPUT_DIR)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Validate video file exists
    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  • Video: {args.video}")
    print(f"  • Method: {args.method}")
    print(f"  • Output: {args.output}")
    print(f"  • Frame Skip: {config.FRAME_SKIP}")
    print(f"  • Smoothing Window: {config.SMOOTHING_WINDOW}")
    print()
    
    # Record start time
    start_time = datetime.now()
    
    try:
        # ============================================================
        # STEP 1: Extract Motion Vectors using Optical Flow
        # ============================================================
        print("\n" + "█"*60)
        print("STEP 1: OPTICAL FLOW ANALYSIS")
        print("█"*60 + "\n")
        
        tracker = OpticalFlowTracker(args.video, method=args.method)
        motion_vectors, timestamps = tracker.compute_flow()
        
        if len(motion_vectors) == 0:
            print("ERROR: No motion vectors extracted from video")
            sys.exit(1)
        
        # ============================================================
        # STEP 2: Estimate Flight Path
        # ============================================================
        print("\n" + "█"*60)
        print("STEP 2: PATH ESTIMATION")
        print("█"*60 + "\n")
        
        estimator = PathEstimator(motion_vectors, timestamps)
        coordinates = estimator.estimate_path()
        
        # Get statistics
        statistics = estimator.get_statistics()
        
        # ============================================================
        # STEP 3: Create Visualizations
        # ============================================================
        print("\n" + "█"*60)
        print("STEP 3: VISUALIZATION")
        print("█"*60 + "\n")
        
        visualizer = MapVisualizer(coordinates)
        visualizer.create_comprehensive_visualization(
            args.output,
            motion_vectors=motion_vectors,
            timestamps=timestamps
        )
        
        # ============================================================
        # STEP 4: Export Data
        # ============================================================
        print("\n" + "█"*60)
        print("STEP 4: DATA EXPORT")
        print("█"*60 + "\n")
        
        # Export motion vectors and coordinates
        if config.SAVE_MOTION_VECTORS:
            json_path = os.path.join(args.output, 'motion_data.json')
            estimator.export_to_json(json_path)
        
        # Export GeoJSON
        geojson_path = os.path.join(args.output, 'path.geojson')
        estimator.export_to_geojson(geojson_path)
        
        # Save configuration used
        config_path = os.path.join(args.output, 'config_used.json')
        with open(config_path, 'w') as f:
            config_data = {
                'video_path': args.video,
                'method': args.method,
                'frame_skip': config.FRAME_SKIP,
                'smoothing_window': config.SMOOTHING_WINDOW,
                'outlier_threshold': config.OUTLIER_THRESHOLD,
                'scale_factor': config.SCALE_FACTOR,
                'map_center': config.MAP_CENTER,
                'timestamp': start_time.isoformat()
            }
            json.dump(config_data, f, indent=2)
        print(f"Saved configuration to {config_path}")
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_summary(statistics, args.method, args.video)
        
        print(f"✓ Processing completed in {duration:.2f} seconds")
        print(f"✓ Results saved to: {os.path.abspath(args.output)}")
        print(f"\nOutput Files:")
        print(f"  • Interactive Map: {os.path.join(args.output, 'path_map.html')}")
        print(f"  • Trajectory Plot: {os.path.join(args.output, 'trajectory_plot.png')}")
        print(f"  • Motion Analysis: {os.path.join(args.output, 'motion_analysis.png')}")
        print(f"  • GeoJSON Path: {os.path.join(args.output, 'path.geojson')}")
        if config.SAVE_MOTION_VECTORS:
            print(f"  • Motion Data: {os.path.join(args.output, 'motion_data.json')}")
        print()
        
        print("✓ Done! Open path_map.html in a web browser to view the interactive map.")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
