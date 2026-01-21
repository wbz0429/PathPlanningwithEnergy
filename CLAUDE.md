# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a drone autonomous flight and obstacle avoidance system based on AirSim simulation. The project implements the **OctoMap + ESDF + RRT*** technical pipeline for 3D path planning and obstacle avoidance.

**Technology Stack**: Python 3.9, AirSim, OpenCV, NumPy, SciPy, Matplotlib

## Environment Setup

### Creating Conda Environment
```cmd
conda create -n drone python=3.9
conda activate drone
```

### Installing Dependencies
```cmd
cd drone_sim
pip install numpy
pip install airsim
pip install opencv-python matplotlib pandas scipy
```

Or use the installation script:
```cmd
cd drone_sim
install_deps.bat
```

## Running the System

### Prerequisites
- AirSim simulator must be running before executing any drone control scripts
- Activate the conda environment: `conda activate drone`
- All commands should be run from the `drone_sim/` directory

### Main Commands

**Test AirSim connection:**
```cmd
python connect_test.py
```

**Run main autonomous flight with obstacle detection:**
```cmd
python main_control.py
```
- Drone flies forward at 3 m/s for 60 seconds
- Uses simple reactive obstacle avoidance (turn right when obstacle detected)
- Displays live camera feed with obstacle detection visualization
- Logs flight data to `logs/` directory

**Test path planning (feasibility verification):**
```cmd
python test_path_planning.py
```
- Captures depth image from AirSim
- Builds 3D voxel grid map
- Computes ESDF distance field
- Plans path using RRT*
- Generates visualization: `path_planning_result.png`

**Execute receding horizon planning flight (Phase 2):**
```cmd
python fly_planned_path.py
```
- Takes off to specified height
- Implements receding horizon planning loop:
  - Captures depth image and updates incremental map
  - Plans local path (6m horizon) toward global goal
  - Executes 40% of planned path
  - Repeats until goal reached
- Real-time 3D visualization of mapping and planning
- Saves result as `receding_horizon_result.png`
- Lands automatically

**Run unit tests:**
```cmd
python test_modules.py
```
- Tests all Phase 2 modules: VoxelGrid, ESDF, IncrementalMap, RRT*, transforms, performance monitoring
- 7 test cases covering core functionality

**Visualize depth data:**
```cmd
python visualize_depth.py    # 2D depth visualization
python show_3d_depth.py       # 3D point cloud visualization
```

**Generate visualization for paper/presentation (no AirSim required):**
```cmd
python generate_simple_visualization.py
```
- Creates simulated flight visualization with obstacles
- Output: `simple_3d_visualization.mp4`, `simple_3d_final.png`

**Real flight with 3D visualization:**
```cmd
python fly_with_simple_visualization.py
```
- Combines receding horizon flight with real-time 3D visualization
- Output: `airsim_flight_visualization.mp4`, `airsim_flight_final.png`

### Quick Start (Batch Files)

```cmd
start_airsim.bat           # Launch AirSim simulator
launch_visualization.bat   # Interactive menu for visualization options
```

## Architecture

### Core Pipeline (Phase 2: Receding Horizon Planning)
```
Depth Image → Incremental Map → ESDF → RRT* (Local) → Execute Partial Path → Repeat
     ↓              ↓              ↓         ↓
  Perception    Ray Casting   Distance   Collision      (Loop until goal reached)
                              Field      Detection
```

### Module Structure

The codebase is organized into modular packages:

**`mapping/`** - Map building and representation
- `voxel_grid.py`: VoxelGrid class - 3D occupancy grid using numpy arrays
  - Converts depth images to 3D occupancy grid
  - World coordinate ↔ grid index conversion
  - Grid size: 80×80×40, voxel size: 0.5m
- `esdf.py`: ESDF class - Euclidean Signed Distance Field
  - Uses `scipy.ndimage.distance_transform_edt` for efficient computation
  - Provides distance to nearest obstacle for any point
  - Supports gradient computation for potential field methods
- `incremental_map.py`: IncrementalMapManager - Multi-frame map accumulation
  - Accumulates depth images across multiple frames
  - Handles coordinate transforms (camera → world NED)
  - Maintains sliding window to prevent memory overflow
  - Ray casting to mark free space
  - Preserves observed obstacles (no overwriting)
  - Performance: ~20ms map update, ~40ms ESDF computation

**`planning/`** - Path planning algorithms
- `config.py`: PlanningConfig dataclass - Centralized configuration parameters
- `rrt_star.py`: RRTStar class - Sampling-based path planner
  - Implements RRT* with rewiring optimization
  - Collision detection based on ESDF safety margins
  - Path smoothing post-processing
  - Performance: ~52ms for local planning
- `receding_horizon.py`: RecedingHorizonPlanner - Main planning controller
  - Implements receding horizon (model predictive) planning
  - Loop: sense → map → plan local → execute partial → repeat
  - Local horizon: 6m, executes 40% of planned path
  - Exploration strategy with information gain scoring
  - Fallback mechanisms for planning failures
  - Total cycle time: ~100-150ms

**`control/`** - Drone interface and control
- `drone_interface.py`: DroneInterface class - AirSim API wrapper
  - Simplified interface for takeoff, movement, landing
  - Pose and depth image acquisition
  - Handles AirSim connection management

**`utils/`** - Utility functions
- `transforms.py`: Coordinate transformation utilities
  - Quaternion → rotation matrix
  - Depth image → camera frame point cloud
  - Camera frame → world frame (NED) transformation
  - Coordinate systems: AirSim camera (Z forward, X right, Y down), body (X forward, Y right, Z down), world (NED)
- `performance.py`: PerformanceMonitor - Timing and profiling
  - Tracks perception, mapping, planning, execution times
  - Context manager for easy timing blocks

**`visualization/`** - Visualization tools
- `planning_visualizer.py`: PlanningVisualizer - Real-time 3D visualization
  - Three-view display: 3D map, XY plane (top), XZ plane (side)
  - Shows obstacles, current position, goals, planned path, executed trajectory
  - Updates dynamically during flight
  - Saves final result as PNG

**Legacy modules (Phase 1):**
- `perception.py`: ObstacleDetector - Simple depth-based obstacle detection
- `main_control.py`: DroneController - Reactive obstacle avoidance (turn right)
- `logger.py`: FlightLogger - CSV-based flight data recording
- `path_planning.py`: Original monolithic implementation (superseded by modular version)

### Coordinate System

**AirSim NED (North-East-Down) coordinates:**
- X: Forward (North)
- Y: Right (East)
- Z: Down (negative altitude)

**Important:** Flight height is specified as negative Z values (e.g., -8.0 means 8 meters above ground)

### Key Configuration

All planning parameters are centralized in `planning/config.py` (`PlanningConfig` dataclass). Key defaults:
- Grid: 80×80×40 voxels at 0.5m resolution (40×40×20m space)
- RRT*: 2000 iterations, 1.5m step size, 1.0m safety margin
- Receding horizon: 6m local horizon, 40% execution ratio, 2.5 m/s velocity

See `fly_planned_path.py:62-86` for runtime configuration example.

## Known Limitations

1. **Single camera with limited FOV**: Only 90° forward-facing view
2. **No loop closure**: Map drift may accumulate over long flights
3. **Static environment assumption**: Does not handle dynamic obstacles
4. **Local minima**: RRT* can get stuck in complex environments, fallback strategies help but don't guarantee success

## Development Notes

### Phase 2 (Current - Receding Horizon Planning)
- Modular architecture with separate packages for mapping, planning, control, utils, visualization
- Incremental mapping solves occlusion problem by accumulating observations
- Receding horizon planning enables navigation to goals behind obstacles
- Performance: ~100-150ms total cycle time (perception + mapping + planning + execution)
- Unit tests cover all core modules (7/7 passing)
- Real-time 3D visualization shows mapping and planning process

### Phase 1 (Legacy - Single-shot Planning)
- Monolithic `path_planning.py` demonstrates OctoMap + ESDF + RRT* pipeline
- `main_control.py` implements simple reactive avoidance (turn right when obstacle detected)
- Performance benchmarks: Map building ~40ms, ESDF ~38ms, RRT* ~24ms
- Test scripts generate PNG visualizations for verification
- Flight logs are CSV files with full state history

### Testing Strategy
- Run `test_modules.py` for unit tests before flight tests
- Use `test_path_planning.py` for single-shot planning verification
- Use `fly_planned_path.py` for full receding horizon flight tests
- Always ensure AirSim is running before executing flight scripts
