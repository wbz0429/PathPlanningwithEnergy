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

**Test energy consumption models (Phase 3):**
```cmd
python test_energy_model.py
```
- Tests physics model (BEMT), neural residual model, and hybrid model
- Validates power computation for various flight states

**Test energy-aware path planning:**
```cmd
python test_energy_aware_planning.py
```
- Tests EnergyAwareCostFunction with different weights
- Compares energy-aware vs distance-only planning
- Validates climb penalty effect (爬升能耗惩罚)

**Compare energy planning strategies:**
```cmd
python test_energy_comparison.py
```
- Detailed analysis of path energy consumption
- Tests different obstacle scenarios (wide-low vs narrow-tall)
- Weight sensitivity analysis

**Energy-aware flight with visualization (Phase 3):**
```cmd
python fly_with_energy_visualization.py
```
- Resets drone to initial position
- Executes energy-aware path planning
- Tracks energy consumption per segment
- Generates `energy_flight_visualization.png` with:
  - 3D trajectory colored by cumulative energy
  - Power consumption charts
  - Energy distribution analysis
  - Flight statistics summary

**Generate energy analysis report (no AirSim required):**
```cmd
python generate_energy_analysis_report.py
```
- Generates three analysis figures:
  - `energy_model_analysis.png`: Power model characteristics
  - `cost_function_analysis.png`: Cost function breakdown
  - `planning_comparison.png`: Energy-aware vs distance-only comparison

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
  - Includes energy-aware planning weights (weight_energy, weight_distance, weight_time)
  - Normalization references (energy_ref, distance_ref, time_ref)
- `rrt_star.py`: RRTStar class - Sampling-based path planner
  - Implements RRT* with rewiring optimization
  - **Energy-aware cost function**: `Cost = w_e×(E/E_ref) + w_d×(D/D_ref) + w_t×(T/T_ref)`
  - EnergyAwareCostFunction class for multi-objective optimization
  - Collision detection based on ESDF safety margins (hard constraint)
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

**`energy/`** - Energy consumption modeling (Phase 3)
- `physics_model.py`: PhysicsEnergyModel - BEMT-based power computation
  - Computes induced, profile, parasite, and climb power components
  - QuadrotorParams dataclass for vehicle parameters
- `neural_model.py`: NeuralResidualModel - Neural network for residual correction
  - Learns discrepancy between physics model and real data
- `hybrid_model.py`: HybridEnergyModel - Combined physics + neural model
  - EnergyCostFunction for path planning integration

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
- RRT*: 2000 iterations, 1.5m step size, 0.8m safety margin
- Receding horizon: 6m local horizon, 40% execution ratio, 2.5 m/s velocity

**Energy-aware planning parameters:**
- `energy_aware`: bool = True (enable/disable energy optimization)
- `weight_energy`: float = 0.6 (energy weight, highest priority)
- `weight_distance`: float = 0.3 (distance weight)
- `weight_time`: float = 0.1 (time weight)
- `energy_ref`: float = 500.0 J (normalization reference)
- `distance_ref`: float = 10.0 m
- `time_ref`: float = 5.0 s

See `fly_planned_path.py:62-86` for runtime configuration example.

## Known Limitations

1. **Single camera with limited FOV**: Only 90° forward-facing view
2. **No loop closure**: Map drift may accumulate over long flights
3. **Static environment assumption**: Does not handle dynamic obstacles
4. **Local minima**: RRT* can get stuck in complex environments, fallback strategies help but don't guarantee success

## Troubleshooting

### Ground Misidentification (Solved)
The system filters ground points in `mapping/incremental_map.py:134-187` using:
- Ground threshold filter: Points with Z > -0.5m (NED coordinates) are filtered as ground
- Drone protection radius: Points within 2.0m of drone are ignored
- Drone area clearing: 2.5m radius cleared before each map update

### Coordinate Systems
- **AirSim camera**: Z forward, X right, Y down
- **AirSim body**: X forward, Y right, Z down
- **World (NED)**: X north, Y east, Z down (ground = 0, altitude = negative Z)

Transform chain: `depth image → camera frame → body frame → world frame (NED)`
See `utils/transforms.py` for implementation.

## Development Notes

### Phase 3 (Current - Energy-Aware Planning)
- `energy/` module implements hybrid physics + neural energy model
- BEMT (Blade Element Momentum Theory) for physics-based power estimation
  - Hover power: ~150W, Cruise power: ~144W at 2m/s
  - Climb penalty: +4.7% to +25% depending on climb rate
- Neural residual model learns correction from real flight data
- **Energy-aware RRT* cost function** integrated in `planning/rrt_star.py`
  - Multi-objective: safety (hard constraint) > energy > distance > time
  - Normalized weighted sum: `Cost = w_e×(E/E_ref) + w_d×(D/D_ref) + w_t×(T/T_ref)`
  - Climb path costs ~3-5% more than horizontal (diluted by distance term)
- Energy tracking in `RecedingHorizonPlanner`:
  - Tracks `total_energy_consumed` during flight
  - Records `energy_history` per segment
  - `get_energy_stats()` returns full statistics
- Visualization: `fly_with_energy_visualization.py` generates flight energy reports
- Test scripts: `test_energy_model.py`, `test_energy_aware_planning.py`, `test_energy_comparison.py`
- Analysis: `generate_energy_analysis_report.py` creates comparison figures

### Phase 2 (Receding Horizon Planning)
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
