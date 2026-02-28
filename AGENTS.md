# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Summary

Drone autonomous flight and obstacle avoidance system built on AirSim simulation.
Pipeline: OctoMap + ESDF + RRT* for 3D path planning with energy-aware optimization.
Language: Python 3.9. All code lives under `drone_sim/`.

## Build & Run Commands

All commands run from `drone_sim/` directory with `conda activate drone`.

```cmd
# Install dependencies
pip install numpy airsim opencv-python matplotlib pandas scipy

# Run unit tests (all 7 test cases)
python test_modules.py

# Run a single test function — no test framework, edit main() in test_modules.py:
#   Change the `tests` list to only include the desired test, e.g.:
#   tests = [("RRT*", test_rrt_star)]
python test_modules.py

# Test energy model (Phase 3)
python test_energy_model.py

# Test energy-aware planning
python test_energy_aware_planning.py

# Test energy comparison
python test_energy_comparison.py

# Test AirSim connection (requires running AirSim)
python connect_test.py

# Run path planning verification (requires AirSim)
python test_path_planning.py

# Run full receding horizon flight (requires AirSim)
python fly_planned_path.py

# Generate analysis figures (no AirSim required)
python generate_energy_analysis_report.py
python generate_simple_visualization.py
```

## Test Framework

Tests use a custom framework, not pytest or unittest. Each test file has standalone
`test_*()` functions that return `True`/`False` and print `[OK]`/`[FAIL]` status.
The main runner in `test_modules.py` catches exceptions and prints a summary.

When adding tests, follow the existing pattern:
```python
def test_my_feature():
    """测试描述"""
    print("\n" + "=" * 60)
    print("Test N: MyFeature")
    print("=" * 60)
    # ... test logic ...
    print(f"[OK] Description of what passed")
    return True
```

Then add the tuple `("MyFeature", test_my_feature)` to the `tests` list in `main()`.

## Code Style

### Formatting
- 4-space indentation, no tabs
- Max line length ~100 characters (soft limit, not enforced)
- Blank line between top-level definitions, one blank line between methods
- Two blank lines before top-level class/function definitions

### Imports
Order: stdlib, third-party, local modules. Separated by blank lines.
```python
import numpy as np
import time
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from mapping.voxel_grid import VoxelGrid
from planning.config import PlanningConfig
```

Some modules use `sys.path.append` for cross-package imports — this is an existing
pattern but prefer relative imports within packages when possible.

Use `TYPE_CHECKING` guard for imports only needed by type hints to avoid circular deps:
```python
if TYPE_CHECKING:
    from mapping.voxel_grid import VoxelGrid
```

### Naming
- Classes: `PascalCase` — `VoxelGrid`, `PhysicsEnergyModel`, `RecedingHorizonPlanner`
- Functions/methods: `snake_case` — `compute_esdf`, `get_distance`, `plan_and_execute`
- Private methods: `_underscore_prefix` — `_precompute_constants`, `_smart_sample`
- Constants: defined as class attributes or dataclass fields, not `UPPER_CASE` module globals
- Config parameters: `snake_case` dataclass fields in `planning/config.py`

### Type Hints
Use type hints on all public method signatures. Use `typing` module types:
```python
def compute_energy_for_segment(self, start_pos: np.ndarray, end_pos: np.ndarray,
                                velocity: float, dt: float = 0.1) -> Tuple[float, float]:
```

### Docstrings
Module-level docstring required. Bilingual style: English title, Chinese description.
Method docstrings use Google-style with Chinese descriptions:
```python
def method(self, velocity: np.ndarray) -> float:
    """
    计算诱导功率 (W)

    Args:
        velocity: 飞行速度向量 [vx, vy, vz] (m/s)

    Returns:
        诱导功率 (W)
    """
```

### Data Structures
- Use `@dataclass` for configuration and parameter containers (`PlanningConfig`, `QuadrotorParams`)
- Use `np.ndarray` for all numerical/spatial data (positions, orientations, grids)
- Use `dict` for stats and return values with multiple fields
- Positions are always `[x, y, z]` numpy arrays in NED coordinates

### Error Handling
- `RuntimeError` for precondition failures (e.g., not connected to AirSim)
- `try/except` with `traceback.print_exc()` in test runners
- Print-based status logging with prefixes: `[OK]`, `[FAIL]`, `[WARNING]`, `[DANGER]`
- No formal logging framework — use `print()` with bracketed prefixes

### Coordinate System
All spatial code uses AirSim NED (North-East-Down):
- X: Forward (North), Y: Right (East), Z: Down
- Altitude is negative Z (e.g., -8.0 = 8m above ground)
- Ground level is Z = 0

Transform chain: `depth image -> camera frame -> body frame -> world frame (NED)`
Camera frame: Z forward, X right, Y down.

### Architecture Patterns
- Each package (`mapping/`, `planning/`, `control/`, `utils/`, `energy/`, `visualization/`)
  has an `__init__.py` and focused modules
- Configuration centralized in `planning/config.py` as a `PlanningConfig` dataclass
- `PerformanceMonitor` uses context managers for timing blocks
- Visualization is optional and loaded conditionally with `try/except ImportError`
- Energy model is optional — code degrades gracefully when unavailable

### Key Constraints
- Single forward-facing camera with 90 deg FOV
- Static environment assumption (no dynamic obstacles)
- Grid: 80x80x40 voxels at 0.5m resolution (40x40x20m space)
- Safety margin: 1.0m from obstacles
- Performance target: <150ms total cycle time (perception + mapping + planning + execution)
- AirSim must be running for any flight or perception tests
