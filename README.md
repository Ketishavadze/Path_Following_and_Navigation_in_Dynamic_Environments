# Numerical Programming Final Project  
**Path Following and Navigation in Dynamic Environments**  

Ketevan Shavadze  
Kutaisi International University  

---

# Project Overview

This project implements three progressively complex navigation tasks:

1. **Task 1 — Path Extraction and Robot Tracking**
2. **Task 2 — Bidirectional Swarm Navigation**
3. **Task 3 — Robot Navigation in Pedestrian Flow**

All tasks are implemented using:
- numerical integration (RK4),
- potential-field control,
- spline-based path modeling,
- optical flow velocity fields,
- modular architecture with centralized configuration.

---

# Project Structure

```

NP_FINALPROJECT/
│
├── data/
│   ├── map/
│   ├── video/
│   └── params/
│       ├── task1_default.yaml
│       ├── task2_default.yaml
│       └── task3_default.yaml
│
├── outputs/
│   ├── task1/
│   ├── task2/
│   └── task3/
│
├── src/
│   ├── common/
│   │   ├── config.py
│   │   ├── io.py
│   │   └── viz.py
│   │
│   ├── models/
│   │   ├── controllers.py
│   │   ├── forces.py
│   │   ├── robot.py
│   │   └── swarm.py
│   │
│   ├── numerics/
│   │   └── rk.py
│   │
│   ├── path/
│   │   ├── skeleton_route.py
│   │   ├── splines.py
│   │   └── corridor.py
│   │
│   └── tasks/
│       ├── task1_path_follow.py
│       ├── task2_swarm_corridor.py
│       └── task3_pedestrian_navigation.py
│
├── tests/
├── requirements.txt
├── report.pdf
└── Presentation.pdf

````

---

# Installation

Create virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate   
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Tasks

## Task 1 — Path Following

```bash
python src/tasks/task1_path_follow.py
```

Output:

```
outputs/task1/task1_map.png
outputs/task1/task1_border_violation.png
```

---

## Task 2 — Bidirectional Swarm Navigation

```bash
python src/tasks/task2_swarm_corridor.py
```

Output:

```
outputs/task2/task2_map.png
outputs/task2/task2_min_distance.png
outputs/task2/task2_animation.gif
```

---

## Task 3 — Pedestrian Flow Navigation

```bash
python src/tasks/task3_pedestrian_navigation.py
```

Output:

```
outputs/task3/task3_two_robots_navigation.mp4
```

Robots:

* **K-VRC** (top flow)
* **XBOT** (bottom flow)

---

# Mathematical Model


All robots use second-order dynamics:

$$
\dot{x} = v
$$

$$
\dot{v} = \frac{1}{m} (F_{track} + F_{rep} + F_{lane})
$$

where:

* $F_{track}$ = tracking control
* $F_{rep}$ = smooth repulsion
* $F_{lane}$ = vertical band stabilization (Task 3)

Numerical integration is performed using **RK4**.

---

# Automated Tests

Tests are implemented using `pytest`.

Run:

```bash
python -m pytest -v
```

Example output:

```
6 tests collected
4 PASSED
1 XPASS
1 FAILED
```

### Passing Tests

* RK4 convergence (harmonic oscillator)
* Path following inside corridor
* Bidirectional swarm movement
* Optical flow direction detection

### Known Limitation

One swarm safety test may fail under strict density conditions.
This highlights a limitation of potential-field collision avoidance in dense environments.

---

# Numerical Methods

* RK4 integration
* BFS shortest path
* Skeletonization
* Cubic spline fitting
* Optical flow (Farnebäck method)

---

# Configuration

All parameters are centralized in:

```
data/params/task1_default.yaml
data/params/task2_default.yaml
data/params/task3_default.yaml
```

No hardcoded parameters exist inside task scripts.

---

# Key Features

* Modular architecture
* Centralized configuration
* Clean separation of models, numerics, and visualization
* Reproducible results
* Demonstrated limitations
* Automated testing

---

# Limitations

* Potential-field methods do not guarantee strict collision avoidance under extreme density.
* Optical flow can introduce noise near boundaries.
* Performance depends on parameter tuning (gains, safe radius, dt).

---

# Technologies Used

* Python
* NumPy
* OpenCV
* SciPy
* Matplotlib
* scikit-image
* imageio
* PyYAML
* pytest

---

# Academic Note

This project was developed as part of the Numerical Programming course at Kutaisi International University.

All simulations and tests are fully reproducible using the provided configuration files.

