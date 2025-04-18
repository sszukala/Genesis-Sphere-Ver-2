# 🛣️ Genesis-Sphere Project Roadmap

This document outlines the development roadmap for the Genesis-Sphere framework, tracking progress on key objectives and potential applications.

## 🎯 Primary Objectives

- [ ] Explore time-space distortions during cosmic evolution
- [ ] Model behavior near singularities with simple, interpretable math
- [x] Build tools or simulations for visualizing time-density spheres
- [ ] Develop alternate cosmological insights using modulated functions

## 🔬 Potential Applications

- [ ] Simulating gravitational lensing using sinusoidal distortions
- [ ] Modeling early universe inflation and contraction
- [ ] Creating cyclic or bouncing universe models
- [ ] Exploring entropy flow and dark energy expansion zones

## 📅 Phase Timeline

### Phase 1: Foundation (Q2-Q3 2025)
- [x] Define mathematical framework
- [x] Create initial proof-of-concept simulations
- [x] Document core equations and principles

### Phase 2: Implementation (Q4 2025)
- [x] Develop visualization tools
- [x] Create 3D/4D visualizations of model dynamics
- [✓] Build computational models  <!-- Changed from unchecked to in-progress -->
- [ ] Validate against existing cosmological data

### Phase 3: Application (Q1-Q2 2026)
- [ ] Apply framework to specific cosmic phenomena
- [ ] Publish findings and methodology
- [ ] Explore interdisciplinary applications

## 📚 Resources Needed

- [ ] Computational resources for simulations
- [ ] Cosmological datasets for validation
- [ ] Collaboration with physics and math specialists
- [ ] Visualization expertise for complex data representation

## 🛠️ Development Setup & Commands

### Environment Setup
```bash
# Install required dependencies
pip install -r requirements.txt

# FFmpeg Installation (required for animations)
# Windows: 
# 1. Download from https://ffmpeg.org/download.html or https://github.com/BtbN/FFmpeg-Builds/releases
# 2. Extract the ZIP file to a folder (e.g., C:\FFmpeg)
# 3. Add the bin folder to your PATH:
#    - Right-click on "This PC" → Properties → Advanced system settings → Environment Variables
#    - Edit the Path variable and add the path to FFmpeg's bin folder (e.g., C:\FFmpeg\bin)
#    - Restart your command prompt/terminal after updating PATH
# 
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

### Running Simulations

The Genesis-Sphere framework organizes code into two main directories:

#### Core Model Implementation (`/models`)

The `/models` directory contains the fundamental computational implementations:

```bash
# Run from project root directory

# Core model parameter sensitivity analysis
python models/run_static_simulation.py

# Parameter evolution animations
python models/animate_density.py
```

#### Visualization Scripts (`/simulations`)

The `/simulations` directory contains higher-level visualization scripts:

```bash
# Run from project root directory

# Basic simulation (static image)
python simulations/genesis_sphere_simulation.py

# Animated simulation (requires FFmpeg for video output)
python simulations/genesis_sphere_animation.py

# Alternative animation script with fallback (works without FFmpeg)
python simulations/genesis_sphere_animation_fallback.py

# Generate 3D and 4D visualizations (static images)
python simulations/genesis_sphere_3d_visualization.py

# Generate animated 3D/4D visualizations (requires FFmpeg)
python simulations/animation_3d_density.py         # Rotating 3D density surface
python simulations/animation_3d_parametric.py      # Growing and rotating 3D parametric curve
python simulations/animation_spacetime_folding.py  # Evolving space-time folding with parameter changes
python simulations/animation_4d_visualization.py   # 4D visualization with pressure wave

# Run all animations in sequence
python simulations/run_all_animations.py
```

#### Interactive Notebooks

For interactive exploration using Jupyter notebooks:

```bash
# Basic model notebook
jupyter notebook simulations/genesis_sphere_notebook.ipynb

# Advanced interactive model exploration with parameter sliders
jupyter notebook models/simulation.ipynb
```

### Completed Visualizations

The project now includes the following visualizations:

- **2D Static Plots**: Basic time-dependent function relationships
- **2D Animations**: Dynamic evolution of model variables over time
- **3D Surface Plots**: Space-time density variation across parameters
- **3D Parametric Curves**: System evolution traces in three dimensions
- **4D Color Mapping**: Using color as a fourth dimension for pressure
- **Space-Time Folding**: Visualization of distortions near singularities

### File Structure
```
Genesis-Sphere-Ver-2/
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
├── mathematical_framework.md      # Formal mathematical definitions
├── roadmap.md                     # This project roadmap
├── models/                        # Core model implementation
│   ├── genesis_model.py           # Main class implementation
│   ├── run_static_simulation.py   # Parameter sensitivity analysis
│   ├── animate_density.py         # Parameter evolution animations
│   └── simulation.ipynb           # Advanced interactive notebook
├── simulations/                   # Visualization scripts
│   ├── genesis_sphere_simulation.py     # Basic simulation script
│   ├── genesis_sphere_animation.py      # Animated visualization script
│   ├── genesis_sphere_animation_fallback.py # Animation with fallback options
│   ├── genesis_sphere_3d_visualization.py   # Static 3D/4D visualization script
│   ├── animation_3d_density.py          # 3D density surface animation
│   ├── animation_3d_parametric.py       # 3D parametric curve animation
│   ├── animation_spacetime_folding.py   # Space-time folding animation
│   ├── animation_4d_visualization.py    # 4D visualization animation
│   ├── run_all_animations.py            # Script to run all animations
│   └── genesis_sphere_notebook.ipynb    # Jupyter notebook version
└── output/                        # Generated visualization outputs
    ├── simulation_output.png          # Static visualization
    ├── genesis_sphere_animation.mp4   # Animation output
    ├── 3d_density_surface.png         # 3D surface visualization
    ├── 3d_parametric_curve.png        # 3D curve visualization
    ├── 4d_visualization.png           # 4D (color as dimension) visualization
    ├── 3d_spacetime_folding.png       # Space-time folding visualization
    ├── 3d_density_animation.mp4       # Animated 3D density surface
    ├── 3d_parametric_animation.mp4    # Animated 3D parametric curve
    ├── spacetime_folding_animation.mp4 # Animated space-time folding
    └── 4d_visualization_animation.mp4 # Animated 4D visualization
```

---

*This roadmap is a living document and will be updated as the project progresses.*
