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
- [✓] Build computational models 
- [✓] Implement cyclic universe and black hole correspondence mapping
- [ ] Validate against existing cosmological data

### Phase 3: Application (Q1-Q2 2026)
- [ ] Apply framework to specific cosmic phenomena
- [ ] Publish findings and methodology
- [ ] Explore interdisciplinary applications

## 🔁 Reverse-Engineered Formulas for Cyclic Cosmology

After identifying strong alignment between the Genesis-Sphere framework and cyclic cosmological models (e.g., Steinhardt–Turok, Tolman), we reverse-engineered the original functions to better fit oscillatory, time-symmetric, and bounded universe behavior.

---

### ✅ Key Observations

- The original $\omega$-based sinusoidal projection already maps well to cosmic cycles.
- The original $D(t) = 1 + \alpha t^2$ and $Tf(t) = \frac{1}{1 + \beta(|t| + \epsilon)}$ were unbounded or monotonic — not ideal for modeling repeating or bouncing universes.
- We refactored both to support **recurrence**, **bounded growth**, and **cyclic temporal modulation**.

---

### 📐 Revised Functions

#### 🧭 Sinusoidal Projection Function (unchanged):

$$
S(t) = \frac{1}{1 + \sin^2(\omega t)}
$$

> Still effective for modeling oscillatory projection and smooth time-space modulation.

---

#### 🔁 Modified Dimension Expansion Function:

$$
D_{\text{cyc}}(t) = \frac{1 + \alpha t^2}{1 + \gamma t^4}
$$

> Bounded, symmetric, and avoids runaway inflation. Models entropy or spatial degrees of freedom with damping.

---

#### 🔮 Modified Temporal Flow Function:

$$
Tf_{\text{cyc}}(t) = \frac{\cos^2(\omega t)}{1 + \beta t^2}
$$

> Oscillatory and bounded. Reflects recurring time distortions instead of one-time slowdowns.

---

#### 🧪 New Time-Density Function:

$$
\rho_{\text{cyc}}(t) = S(t) \cdot D_{\text{cyc}}(t)
$$

> Naturally produces smooth, recurring density patterns — ideal for simulating cycles or "bounces" in cosmological time.

---

### 📊 Visual Comparison Summary

- **ρ(t)**: Revised version shows smooth oscillating density vs. original ever-increasing form.
- **Tf(t)**: Now evolves cyclically instead of decaying once near \( t = 0 \).
- **D(t)**: Prevents unbounded growth while preserving expansion-like behavior.
- **S(t)**: Remains effective and elegant for cyclical modulation.

---

### 🚧 Next Steps

- [ ] Replace original functions in codebase with new `*_cyc(t)` variants.
- [ ] Add toggle between inflationary and cyclic models.
- [ ] Run simulations to analyze entropy buildup per cycle.
- [ ] Compare to Steinhardt–Turok cyclic energy curves.

---

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

#### Cyclic Cosmology & Black Hole Physics (`/models`, `/simulations`)

The cyclic cosmology and black hole correspondence model shows how Genesis-Sphere naturally connects these phenomena:

```bash
# Run from project root directory

# Core cyclic/black hole model
python models/cyclic_bh_mapping.py

# Parameter exploration for cyclic cosmology
python simulations/cyclic_cosmology_simulation.py --param-exploration

# Run with custom parameters
python simulations/cyclic_cosmology_simulation.py --omega 1.5 --beta 0.6

# Specify cycle period directly
python simulations/cyclic_cosmology_simulation.py --cycle-period 8
```

Visualizations and documentation are saved to the `/output/cyclic_bh/` directory.

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
- **Cyclic Universe and Black Hole Correspondence**: Visualizations and animations showing how the model connects black hole physics with cyclic cosmology

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
│   ├── cyclic_bh_mapping.py       # Cyclic cosmology and black hole mapping
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
│   ├── cyclic_cosmology_simulation.py   # Cyclic cosmology simulation
│   ├── run_all_animations.py            # Script to run all animations
│   └── genesis_sphere_notebook.ipynb    # Jupyter notebook version
├── validation/                    # Model validation against cosmological data
│   ├── README.md                  # Validation documentation
│   ├── ned_validation.py          # NED Cosmology Calculator validation
│   ├── astropy_validation.py      # Astropy cosmology models comparison
│   ├── observational_validation.py # Observational datasets validation
│   ├── datasets/                  # Directory for cosmological datasets
│   └── results/                   # Validation results and visualizations
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
    └── cyclic_bh/                     # Cyclic cosmology and black hole outputs
        ├── cyclic_cosmology_output.png # Cyclic cosmology visualization
        ├── cyclic_cosmology_animation.mp4 # Cyclic cosmology animation
        ├── bh_mapping_output.png       # Black hole mapping visualization
        └── bh_mapping_animation.mp4    # Black hole mapping animation
```

### Dependency Management

For validation against cosmological models and data, additional Python packages are required:

```bash
# Install standard dependencies
pip install -r requirements.txt

# Install additional packages for cosmological validation
pip install astropy scipy requests beautifulsoup4
```

When working with specific datasets:

1. **Supernovae data**: The Pantheon+ or Union2.1 datasets can be downloaded from their respective websites
2. **CMB data**: Planck mission data available from the [ESA Planck Legacy Archive](https://pla.esac.esa.int/)
3. **BAO measurements**: Available from various surveys like SDSS, BOSS, and eBOSS

Example of loading standard cosmological parameters from astropy:

```python
from astropy.cosmology import Planck18
print(f"Hubble constant (H0): {Planck18.H0.value} km/s/Mpc")
print(f"Matter density (Omega_m): {Planck18.Om0}")
print(f"Dark energy density (Omega_Lambda): {Planck18.Ode0}")
```

---

*This roadmap is a living document and will be updated as the project progresses.*
