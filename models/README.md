# Genesis-Sphere Computational Model

This directory contains the computational implementation of the Genesis-Sphere theoretical framework, allowing for parameterized simulations, analysis, and visualization of the model's behavior.

## 📋 Overview

The computational model encapsulates the Genesis-Sphere equations in a structured, object-oriented design that enables:

- Parameter exploration and sensitivity analysis
- Time-evolving simulations
- Interactive visualizations and animations
- Scenario-based model investigations

## 🧮 Key Components

### `genesis_model.py`

The core class implementation of the Genesis-Sphere model with all fundamental equations:

```python
# Example: Creating a model with custom parameters
from genesis_model import GenesisSphereModel

model = GenesisSphereModel(alpha=0.03, beta=1.2, omega=1.5, epsilon=0.08)
t = np.linspace(-12, 12, 1000)  # Time domain
results = model.evaluate_all(t)  # Calculate all functions
```

Key methods:
- `rho(t)` - Calculate time-density geometry
- `tf(t)` - Calculate temporal flow ratio
- `velocity(t)` - Calculate modulated velocity
- `pressure(t)` - Calculate modulated pressure
- `evaluate_all(t)` - Calculate all functions at once
- `plot_all()` - Generate comprehensive plots

### `run_static_simulation.py`

Runs static parameter analyses to understand model behavior:

```bash
python run_static_simulation.py
```

This script:
1. Performs parameter sensitivity analysis
2. Generates scenario-based simulations
3. Produces visualizations showing how parameters affect outcomes

### `animate_density.py`

Creates animated visualizations of the model:

```bash
python animate_density.py
```

This script produces:
1. Parameter evolution animations
2. Time-window animations that slide through the model's behavior
3. 3D density evolution animations

### `simulation.ipynb`

An interactive Jupyter notebook for exploring the model, with:

1. Parameter sliders for real-time experimentation
2. Comparative parameter space mapping
3. Custom scenario creation and analysis
4. Tutorials on model usage

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib
- (Optional) Jupyter for notebook exploration

### Basic Usage

1. Import the model:
   ```python
   from genesis_model import GenesisSphereModel
   ```

2. Create a model instance:
   ```python
   model = GenesisSphereModel()  # Default parameters
   # or
   model = GenesisSphereModel(alpha=0.02, beta=0.8, omega=1.0, epsilon=0.1)  # Custom parameters
   ```

3. Calculate model values:
   ```python
   t = np.linspace(-12, 12, 1000)  # Time domain
   density = model.rho(t)          # Calculate density
   temporal_flow = model.tf(t)     # Calculate temporal flow
   # or
   results = model.evaluate_all(t)  # Calculate all functions at once
   ```

4. Visualize results:
   ```python
   model.plot_all()  # Generate comprehensive plots
   plt.show()
   ```

## 🧪 Example Scenarios

The model comes with pre-defined scenarios to explore different physics:

1. **Standard Universe** - Baseline parameters representing normal space-time
2. **High Density Oscillation** - Increased frequency with higher oscillation amplitude
3. **Extreme Time Dilation** - Strong time dilation effects near singularities
4. **Rapid Expansion** - Accelerated spatial expansion with moderate flow modulation

## 🔗 Integration with Visualizations

This computational model serves as the foundation for all visualizations in the broader Genesis-Sphere project. The model outputs can be directly fed into the visualization scripts to create:

- 2D Function plots
- 3D Surface visualizations
- 4D color-mapped visualizations
- Animations showing parameter evolution
- Interactive dashboards

## 📊 Example Output

Running the static simulation produces parameter sensitivity analyses like this:

```
Analyzing sensitivity to alpha...
Analyzing sensitivity to beta...
Analyzing sensitivity to omega...
Analyzing sensitivity to epsilon...
Sensitivity analysis completed. Results saved to ../output/model_analysis
```

## 🔍 Advanced Usage

For more advanced applications, explore:

1. Parameter space mapping to find optimal configurations
2. Time-stepping simulations for dynamic evolution
3. Custom metric extraction for quantitative analysis
4. Integration with other physical models
