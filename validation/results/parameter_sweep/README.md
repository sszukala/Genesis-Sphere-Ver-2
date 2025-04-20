# Parameter Sweep Validation

This directory contains results from systematic parameter sweeps of the Genesis-Sphere model. These sweeps help identify optimal parameter combinations by testing model performance against astronomical datasets.

## Overview

The parameter sweep systematically explores the parameter space (primarily ω and β) to find combinations that best match astronomical observations including:
- Hubble constant (H₀) measurements
- Type Ia supernovae distance modulus data
- Baryon acoustic oscillation (BAO) signals

## Latest Results

Our most recent comprehensive parameter sweep found optimal values:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Omega (ω) | 3.5000 | Angular frequency |
| Beta (β)  | -0.0333 | Temporal damping factor |

**Performance Metrics:**
- H₀ Correlation: -15.39%
- Supernovae R²: -9.61%
- BAO Effect Size: 13.42
- Combined Score: 0.1000

## Running Parameter Sweeps

### Basic Usage

```bash
# Run with default settings (large parameter sweep)
python validation/parameter_sweep_validation.py

# Test mode with limited computation for quick verification
python validation/parameter_sweep_validation.py --test_mode

# Resume from a previous state file
python validation/parameter_sweep_validation.py --resume="path/to/state_file.npz"
```

### Customizing Parameters

```bash
# Change parameter ranges with custom initial values
python validation/parameter_sweep_validation.py --initial_omega 3.5 --initial_beta -0.0333

# Change fixed parameters
python validation/parameter_sweep_validation.py --alpha 0.05 --epsilon 0.2

# Adjust MCMC settings
python validation/parameter_sweep_validation.py --nwalkers 64 --nsteps 10000 --nburn 2000

# Set maximum runtime (useful for long-running sweeps)
python validation/parameter_sweep_validation.py --max_time 120  # 2 hours
```

### GPU Acceleration

Parameter sweeps can be significantly accelerated using CUDA GPU computing:

```bash
# Verify GPU is working correctly and exit
python validation/parameter_sweep_validation.py --verify_gpu

# Force CPU mode even if GPU is available
python validation/parameter_sweep_validation.py --force_cpu

# Run with GPU acceleration (automatic if available)
python validation/parameter_sweep_validation.py
```

### Output Control

```bash
# Add custom suffix to output files
python validation/parameter_sweep_validation.py --output_suffix="omega_range_test"

# Change checkpoint frequency
python validation/parameter_sweep_validation.py --checkpoint_interval 200
```

## GPU Troubleshooting

If you encounter GPU errors like missing CUDA DLLs or CuPy import issues:

1. **Check Python and pip environments**:
   ```bash
   # Verify which Python is being used
   python --version
   python -c "import sys; print(sys.executable)"
   
   # Verify which pip is being used
   pip --version
   
   # Make sure they point to the same Python installation
   ```

2. **Install matching CUDA Toolkit**:
   - Check CUDA version in your system: `nvidia-smi`
   - Install matching CUDA toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-toolkit-archive)
   - Make sure PATH includes the CUDA bin directory

3. **Install CuPy correctly**:
   ```bash
   # First uninstall any existing CuPy installations
   pip uninstall cupy cupy-cuda11x cupy-cuda12x -y
   
   # For CUDA 11.x (11.0-11.8)
   pip install cupy-cuda11x
   
   # For CUDA 12.x
   pip install cupy-cuda12x
   
   # Verify installation works in the SAME Python environment
   python -c "import cupy as cp; print('CuPy works! Version:', cp.__version__)"
   ```

4. **Troubleshoot module not found errors**:
   - If pip says package is installed but Python can't find it:
     ```bash
     # Show where pip is installing packages
     pip list -v
     
     # Check Python's module search paths
     python -c "import sys; print(sys.path)"
     
     # Try installing with the --user flag if using system Python
     pip install --user cupy-cuda12x
     ```

5. **Verify GPU detection**:
   ```bash
   # Check GPU devices
   nvidia-smi
   
   # Run detailed GPU verification
   python validation/parameter_sweep_validation.py --verify_gpu
   ```

6. **Environment variables**:
   - Set `CUDA_PATH` to your CUDA installation directory
   - Add CUDA bin directory to your PATH environment variable
   - Example: `set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`

7. **Fallback to CPU as last resort**:
   ```bash
   python validation/parameter_sweep_validation.py --force_cpu
   ```

## Interpreting Results

The sweep results are visualized through heatmaps in this directory:

- **combined_score_heatmap.png**: Overall performance across all metrics
- **h0_correlation_heatmap.png**: Correlation with H₀ measurements
- **sne_r_squared_heatmap.png**: Goodness of fit for supernovae data
- **bao_high_z_effect_size_heatmap.png**: Detection strength for BAO signals

A detailed summary is available in **parameter_sweep_summary.md**, which includes:
- Best parameter combination found
- Performance metrics for each dataset
- Comparison with theoretical expectations
- Recommendations for further investigation

## How MCMC Parameter Sweep Works

The parameter sweep uses Markov Chain Monte Carlo (MCMC) to efficiently search the parameter space:

1. Multiple "walkers" explore the parameter space in parallel
2. Each step evaluates the model against astronomical data
3. Walkers favor regions with better performance (higher likelihood)
4. After convergence, the posterior distribution reveals optimal parameter regions

This approach is more efficient than grid searches for high-dimensional parameter spaces.

## Using These Results

The parameter combinations identified in these sweeps can be used to:

1. Configure Genesis-Sphere model simulations
2. Generate visualizations with optimized parameters
3. Compare with theoretical predictions from other models
4. Make testable predictions for future astronomical observations

For any questions, please refer to the main documentation or contact the project maintainers.

## Folder Structure

The parameter sweep validation organizes its outputs in the following structure:
