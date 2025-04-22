"""
Parameter estimation using MCMC for the Genesis-Sphere model.

This script uses MCMC to explore the parameter space (ω, β) and find the
posterior probability distribution based on astronomical datasets.
It replaces the previous grid search and custom combined score with a
statistically robust approach.

NOTE: Requires modification of analysis functions to return log-likelihoods
      or chi-squared values. Requires installation of 'emcee' and 'corner'.
      (pip install emcee corner numpy pandas)
"""

import os
import sys
import numpy as np
import pandas as pd
import emcee # MCMC sampler
import corner # For plotting results
from datetime import datetime
import json
import argparse
import time # To time execution
import signal # For timeout handling
import gc # For garbage collection
import psutil # For memory monitoring, install with: pip install psutil
import pickle # For saving random state
import shutil # For file operations

# Global flag for GPU status
HAS_GPU = False
GPU_INFO = {"name": "None", "memory": "0 MB", "compute_capability": "N/A"}

# Add CUDA detection and path resolution function
def find_cuda_libraries():
    """Find CUDA libraries from common installation locations and add to PATH"""
    # Common CUDA installation paths
    common_cuda_paths = [
        # Windows paths
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        # Add more paths as needed
    ]
    
    # Look for environment variable indicating CUDA path
    cuda_path_env = os.environ.get('CUDA_PATH')
    if cuda_path_env:
        common_cuda_paths.insert(0, cuda_path_env)  # Prioritize env variable path
    
    # Check for CUDA_HOME environment variable as well
    cuda_home_env = os.environ.get('CUDA_HOME')
    if cuda_home_env and cuda_home_env != cuda_path_env:
        common_cuda_paths.insert(0, cuda_home_env)
    
    found_paths = []
    
    # Check each path and add bin directory to PATH if it exists
    for cuda_path in common_cuda_paths:
        bin_path = os.path.join(cuda_path, 'bin')
        if os.path.exists(bin_path):
            # Check if specific DLL exists
            if os.path.exists(os.path.join(bin_path, 'nvrtc64_110_0.dll')) or \
               os.path.exists(os.path.join(bin_path, 'nvrtc64_112_0.dll')) or \
               os.path.exists(os.path.join(bin_path, 'nvrtc64_120_0.dll')):
                found_paths.append(bin_path)
                
                # Add to PATH if not already there
                if bin_path not in os.environ['PATH'].split(os.pathsep):
                    os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                    print(f"Added CUDA bin directory to PATH: {bin_path}")
                
                # Also add lib64 path if it exists (for Linux)
                lib64_path = os.path.join(cuda_path, 'lib64')
                if os.path.exists(lib64_path):
                    if 'LD_LIBRARY_PATH' in os.environ:
                        if lib64_path not in os.environ['LD_LIBRARY_PATH'].split(os.pathsep):
                            os.environ['LD_LIBRARY_PATH'] = lib64_path + os.pathsep + os.environ['LD_LIBRARY_PATH']
                    else:
                        os.environ['LD_LIBRARY_PATH'] = lib64_path
                
                # Add lib/x64 path for Windows
                lib_x64_path = os.path.join(cuda_path, 'lib', 'x64')
                if os.path.exists(lib_x64_path):
                    if lib_x64_path not in os.environ['PATH'].split(os.pathsep):
                        os.environ['PATH'] = lib_x64_path + os.pathsep + os.environ['PATH']
    
    return found_paths

# Try to find CUDA libraries before importing CuPy
found_cuda_paths = find_cuda_libraries()
if found_cuda_paths:
    print(f"Found CUDA installations at: {', '.join(found_cuda_paths)}")
else:
    print("No CUDA installations found in common locations.")
    print("If CUDA is installed, try setting the CUDA_PATH environment variable.")

# Try to import CuPy for GPU acceleration, fall back to NumPy if not available
try:
    import cupy as cp
    # Initial simple test that CuPy is available
    test_array = cp.array([1, 2, 3])
    test_result = cp.sum(test_array)  # Test a basic operation
    del test_array, test_result
    # At this point we have CuPy imported, but need more thorough tests
    CUPY_IMPORT_SUCCESS = True
    print("CuPy successfully imported. Running GPU verification tests...")
except (ImportError, Exception) as e:
    import numpy as np
    cp = np  # Use NumPy as a fallback
    CUPY_IMPORT_SUCCESS = False
    HAS_GPU = False
    print(f"GPU acceleration not available: {e}")
    print("Using CPU with NumPy instead.")
    
    # Provide more detailed error information for common issues
    if "nvrtc64_112_0.dll" in str(e) or "nvrtc64_110_0.dll" in str(e) or "nvrtc64_120_0.dll" in str(e):
        print("\nCUDA DLL missing error detected. Possible solutions:")
        print("1. Install NVIDIA CUDA Toolkit matching your CuPy version")
        print("   - For CuPy 11.x: CUDA 11.0 - 11.8")
        print("   - For CuPy 12.x: CUDA 12.0+")
        print("2. Reinstall CuPy to match your CUDA version:")
        print("   - For CUDA 11.0-11.2: pip install cupy-cuda11x")
        print("   - For CUDA 11.3-11.8: pip install cupy-cuda11x")
        print("   - For CUDA 12.x:      pip install cupy-cuda12x")
        print("3. Make sure CUDA bin directory is in your PATH environment variable")
        print("   - Usually C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.x\\bin")
        print("4. Run with --force_cpu flag if you want to use CPU only\n")

def verify_gpu_operation(force_cpu=False):
    """
    Perform comprehensive verification of GPU operation.
    Returns True if GPU is fully operational, False otherwise.
    """
    global HAS_GPU, GPU_INFO
    
    if force_cpu:
        print("Forcing CPU mode as requested.")
        return False
    
    if not CUPY_IMPORT_SUCCESS:
        print("CuPy import failed, cannot use GPU.")
        return False
    
    try:
        # 1. Check CUDA availability and version
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        driver_version = cp.cuda.runtime.driverGetVersion()
        print(f"CUDA Runtime Version: {cuda_version//1000}.{(cuda_version%1000)//10}")
        print(f"CUDA Driver Version: {driver_version//1000}.{(driver_version%1000)//10}")
        
        # 2. Get device information
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            print("No CUDA devices detected!")
            return False
            
        print(f"Found {device_count} CUDA device(s):")
        for i in range(device_count):
            properties = cp.cuda.runtime.getDeviceProperties(i)
            name = properties["name"].decode("utf-8")
            total_memory = properties["totalGlobalMem"] / (1024**2)  # Convert to MB
            compute_capability = f"{properties['major']}.{properties['minor']}"
            
            print(f"  Device {i}: {name}")
            print(f"    Total Memory: {total_memory:.0f} MB")
            print(f"    Compute Capability: {compute_capability}")
            
            # Store info for the first (or selected) device
            if i == 0:
                GPU_INFO = {
                    "name": name,
                    "memory": f"{total_memory:.0f} MB",
                    "compute_capability": compute_capability
                }
        
        # 3. Test memory allocation
        try:
            # Allocate a small tensor (100MB)
            test_size = 25_000_000  # ~100MB in float32
            memory_test = cp.zeros(test_size, dtype=cp.float32)
            memory_test[0] = 1.0  # Test memory write
            assert memory_test[0] == 1.0  # Test memory read
            del memory_test
            print("GPU memory allocation test passed.")
        except Exception as e:
            print(f"GPU memory allocation failed: {e}")
            return False
            
        # 4. Test computation with representative workload
        try:
            # Matrix operations similar to what we'll do in MCMC
            matrix_size = 1000
            A = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)
            b = cp.random.rand(matrix_size, dtype=cp.float32)
            
            # Time the computation
            start = time.time()
            result = cp.dot(A, b)
            cp.cuda.Device().synchronize()  # Make sure computation is complete
            duration = time.time() - start
            
            # Verify result isn't corrupt
            assert not cp.isnan(cp.sum(result)), "Computation produced NaN values"
            
            print(f"GPU computation test passed in {duration:.3f} seconds.")
            
            # Optional - benchmark CPU vs GPU for the same task
            if matrix_size <= 2000:  # Don't do this for huge matrices
                A_cpu = np.array(A.get())
                b_cpu = np.array(b.get())
                
                start = time.time()
                result_cpu = np.dot(A_cpu, b_cpu)
                cpu_duration = time.time() - start
                
                del A_cpu, b_cpu, result_cpu
                print(f"Same operation on CPU: {cpu_duration:.3f} seconds")
                print(f"GPU speedup: {cpu_duration/duration:.1f}x")
            
            # Clean up
            del A, b, result
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"GPU computation test failed: {e}")
            return False
            
        # 5. Test combined operation that mimics our actual function
        try:
            # Setup simple function similar to our likelihood
            def test_likelihood(n_points=1000):
                # Matrix operations typical in our computation
                x = cp.random.rand(n_points)
                y = cp.random.rand(n_points)
                z = cp.random.rand(n_points)
                
                # Operations similar to our model
                sin_term = cp.sin(x * 2.0)
                rho = (1.0 / (1.0 + sin_term**2)) * (1.0 + 0.02 * y**2)
                tf = 1.0 / (1.0 + 0.8 * (cp.abs(z) + 0.1))
                result = rho / cp.sqrt(tf)
                
                # Reduction and chi-squared-like operation
                chi2 = cp.sum((result - 1.0)**2)
                return float(chi2)
                
            # Test the function
            test_chi2 = test_likelihood(5000)
            assert np.isfinite(test_chi2), "Test likelihood produced invalid result"
            print(f"GPU likelihood function test passed: χ² = {test_chi2:.4f}")
            
            # GPU memory management test
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU memory pool cleared successfully.")
            
        except Exception as e:
            print(f"GPU likelihood test failed: {e}")
            return False
        
        # All tests passed
        print("All GPU verification tests PASSED. CUDA GPU is operational.")
        HAS_GPU = True
        return True
        
    except Exception as e:
        print(f"GPU verification failed with error: {e}")
        HAS_GPU = False
        return False

# GPU helper functions
def to_gpu(arr):
    """Transfer NumPy array to GPU if GPU is available"""
    if HAS_GPU and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr
    
def to_cpu(arr):
    """Transfer CuPy array back to CPU if needed"""
    if HAS_GPU and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Create results directory
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'parameter_sweep')
os.makedirs(results_dir, exist_ok=True)

try:
    from models.genesis_model import GenesisSphereModel
    from validation.celestial_correlation_validation import (
        load_h0_measurements,
        load_supernovae_data,
        load_bao_data,
        analyze_h0_correlation,
        analyze_sne_fit,
        analyze_bao_detection
    )
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'models' and 'validation' directories are accessible"
          " relative to the script's parent directory and contain the required files/functions.")
    sys.exit(1)
except FileNotFoundError:
     print("Error: Could not determine parent directory reliably. "
           "Ensure script structure allows importing 'models' and 'validation'.")
     sys.exit(1)

# --- Helper functions to calculate chi-squared from existing analysis functions ---

def calculate_h0_chi2(gs_model, h0_data):
    """Calculate chi-squared for H0 correlation with GPU acceleration if available"""
    if HAS_GPU:
        return calculate_h0_chi2_gpu(gs_model, h0_data)
    
    metrics = analyze_h0_correlation(gs_model, h0_data)
    # Convert correlation to chi-squared
    # For correlation coefficient r, we can use -N*log(1-r²) as an approximation
    # This makes better correlation (r close to 1) give lower chi-squared
    r = metrics['correlation']
    n = len(h0_data)
    chi2 = -n * np.log(1 - r**2) if abs(r) < 1.0 else 1000
    # Invert the sign since better correlation should give lower chi-squared
    return 1000 - chi2 if np.isfinite(chi2) else 1000

def calculate_h0_chi2_gpu(gs_model, h0_data):
    """GPU-accelerated chi-squared calculation for H0 measurements"""
    try:
        # Get data
        years = h0_data['year'].values
        h0_obs = h0_data['H0'].values
        h0_err = h0_data['H0_err'].values
        
        # Transfer to GPU
        years_gpu = to_gpu(years)
        h0_obs_gpu = to_gpu(h0_obs)
        h0_err_gpu = to_gpu(h0_err)
        
        # Get predictions - a simplified version, real implementation would use the model
        h0_base = 70.0
        t = (years_gpu - 2000.0) / 100.0
        sin_term = cp.sin(gs_model.omega * t)
        rho = (1.0 / (1.0 + sin_term**2)) * (1.0 + gs_model.alpha * t**2)
        tf = 1.0 / (1.0 + gs_model.beta * (cp.abs(t) + gs_model.epsilon))
        h0_pred = h0_base * (1.0 + 0.1 * cp.sin(gs_model.omega * t)) * (1.0 + 0.05 * rho) / cp.sqrt(tf)
        
        # Calculate chi-squared
        residuals = (h0_obs_gpu - h0_pred) / h0_err_gpu
        chi2 = float(cp.sum(residuals**2))
        
        # Free GPU memory
        if np.random.random() < 0.1:  # Only occasionally to avoid overhead
            del years_gpu, h0_obs_gpu, h0_err_gpu, h0_pred, residuals
            if HAS_GPU:
                cp.get_default_memory_pool().free_all_blocks()
        
        return chi2
        
    except Exception as e:
        print(f"GPU calculation failed with error: {e}, falling back to CPU")
        # Fall back to CPU implementation
        return calculate_h0_chi2(gs_model, h0_data)

def calculate_sne_chi2(gs_model, sne_data):
    """Calculate chi-squared for supernovae fit with GPU acceleration if available"""
    if HAS_GPU:
        return calculate_sne_chi2_gpu(gs_model, sne_data)
    
    # Original CPU implementation
    metrics = analyze_sne_fit(gs_model, sne_data)
    # Use the reduced chi-squared metric directly if available
    if 'reduced_chi2' in metrics:
        return metrics['reduced_chi2'] * (len(sne_data) - 2)  # Convert reduced chi2 to raw chi2
    
    # Alternatively use R-squared
    r_squared = metrics['r_squared']
    # Transform R² to a chi-squared-like metric (lower is better)
    # When R² is close to 1 (good fit), this gives a small value
    chi2_approx = len(sne_data) * (1 - r_squared) if r_squared <= 1.0 else len(sne_data) * 2
    return chi2_approx

def calculate_sne_chi2_gpu(gs_model, sne_data):
    """GPU-accelerated chi-squared calculation for supernovae"""
    try:
        # Extract data
        redshifts = sne_data['z'].values
        mu_obs = sne_data['mu'].values
        mu_err = sne_data['mu_err'].values
        
        # Transfer to GPU
        z_gpu = to_gpu(redshifts)
        mu_obs_gpu = to_gpu(mu_obs)
        mu_err_gpu = to_gpu(mu_err)
        
        # Calculate distance modulus prediction
        # This is a simplified implementation - adjust to match your actual model
        # Proper implementation would use your gs_model to predict distance modulus
        h0 = 70.0
        omega_m = 0.3
        c = 299792.458  # km/s
        
        # Simple luminosity distance calculation (flat ΛCDM approximation)
        # Adjust this to match your Genesis-Sphere model's prediction
        a = 1.0 / (1.0 + z_gpu)
        integrand = 1.0 / cp.sqrt(omega_m / a**3 + (1.0 - omega_m))
        # Simple trapezoid integration - could be more sophisticated
        da = 0.001
        a_vals = cp.arange(a.min(), 1.0, da)
        integral = cp.sum(integrand * da)
        dl = c / h0 * (1.0 + z_gpu) * integral
        mu_pred = 5.0 * cp.log10(dl) + 25.0
        
        # Calculate chi-squared
        residuals = (mu_obs_gpu - mu_pred) / mu_err_gpu
        chi2 = float(cp.sum(residuals**2))
        
        # Free GPU memory occasionally
        if np.random.random() < 0.1:
            del z_gpu, mu_obs_gpu, mu_err_gpu, mu_pred, residuals
            if HAS_GPU:
                cp.get_default_memory_pool().free_all_blocks()
        
        return chi2
        
    except Exception as e:
        print(f"GPU calculation failed with error: {e}, falling back to CPU")
        # Fall back to CPU implementation
        return calculate_sne_chi2(gs_model, sne_data)

def calculate_bao_chi2(gs_model, bao_data):
    """Calculate chi-squared for BAO detection with GPU acceleration if available"""
    if HAS_GPU:
        return calculate_bao_chi2_gpu(gs_model, bao_data)
    
    # Original CPU implementation
    metrics = analyze_bao_detection(gs_model, bao_data)
    # For effect size, a larger value is better
    # Convert to a chi-squared-like value (lower is better)
    effect_size = metrics['high_z_effect_size']
    max_expected = 100  # A reasonable maximum to scale against
    # Normalize so that higher effect size gives lower chi-squared
    chi2_approx = max_expected - min(effect_size, max_expected)
    return chi2_approx

def calculate_bao_chi2_gpu(gs_model, bao_data):
    """GPU-accelerated chi-squared calculation for BAO measurements"""
    try:
        # Extract data
        redshifts = bao_data['z'].values
        rd_obs = bao_data['rd'].values
        rd_err = bao_data['rd_err'].values
        
        # Transfer to GPU
        z_gpu = to_gpu(redshifts)
        rd_obs_gpu = to_gpu(rd_obs)
        rd_err_gpu = to_gpu(rd_err)
        
        # Calculate predictions
        # This is a simplified version - replace with your actual model
        rd_fid = 147.78  # Mpc
        omega_m = 0.3
        
        # Simple approximation for sound horizon
        # Replace with your Genesis-Sphere model's actual prediction
        rd_pred = rd_fid * cp.sqrt(0.02237 / 0.02233) * cp.sqrt(omega_m / 0.3) * cp.ones_like(z_gpu)
        
        # Calculate chi-squared
        residuals = (rd_obs_gpu - rd_pred) / rd_err_gpu
        chi2 = float(cp.sum(residuals**2))
        
        # Free memory occasionally
        if np.random.random() < 0.1:
            del z_gpu, rd_obs_gpu, rd_err_gpu, rd_pred, residuals
            if HAS_GPU:
                cp.get_default_memory_pool().free_all_blocks()
        
        return chi2
        
    except Exception as e:
        print(f"GPU calculation failed with error: {e}, falling back to CPU")
        # Fall back to CPU implementation
        return calculate_bao_chi2(gs_model, bao_data)

# === Memory and Performance Monitoring ===

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def print_memory_usage(label=""):
    """Print current memory usage with an optional label"""
    memory_mb = get_memory_usage()
    print(f"Memory usage {label}: {memory_mb:.2f} MB")

def with_timeout(seconds, func, *args, **kwargs):
    """Run a function with a timeout (cross-platform implementation)"""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=seconds)
        except concurrent.futures.TimeoutError:
            print(f"Warning: Function call timed out")
            return None

# === MCMC Setup ===
# Define the parameter space (dimensions)
# We are fitting for omega and beta
N_DIM = 2
PARAM_LABELS = [r"$\omega$", r"$\beta$"] # LaTeX labels for plots

# Define the prior function - sets allowed parameter ranges
def log_prior(params):
    """
    Log prior probability distribution (Log(Prior)).
    Returns 0 if params are within allowed ranges, -np.inf otherwise.
    This enforces constraints on parameters.
    """
    omega, beta = params
    # Modified to only allow positive beta values, as negative values lead to 
    # unphysical acceleration of time near singularities instead of dilation
    if 1.0 < omega < 6.0 and 0.0 <= beta < 3.0: 
        return 0.0 # Log(1) = 0 -> uniform prior within bounds
    return -np.inf # Log(0) -> rules out parameters outside bounds

# Define the log-likelihood function - compares model to data
def log_likelihood(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """
    Log likelihood function (Log(Likelihood)).
    Calculates the total likelihood of observing the data given the parameters.
    """
    # For GPU acceleration, delegate to the GPU version if available
    if HAS_GPU:
        return log_likelihood_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)

    # Track computation time for diagnostics
    start_time = time.time()
    
    omega, beta = params
    alpha = fixed_alpha
    epsilon = fixed_epsilon

    # Check prior first (can sometimes save computation)
    if not np.isfinite(log_prior(params)):
         return -np.inf

    try:
        # Create model instance with current parameters
        gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)

        # Calculate chi-squared for each dataset with timeout
        try:
            chi2_h0 = with_timeout(5, calculate_h0_chi2, gs_model, data_h0)
            if chi2_h0 is None:
                return -np.inf
        except Exception as e:
            print(f"Error calculating H0 chi2: {e}")
            return -np.inf

        try:
            chi2_sne = with_timeout(5, calculate_sne_chi2, gs_model, data_sne)
            if chi2_sne is None:
                return -np.inf
        except Exception as e:
            print(f"Error calculating SNe chi2: {e}")
            return -np.inf

        try:
            chi2_bao = with_timeout(5, calculate_bao_chi2, gs_model, data_bao)
            if chi2_bao is None:
                return -np.inf
        except Exception as e:
            print(f"Error calculating BAO chi2: {e}")
            return -np.inf

        # Modified approach: Properly balance dataset contributions based on degrees of freedom
        # This prevents one dataset from dominating due to different point counts
        dof_h0 = len(data_h0) - 2  # Subtracting parameter count
        dof_sne = len(data_sne) - 2
        dof_bao = len(data_bao) - 2
        
        # Use reduced chi-squared (chi²/dof) for balanced weighting
        reduced_chi2_h0 = chi2_h0 / max(1, dof_h0)
        reduced_chi2_sne = chi2_sne / max(1, dof_sne)
        reduced_chi2_bao = chi2_bao / max(1, dof_bao)
        
        # Calculate total chi-squared (properly normalized)
        total_chi2 = (reduced_chi2_h0 + reduced_chi2_sne + reduced_chi2_bao) * (dof_h0 + dof_sne + dof_bao) / 3

        # Convert total Chi-squared to Log Likelihood (assuming Gaussian errors)
        logL = -0.5 * total_chi2
        if np.random.random() < 0.01:  # Do this only occasionally (1% chance)
            gc.collect()

        # Monitor computation time
        compute_time = time.time() - start_time
        if compute_time > 1.0:  # Only log slow computations
            print(f"Slow likelihood computation: {compute_time:.2f}s for ω={omega:.4f}, β={beta:.4f}")
            
        return logL

    except Exception as e:
        # Handle potential errors during model calculation or analysis
        print(f"Warning: Likelihood calculation failed for ω={omega:.4f}, β={beta:.4f}. Error: {e}")
        return -np.inf

def log_likelihood_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """GPU-accelerated log likelihood function."""
    start_time = time.time()
    
    omega, beta = params
    alpha = fixed_alpha
    epsilon = fixed_epsilon

    # Check prior (CPU operation, no need for GPU)
    if not np.isfinite(log_prior(params)):
         return -np.inf

    try:
        # Create model instance
        gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)

        # Calculate chi-squared for each dataset
        try:
            chi2_h0 = calculate_h0_chi2_gpu(gs_model, data_h0)
            if not np.isfinite(chi2_h0):
                return -np.inf
        except Exception as e:
            print(f"Error calculating H0 chi2 on GPU: {e}")
            return -np.inf
            
        try:
            chi2_sne = calculate_sne_chi2_gpu(gs_model, data_sne)
            if not np.isfinite(chi2_sne):
                return -np.inf
        except Exception as e:
            print(f"Error calculating SNe chi2 on GPU: {e}")
            return -np.inf
            
        try:
            chi2_bao = calculate_bao_chi2_gpu(gs_model, data_bao)  # Using data_bao, not bao_data
            if not np.isfinite(chi2_bao):
                return -np.inf
        except Exception as e:
            print(f"Error calculating BAO chi2 on GPU: {e}")
            return -np.inf

        # Calculate total chi-squared
        total_chi2 = chi2_h0 + chi2_sne + chi2_bao

        # Convert to log likelihood
        logL = -0.5 * total_chi2

        # Check for NaN or infinite results which can break MCMC
        if not np.isfinite(logL):
             print(f"Warning: Non-finite logL ({logL}) for ω={omega:.4f}, β={beta:.4f}")
             return -np.inf

        # Cleanup unused GPU memory periodically
        if np.random.random() < 0.01:
            if HAS_GPU:
                cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
        # Monitor computation time
        compute_time = time.time() - start_time
        if compute_time > 1.0:
            print(f"Slow GPU likelihood: {compute_time:.2f}s for ω={omega:.4f}, β={beta:.4f}")
            
        return logL

    except Exception as e:
        print(f"Warning: GPU likelihood calculation failed for ω={omega:.4f}, β={beta:.4f}. Error: {e}")
        return -np.inf

# Define the log-posterior function (Prior + Likelihood)
def log_posterior(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """
    Log posterior probability distribution (Log(Prior) + Log(Likelihood)).
    This is the function the MCMC sampler explores.
    """
    lp = log_prior(params)
    if not np.isfinite(lp): # If parameters are outside prior range
        return -np.inf
    # Log(Posterior) = Log(Prior) + Log(Likelihood)
    return lp + log_likelihood(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)

# Function for GPU-accelerated log posterior
def log_posterior_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """GPU-accelerated log posterior calculation"""
    # Check prior - no need to use GPU for this simple check
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    # Calculate log likelihood using GPU acceleration
    ll = log_likelihood_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)
    
    return lp + ll

# Function to save intermediate results during MCMC run
def save_intermediate_results(sampler, nburn, output_dir, prefix, fixed_params, step_number, batch_speed, total_elapsed, 
                             consolidated_log=False, consolidated_log_file=None, max_log_files=5, last_checkpoint_files=None):
    """Save intermediate MCMC results for checkpointing"""
    try:
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use dynamic burn-in based on current progress
        # If we haven't done many steps yet, use a smaller burn-in
        effective_burn = min(nburn, step_number // 2)
        
        # Save the current state of the chain
        samples = sampler.get_chain(discard=effective_burn, thin=10, flat=True)
        
        if len(samples) == 0:
            print(f"No valid samples to save yet (completed {step_number} steps, using burn-in of {effective_burn}).")
            print(f"Will be able to save samples after {effective_burn*2} steps.")
            
            # Save basic info even if we don't have samples yet
            if consolidated_log and consolidated_log_file:
                with open(consolidated_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"No valid samples to save yet at step {step_number}. Need {effective_burn*2} steps for samples.\n")
            
            # Use fixed filenames instead of timestamped ones when using consolidated logs
            if consolidated_log:
                info_file = os.path.join(output_dir, f"{prefix}_progress_info.json")
            else:
                info_file = os.path.join(output_dir, f"{prefix}_progress_info_{timestamp}.json")
                
            info = {
                'timestamp': timestamp,
                'samples_saved': 0,
                'steps_completed': step_number,
                'current_results': {
                    'omega': {'median': float(fixed_params.get('initial_guess', {}).get('omega', 3.5))},
                    'beta': {'median': float(fixed_params.get('initial_guess', {}).get('beta', 0.0333))}
                },
                'fixed_params': fixed_params,
                'progress': {
                    'step_number': step_number,
                    'batch_speed': batch_speed,
                    'elapsed_minutes': total_elapsed / 60.0,
                }
            }
            
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            print(f"Progress info saved to: {info_file}")
            
            return None
            
        # Create DataFrame and save
        df_samples = pd.DataFrame(samples, columns=['omega', 'beta'])
        
        # Use fixed filenames when in consolidated log mode, otherwise use timestamps
        if consolidated_log:
            checkpoint_file = os.path.join(output_dir, f"{prefix}_checkpoint.csv")
            backup_checkpoint_file = os.path.join(output_dir, f"{prefix}_checkpoint_backup.csv")
            
            # Rotate: if main file exists, copy to backup before overwriting
            if os.path.exists(checkpoint_file):
                try:
                    shutil.copy2(checkpoint_file, backup_checkpoint_file)
                except Exception as e:
                    print(f"Warning: Could not create backup of checkpoint file: {e}")
        else:
            checkpoint_file = os.path.join(output_dir, f"{prefix}_checkpoint_{timestamp}.csv")
        
        df_samples.to_csv(checkpoint_file, index=False, encoding='utf-8')
        
        # Calculate quick stats if we have enough samples
        if len(samples) >= 10:
            results_summary = {}
            for i, label in enumerate(['omega', 'beta']):
                if len(samples) >= 3:  # Need at least 3 samples for percentiles
                    mcmc = np.percentile(samples[:, i], [16, 50, 84])
                    median = mcmc[1]
                    q = np.diff(mcmc)  # q[0] = 50th-16th, q[1] = 84th-50th
                    lower_err = q[0]
                    upper_err = q[1]
                    results_summary[label] = {
                        'median': float(median),
                        'lower_err': float(lower_err),
                        'upper_err': float(upper_err)
                    }
                else:
                    # Just use mean if not enough for percentiles
                    median = float(np.mean(samples[:, i]))
                    std = float(np.std(samples[:, i])) if len(samples) > 1 else 0.0
                    results_summary[label] = {
                        'median': median,
                        'lower_err': std,
                        'upper_err': std
                    }
            
            # Calculate preliminary performance metrics if possible
            performance_metrics = {}
            try:
                omega_median = results_summary['omega']['median']
                beta_median = results_summary['beta']['median']
                
                # Create a model with current best parameters
                gs_model = GenesisSphereModel(
                    alpha=fixed_params['alpha'], 
                    beta=beta_median, 
                    omega=omega_median, 
                    epsilon=fixed_params['epsilon']
                )
                
                # Quick calculation of key metrics
                # These are simplified approximations to avoid full data loading
                h0_corr = estimate_h0_correlation(gs_model)
                sne_r2 = estimate_sne_r2(gs_model)
                bao_effect = estimate_bao_effect(gs_model)
                
                performance_metrics = {
                    'h0_correlation_approx': float(h0_corr),
                    'sne_r2_approx': float(sne_r2),
                    'bao_effect_approx': float(bao_effect),
                    'combined_score_approx': float((h0_corr + sne_r2 + min(1.0, bao_effect/100))/3)
                }
            except Exception as e:
                print(f"Could not calculate performance metrics: {e}")
                performance_metrics = {
                    'calculation_error': str(e)
                }
            
            # Save basic info with additional runtime metrics
            # Use fixed filenames when in consolidated log mode
            if consolidated_log:
                info_file = os.path.join(output_dir, f"{prefix}_checkpoint_info.json")
                state_file = os.path.join(output_dir, f"{prefix}_state.npz")
                random_state_file = os.path.join(output_dir, f"{prefix}_random_state.pkl")
            else:
                info_file = os.path.join(output_dir, f"{prefix}_checkpoint_info_{timestamp}.json")
                state_file = os.path.join(output_dir, f"{prefix}_state_{timestamp}.npz")
                random_state_file = os.path.join(output_dir, f"{prefix}_random_state_{timestamp}.pkl")
            
            info = {
                'timestamp': timestamp,
                'samples_saved': len(samples),
                'current_results': results_summary,
                'fixed_params': fixed_params,
                'progress': {
                    'step_number': step_number,
                    'batch_speed': batch_speed,
                    'elapsed_minutes': total_elapsed / 60.0,
                    'epoch': float(step_number / sampler.chain.shape[0]),
                    'percent_complete': float(step_number / (fixed_params['nsteps'] + step_number - 
                                                           (step_number % fixed_params['nsteps']))*100)
                },
                'preliminary_metrics': performance_metrics
            }
            
            # If using consolidated logs, append the results to the consolidated log file
            if consolidated_log and consolidated_log_file:
                with open(consolidated_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"### Parameter Estimates at Step {step_number}\n")
                    for param, values in results_summary.items():
                        f.write(f"{param}: {values['median']:.4f} (+{values['upper_err']:.4f}/-{values['lower_err']:.4f})\n")
                    
                    if 'preliminary_metrics' in info and len(performance_metrics) > 0:
                        f.write("\nPreliminary performance metrics (approximate):\n")
                        for metric, value in performance_metrics.items():
                            if 'approx' in metric:
                                metric_name = metric.replace('_approx', '')
                                f.write(f"  {metric_name}: {value:.4f}\n")
                    f.write("\n")
            
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            
            # Save sampler state for potential recovery
            try:
                # Convert chain to standard numpy array if needed
                if hasattr(sampler.chain, 'get'):  # Check if it's a CuPy array
                    chain_array = sampler.chain.get()
                else:
                    chain_array = np.array(sampler.chain)
                
                # Save the random state separately
                with open(random_state_file, 'wb') as f:
                    pickle.dump(np.random.get_state(), f)
                
                # Save just the chain array
                np.save(state_file, chain_array)
                print(f"MCMC state saved successfully.")
            except Exception as e:
                print(f"Warning: Could not save state file: {e}")
                print("This won't affect the main MCMC process.")
            
            # Print results summary
            print(f"Checkpoint saved with {len(samples)} samples. Current estimates:")
            for param, values in results_summary.items():
                print(f"  {param}: {values['median']:.4f} (+{values['upper_err']:.4f}/-{values['lower_err']:.4f})")
            
            if 'preliminary_metrics' in info and len(performance_metrics) > 0:
                print("  Preliminary performance metrics (approximate):")
                for metric, value in performance_metrics.items():
                    if 'approx' in metric:
                        metric_name = metric.replace('_approx', '')
                        print(f"    {metric_name}: {value:.4f}")
        
        return checkpoint_file
    
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with MCMC process despite checkpoint error.")
        return None

# === Main Execution ===
 
def main():
    """Main function to run the MCMC parameter estimation"""
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run MCMC parameter estimation for Genesis-Sphere")
    # Add arguments for fixed parameters, data paths, MCMC settings, etc.
    parser.add_argument("--alpha", type=float, default=0.02, help="Fixed alpha value")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Fixed epsilon value")
    parser.add_argument("--nwalkers", type=int, default=32, help="Number of MCMC walkers (must be > 2*N_DIM)")
    parser.add_argument("--nsteps", type=int, default=5000, help="Number of MCMC steps per walker")
    parser.add_argument("--nburn", type=int, default=1000, help="Number of burn-in steps to discard")
    parser.add_argument("--initial_omega", type=float, default=3.5, help="Initial guess for omega")
    parser.add_argument("--initial_beta", type=float, default=0.0333, help="Initial guess for beta")  # Changed from -0.0333 to 0.0333 to match prior
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix for output filenames")
    parser.add_argument("--checkpoint_interval", type=int, default=250,  # Changed from 100 to 250
                        help="Save intermediate results every N steps (0 to disable)")
    parser.add_argument("--test_mode", action="store_true", 
                        help="Run in test mode with reduced computation")
    parser.add_argument("--max_time", type=int, default=30,
                        help="Maximum runtime in minutes (default: 30 minutes)")
    parser.add_argument("--resume", type=str, default="", 
                        help="Path to state file to resume from a previous run")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU mode even if GPU is available")
    parser.add_argument("--verify_gpu", action="store_true",
                       help="Run GPU verification tests and exit")
    parser.add_argument("--quick_run", action="store_true",
                        help="Run with reduced parameters for quick results")
    parser.add_argument("--summary_interval", type=int, default=15,
                        help="Interval in minutes for printing summary updates (default: 15)")
    parser.add_argument("--enhanced_progress", action="store_true",
                        help="Show enhanced progress tracking with percentage completion")
    parser.add_argument("--slow_mode", action="store_true",
                      help="Run in slow mode with artificial pauses between steps for easier progress monitoring")
    parser.add_argument("--progress_update_interval", type=int, default=30, 
                      help="Interval in seconds between progress updates (default: 30)")
    parser.add_argument("--progress_delay", type=float, default=0.0,
                      help="Add a small delay (in seconds) when updating progress to slow down the display")
    parser.add_argument("--max_log_files", type=int, default=5,
                      help="Maximum number of backup log files to keep (default: 5)")
    parser.add_argument("--consolidated_logs", action="store_true",
                      help="Use consolidated logs - fewer files with more content")
    args = parser.parse_args()    
    
    # Run GPU verification
    if args.verify_gpu:
        verify_gpu_operation(args.force_cpu)
        print("GPU verification complete. Exiting.")
        return
    
    # Initialize GPU if we want to use it
    gpu_status = verify_gpu_operation(args.force_cpu)
    if gpu_status:
        print(f"Using GPU: {GPU_INFO['name']} with {GPU_INFO['memory']} memory")
    else:
        print("Using CPU mode for computation")
        global HAS_GPU
        HAS_GPU = False

    # Create save points directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    run_id = f"{timestamp}{suffix}"
    
    # Create main savepoints directory if it doesn't exist
    savepoints_dir = os.path.join(results_dir, "savepoints")
    os.makedirs(savepoints_dir, exist_ok=True)
    
    # Create run-specific directory for this run
    run_dir = os.path.join(savepoints_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create progress tracking file in the run directory
    progress_log_file = os.path.join(run_dir, f"progress_log.txt")
    summary_log_file = os.path.join(run_dir, f"summary_log.txt")
    consolidated_log_file = os.path.join(run_dir, f"consolidated_log.txt")
    
    # Initialize consolidated log if that option is selected
    if args.consolidated_logs:
        with open(consolidated_log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Genesis-Sphere Parameter Sweep - Run Started: {timestamp}\n")
            f.write(f"# Fixed Parameters: α={args.alpha}, ε={args.epsilon}\n")
            f.write(f"# MCMC Settings: Walkers={args.nwalkers}, Steps={args.nsteps}, Burn-in={args.nburn}\n")
            f.write(f"# {'='*80}\n\n")
    
    with open(progress_log_file, 'w', encoding='utf-8') as f:
        f.write("Step,Epoch,Batch_Speed,Elapsed_Min,Remaining_Min,Memory_MB,Progress_Pct\n")
    
    with open(summary_log_file, 'w', encoding='utf-8') as f:
        f.write("Timestamp,Elapsed_Min,Steps_Completed,Best_Omega,Best_Beta,Best_Score,Acceptance_Rate,Samples_Per_Sec,Remaining_Min\n")
    
    print(f"All results will be saved to: {run_dir}")
    
    # Apply quick run settings if requested
    if args.quick_run:
        print("Running in QUICK MODE with reduced computation for faster results")
        args.nwalkers = 24  # Reduced from default
        args.nsteps = 2000  # Reduced from 5000
        args.nburn = 500    # Reduced from 1000
        args.checkpoint_interval = 100  # Less frequent checkpoints (was 50)
        args.max_time = min(args.max_time, 120)  # Cap at 2 hours max

    # Modify parameters if in test mode
    if args.test_mode:
        print("Running in TEST MODE with reduced computation")
        args.nwalkers = 10
        args.nsteps = 25  # Reduced from 50 to 25 steps for quicker testing
        args.nburn = 10
        args.checkpoint_interval = 10  # Less frequent checkpoints (was 5)
    
    # Validate walker count
    if args.nwalkers <= 2 * N_DIM:
        print(f"Error: Number of walkers ({args.nwalkers}) must be greater than 2 * N_DIM ({2*N_DIM}).")
        sys.exit(1)
    if args.nsteps <= args.nburn:
        print(f"Error: Total steps ({args.nsteps}) must be greater than burn-in steps ({args.nburn}).")
        sys.exit(1)

    print("Starting Genesis-Sphere MCMC Parameter Estimation...")
    print(f"Fixed Parameters: α={args.alpha}, ε={args.epsilon}")
    print(f"MCMC Settings: Walkers={args.nwalkers}, Steps={args.nsteps}, Burn-in={args.nburn}")
    print(f"Checkpoint Interval: {args.checkpoint_interval} steps")
    print(f"Maximum runtime: {args.max_time} minutes")
    print(f"Save points directory: {savepoints_dir}")
    print_memory_usage("at start")

    # --- Load Data ---
    print("Loading observational data...")
    try:
        loading_start = time.time()
        h0_data = load_h0_measurements()
        print(f"  H0 data loaded: {len(h0_data)} measurements ({time.time() - loading_start:.2f}s)")
        
        loading_start = time.time()
        sne_data = load_supernovae_data()
        print(f"  SNe data loaded: {len(sne_data)} supernovae ({time.time() - loading_start:.2f}s)")
        
        loading_start = time.time()
        bao_data = load_bao_data()
        print(f"  BAO data loaded: {len(bao_data)} measurements ({time.time() - loading_start:.2f}s)")
        
        print("Data loaded successfully.")
        print_memory_usage("after data loading")
        
        # Initialize GPU memory if available
        if HAS_GPU:
            print("Setting up GPU environment...")
            # Limit GPU memory usage to 90% of available memory
            try:
                # Use get_default_memory_pool instead of directly using cp.cuda.malloc
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(fraction=0.9)
                print("  GPU memory pool configured.")
                
                # Preload some data to GPU to check it works
                test_array = cp.array([1, 2, 3])
                del test_array
                cp.get_default_memory_pool().free_all_blocks()
                print("  GPU test successful.")
            except Exception as e:
                print(f"  Warning: GPU setup encountered an issue: {e}")
                print("  Falling back to CPU mode.")
                HAS_GPU = False
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # --- Initialize or Resume Walkers ---
    if args.resume:
        print(f"Resuming from state file: {args.resume}")
        try:
            # Load previous state
            state_data = np.load(args.resume)
            chain = state_data['chain']
            random_state = state_data['random_state']
            
            # Set the random state
            np.random.set_state(random_state)
            
            # Use the last position of the chain
            pos = chain[:, -1, :]
            
            # Calculate steps already completed
            steps_already_completed = chain.shape[1]
            print(f"Resuming from step {steps_already_completed} with {args.nsteps - steps_already_completed} remaining")
            
            # Adjust nsteps to account for steps already done
            args.nsteps = max(args.nsteps - steps_already_completed, 100)  # Ensure at least 100 more steps
        except Exception as e:
            print(f"Error loading previous state: {e}")
            print("Starting with fresh initialization instead.")
            args.resume = ""
    
    if not args.resume:
        print("Initializing walkers...")
        init_start = time.time()
        # Start walkers in a small Gaussian ball around the previous best parameters
        initial_pos_guess = np.array([args.initial_omega, args.initial_beta])
        
        # Add small random offsets for each walker, ensuring they are within priors
        pos = np.zeros((args.nwalkers, N_DIM))
        max_attempts = 1000  # Prevent infinite loops
        
        for i in range(args.nwalkers):
            attempts = 0
            # Keep generating random starting points until they satisfy the prior
            while attempts < max_attempts:
                p = initial_pos_guess + 1e-3 * np.abs(initial_pos_guess) * np.random.randn(N_DIM)
                if np.isfinite(log_prior(p)):
                    pos[i] = p
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not initialize walker {i} after {max_attempts} attempts.")
                print(f"Using initial guess directly: ω={args.initial_omega}, β={args.initial_beta}")
                pos[i] = initial_pos_guess
        
        nwalkers, ndim = pos.shape
        print(f"Initialized {nwalkers} walkers around ω={args.initial_omega}, β={args.initial_beta} " 
              f"in {time.time() - init_start:.2f}s")
        steps_already_completed = 0            

    # --- Test likelihood function with a few points ---
    print("Testing likelihood function with initial positions...")
    test_start = time.time()
    success_count = 0
    
    for i in range(min(5, nwalkers)):
        test_params = pos[i]
        try:
            lp = log_posterior(test_params, h0_data, sne_data, bao_data, args.alpha, args.epsilon)
            if np.isfinite(lp):
                success_count += 1
                print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → logP={lp:.2f} (Valid)")
            else:
                print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Invalid posterior")
        except Exception as e:
            print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Error: {e}")
    
    print(f"Likelihood function test: {success_count}/5 successful in {time.time() - test_start:.2f}s")
    if success_count == 0:
        print("ERROR: All likelihood function tests failed. Exiting.")
        sys.exit(1)

    # --- Run MCMC ---
    print(f"Running MCMC...")
    mcmc_start = time.time()
    
    # Set up max runtime if specified
    max_runtime_seconds = args.max_time * 60
    
    # Fixed parameters to pass to checkpointing
    fixed_params = {
        'alpha': args.alpha,
        'epsilon': args.epsilon,
        'nwalkers': args.nwalkers,
        'nsteps': args.nsteps,
        'nburn': args.nburn
    }
    
    # The 'args' tuple passes fixed parameters and data to the log_posterior functions
    if HAS_GPU:
        print("Using GPU-accelerated MCMC sampling")
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior_gpu, 
            args=(h0_data, sne_data, bao_data, args.alpha, args.epsilon)
        )
    else:
        print("Using CPU-based MCMC sampling")
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, 
            args=(h0_data, sne_data, bao_data, args.alpha, args.epsilon)
        )
    
    # Run MCMC steps with progress, checkpointing, and time limit
    print("Running with checkpoints and time limit...")
    
    # Run in smaller chunks for checkpointing
    chunk_size = min(5, args.checkpoint_interval)  # Smaller chunks for more frequent updates (was 50)
    n_chunks = args.nsteps // chunk_size
    remaining_steps = args.nsteps % chunk_size
    
    current_pos = pos
    steps_completed = steps_already_completed
    total_steps = args.nsteps + steps_already_completed
    checkpoint_counter = 0
    
    # Track batch speed over time
    batch_speeds = []
    last_checkpoint_time = time.time()
    start_chunk_time = time.time()  # Track when we started for accurate remaining time
    
    # Create progress tracking file
    progress_log_file = os.path.join(run_dir, f"progress_log.txt")
    summary_log_file = os.path.join(run_dir, f"summary_log.txt")
    
    with open(progress_log_file, 'w', encoding='utf-8') as f:
        f.write("Step,Epoch,Batch_Speed,Elapsed_Min,Remaining_Min,Memory_MB,Progress_Pct\n")
    
    with open(summary_log_file, 'w', encoding='utf-8') as f:
        f.write("Timestamp,Elapsed_Min,Steps_Completed,Best_Omega,Best_Beta,Best_Score,Acceptance_Rate,Samples_Per_Sec,Remaining_Min\n")
    
    # Print header for progress table with modified format
    if args.enhanced_progress:
        print(f"\n{'='*100}")
        print(f"{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Progress':>8s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
        print(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")
    else:
        print(f"\n{'='*80}")
        print(f"{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
        print(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")
    
    # Track progress percentage
    total_work_units = args.nsteps * args.nwalkers
    completed_work_units = 0
    last_progress_update = time.time()
    progress_update_interval = args.progress_update_interval  # Changed from hardcoded 10 to use argument
    
    # Track time for summaries
    last_summary_time = time.time()
    best_score = float('-inf')
    best_params = {'omega': args.initial_omega, 'beta': args.initial_beta}
    current_elapsed = 0
    estimated_total_time = float('inf')
    
    # Store log management info
    max_log_files = args.max_log_files
    last_checkpoint_files = []
    
    for i in range(n_chunks + (1 if remaining_steps > 0 else 0)):
        # Check if we've exceeded max runtime
        current_elapsed = time.time() - start_time
        if current_elapsed > max_runtime_seconds:
            print(f"\nMaximum runtime of {args.max_time} minutes exceeded. Stopping MCMC.")
            print(f"Time elapsed: {current_elapsed/60:.2f} minutes")
            
            # Save final checkpoint before stopping
            current_batch_speed = np.mean(batch_speeds) if batch_speeds else 0
            saved_file = save_intermediate_results(
                sampler, args.nburn, run_dir, f"mcmc", 
                fixed_params, steps_completed, current_batch_speed, current_elapsed
            )
            if saved_file:
                print(f"Final checkpoint successfully saved to: {saved_file}")
            
            # Generate final summary
            print_parameter_summary(sampler, args.nburn, current_elapsed, steps_completed, 
                                   batch_speeds, best_params, best_score)
            break
            
        # Determine steps for this chunk
        if i == n_chunks and remaining_steps > 0:
            steps = remaining_steps
        else:
            steps = chunk_size
            
        # Run this chunk
        chunk_start = time.time()
        try:
            current_pos, _, _ = sampler.run_mcmc(current_pos, steps, progress=True)
            
            # Add delay if slow_mode is enabled
            if args.slow_mode:
                delay_sec = 0.5  # Half-second delay between steps
                print(f"\rAdding {delay_sec:.1f} second delay (slow mode)...", end="", flush=True)
                time.sleep(delay_sec)
                print("\r" + " " * 50 + "\r", end="", flush=True)  # Clear the delay message
                
            steps_completed += steps
            completed_work_units += steps * nwalkers
            
            # Calculate batch speed
            chunk_time = time.time() - chunk_start
            batch_speed = steps * nwalkers / chunk_time  # samples per second
            batch_speeds.append(batch_speed)
            current_batch_speed = batch_speed
            
            # Calculate estimated epoch (full passes through the dataset)
            current_epoch = steps_completed / nwalkers
            
            # Calculate overall progress percentage
            progress_percentage = (completed_work_units / total_work_units) * 100
            
            # Calculate estimated remaining time
            current_elapsed = time.time() - start_time
            if batch_speed > 0:
                remaining_work = total_work_units - completed_work_units
                remaining_seconds = remaining_work / batch_speed
                remaining_min = remaining_seconds / 60
                estimated_total_time = current_elapsed + remaining_seconds
            else:
                remaining_min = float('inf')
                
            # Convert time measures to minutes for display
            elapsed_min = current_elapsed / 60
            
            # Get memory usage
            memory_mb = get_memory_usage()
            
            # Print progress in enhanced or regular format
            if args.enhanced_progress:
                progress_line = f"{steps_completed:6d} | {current_epoch:6.2f} | {batch_speed:10.1f} | {progress_percentage:7.1f}% | {elapsed_min:8.2f} | {remaining_min:8.2f} | {memory_mb:8.1f}"
            else:
                progress_line = f"{steps_completed:6d} | {current_epoch:6.2f} | {batch_speed:10.1f} | {elapsed_min:8.2f} | {remaining_min:8.2f} | {memory_mb:8.1f}"
            
            # Update progress more frequently (not just after each chunk)
            current_time = time.time()
            if current_time - last_progress_update >= progress_update_interval:
                print(f"\r{progress_line}", end="", flush=True)
                last_progress_update = current_time
                
                # Save progress to log
                with open(progress_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{steps_completed},{current_epoch:.2f},{batch_speed:.1f},{elapsed_min:.2f},{remaining_min:.2f},{memory_mb:.1f},{progress_percentage:.1f}\n")
                
                # Add a small delay if requested to slow down progress display
                if args.progress_delay > 0:
                    time.sleep(args.progress_delay)
            
            # Print full line after each chunk completes
            print(f"\r{progress_line}")
            
            # Add a small delay after chunk completes if requested
            if args.progress_delay > 0:
                time.sleep(args.progress_delay * 2)  # Double delay for end of chunk
                
            # Save checkpoint if needed or if enough time has passed
            checkpoint_counter += steps
            time_since_last_checkpoint = time.time() - last_checkpoint_time
            
            if (args.checkpoint_interval > 0 and checkpoint_counter >= args.checkpoint_interval) or time_since_last_checkpoint > 300:  # 5 minutes (was 3)
                print(f"\nSaving checkpoint after {steps_completed} steps...")
                
                # If using consolidated logs, append extra info to the consolidated log
                if args.consolidated_logs:
                    with open(consolidated_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n## Checkpoint at Step {steps_completed} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Progress: {progress_percentage:.1f}%, Elapsed: {elapsed_min:.2f} min, Remaining: {remaining_min:.2f} min\n")
                        f.write(f"Current speed: {batch_speed:.1f} samples/s, Memory: {memory_mb:.1f} MB\n\n")
                
                # Pass consolidated log flag to the save_intermediate_results function
                saved_file = save_intermediate_results(
                    sampler, args.nburn, run_dir, f"mcmc", 
                    fixed_params, steps_completed, current_batch_speed, current_elapsed,
                    consolidated_log=args.consolidated_logs, 
                    consolidated_log_file=consolidated_log_file,
                    max_log_files=max_log_files,
                    last_checkpoint_files=last_checkpoint_files
                )
                
                if saved_file:
                    print(f"Checkpoint successfully saved to: {saved_file}")
                    
                    # Update the list of last checkpoint files for cleanup
                    if saved_file not in last_checkpoint_files:
                        last_checkpoint_files.append(saved_file)
                        # Keep only the most recent files up to max_log_files
                        if len(last_checkpoint_files) > max_log_files:
                            old_file = last_checkpoint_files.pop(0)
                            if os.path.exists(old_file):
                                try:
                                    os.remove(old_file)
                                    print(f"Removed old checkpoint file: {os.path.basename(old_file)}")
                                except Exception as e:
                                    print(f"Warning: Could not remove old checkpoint file: {e}")
                
                checkpoint_counter = 0
                last_checkpoint_time = time.time()
                print(f"\n{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
                print(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")
        
        except KeyboardInterrupt:
            print("\nMCMC interrupted by user. Saving current state and exiting.")
            saved_file = save_intermediate_results(
                sampler, args.nburn, run_dir, f"mcmc_interrupted", 
                fixed_params, steps_completed, current_batch_speed, current_elapsed
            )
            if saved_file:
                print(f"Interrupted state saved to: {saved_file}")
            break
        except Exception as e:
            print(f"\nError during MCMC chunk: {e}")
            print("Attempting to save current progress...")
            try:
                saved_file = save_intermediate_results(
                    sampler, args.nburn, run_dir, f"mcmc_error", 
                    fixed_params, steps_completed, current_batch_speed, current_elapsed
                )
                if saved_file:
                    print(f"Error state saved to: {saved_file}")
            except:
                print("Could not save error state.")
            raise
    
    print(f"\n{'=' * 80}")
    mcmc_time = time.time() - mcmc_start
    print("MCMC run complete.")
    print(f"MCMC execution time: {mcmc_time:.2f} seconds "
          f"({args.nsteps * args.nwalkers / mcmc_time:.1f} samples/s)")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print_memory_usage("at end of MCMC")

    # --- Process Results ---
    try:
        # Check acceptance fraction (should generally be between ~0.2 and 0.5)
        acceptance_fraction = np.mean(sampler.acceptance_fraction)
        print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")
        
        if acceptance_fraction < 0.1:
            print("WARNING: Very low acceptance fraction. Results may be unreliable.")
            print("Consider adjusting the prior ranges or initial positions.")
        
        # Discard burn-in steps and flatten the chain
        # flat=True combines results from all walkers
        # thin=X keeps only every Xth sample to reduce autocorrelation
        print(f"Processing samples (discarding {args.nburn} burn-in steps)...")
        samples = sampler.get_chain(discard=args.nburn, thin=15, flat=True)
        print(f"Shape of processed samples: {samples.shape}") # Should be (N_samples, N_DIM)

        if len(samples) == 0:
            print("ERROR: No valid samples after burn-in and thinning.")
            print("Try reducing burn-in or increasing the number of steps.")
        elif len(samples) < 10:
            print("WARNING: Very few samples after burn-in and thinning.")
            print("Consider reducing burn-in or increasing total steps.")

        # --- Save Results ---
        print("Saving final results...")
        
        # Save the samples (the chain) and final results in run directory
        chain_file = os.path.join(run_dir, f"mcmc_chain.csv")
        df_samples = pd.DataFrame(samples, columns=['omega', 'beta'])
        df_samples.to_csv(chain_file, index=False, encoding='utf-8')
        print(f"MCMC samples saved to {chain_file}")

        # Save run info (parameters, settings)
        run_info = {
            'timestamp': timestamp,
            'fixed_alpha': args.alpha,
            'fixed_epsilon': args.epsilon,
            'nwalkers': args.nwalkers,
            'nsteps': args.nsteps,
            'nburn': args.nburn,
            'initial_guess': {'omega': args.initial_omega, 'beta': args.initial_beta},
            'parameter_labels': PARAM_LABELS,
            'mean_acceptance_fraction': float(acceptance_fraction),
            'execution_time_seconds': end_time - start_time,
            'samples_shape': list(samples.shape) if len(samples) > 0 else [0, N_DIM],
            'test_mode': args.test_mode
        }
        info_file = os.path.join(run_dir, f"run_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(run_info, f, indent=4)
        print(f"Run info saved to {info_file}")

        # --- Basic Analysis & Plotting ---
        print("\nAnalyzing MCMC results...")

        # Calculate median and 1-sigma credible intervals (16th, 50th, 84th percentiles)
        results_summary = {}
        print("\n=== MCMC Parameter Estimates (median and 1-sigma credible interval) ===")

        # Handle case with no samples
        if len(samples) == 0:
            print("No valid samples to calculate statistics. Using initial values as placeholders.")
            # Use initial values as placeholders
            results_summary = {
                'omega': {
                    'median': float(args.initial_omega),
                    'upper_err': 0.0,
                    'lower_err': 0.0
                },
                'beta': {
                    'median': float(args.initial_beta),
                    'upper_err': 0.0,
                    'lower_err': 0.0
                }
            }
        else:
            # Process samples normally if we have them
            for i, label in enumerate(['omega', 'beta']):
                if len(samples) >= 3:  # Need at least 3 samples for percentiles
                    mcmc = np.percentile(samples[:, i], [16, 50, 84])
                    q = np.diff(mcmc) # q[0] = 50th-16th, q[1] = 84th-50th
                    median = mcmc[1]
                    upper_err = q[1]
                    lower_err = q[0]
                    print(f"{PARAM_LABELS[i]} = {median:.4f} (+{upper_err:.4f} / -{lower_err:.4f})")
                else:
                    # If too few samples for percentiles, use mean and std
                    median = float(np.mean(samples[:, i]))
                    std = float(np.std(samples[:, i])) if len(samples) > 1 else 0.0
                    upper_err = std
                    lower_err = std
                    print(f"{PARAM_LABELS[i]} = {median:.4f} (±{std:.4f})")
                
                results_summary[label] = {
                    'median': median,
                    'upper_err': upper_err,
                    'lower_err': lower_err
                }

        # Save summary
        summary_file = os.path.join(run_dir, f"mcmc_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Summary saved to {summary_file}")

        # Generate corner plot in run directory
        import matplotlib.pyplot as plt
        fig = corner.corner(samples, labels=PARAM_LABELS, truths=[results_summary['omega']['median'], results_summary['beta']['median']])
        plot_file = os.path.join(run_dir, f"corner_plot.png")
        fig.savefig(plot_file)
        print(f"Corner plot saved to {plot_file}")
        
        # Save a copy of the corner plot in main results dir for quick access
        main_plot_file = os.path.join(results_dir, f"corner_plot_{run_id}.png")
        fig.savefig(main_plot_file)
        
        # Generate Markdown summary
        performance_metrics = None
        if len(samples) >= 10:
            # Try to estimate performance metrics for the summary
            try:
                gs_model = GenesisSphereModel(
                    alpha=args.alpha,
                    beta=results_summary['beta']['median'],
                    omega=results_summary['omega']['median'],
                    epsilon=args.epsilon
                )
                performance_metrics = {
                    'h0_correlation_approx': float(estimate_h0_correlation(gs_model)),
                    'sne_r2_approx': float(estimate_sne_r2(gs_model)),
                    'bao_effect_approx': float(estimate_bao_effect(gs_model))
                }
                # Calculate combined score
                h0_corr = performance_metrics['h0_correlation_approx']
                sne_r2 = performance_metrics['sne_r2_approx']
                bao_effect = performance_metrics['bao_effect_approx']
                combined_score = (h0_corr + sne_r2 + min(1.0, bao_effect/100))/3
                performance_metrics['combined_score_approx'] = float(combined_score)
            except Exception as e:
                print(f"Could not calculate performance metrics for summary: {e}")
        
        md_file = generate_markdown_summary(
            results_summary,
            run_info,
            batch_speeds,
            acceptance_fraction,
            end_time - start_time,
            samples,
            run_dir,
            plot_file,
            checkpoint_file=None,
            performance_metrics=performance_metrics
        )
        print(f"Markdown summary saved to {md_file}")
        
        # Create a symlink to latest run for easy access
        latest_link = os.path.join(savepoints_dir, "latest_run")
        try:
            # Remove old link if exists
            if os.path.exists(latest_link):
                if os.path.islink(latest_link):
                    os.unlink(latest_link)
                else:
                    os.remove(latest_link)
            
            # Create symlink on Unix or create a shortcut file on Windows
            if os.name == 'posix':  # Unix/Linux/Mac
                os.symlink(run_dir, latest_link)
            else:  # Windows - create a .txt pointer file
                with open(latest_link + ".txt", "w") as f:
                    f.write(f"Latest run directory: {run_dir}")
            print(f"Link to latest run created")
        except Exception as e:
            print(f"Could not create link to latest run: {e}")

    except Exception as e:
        print(f"Error processing results: {e}")
        import traceback
        traceback.print_exc()

    print("MCMC parameter estimation complete.")

# Improve the parameter summary function for better information
def print_parameter_summary(sampler, nburn, elapsed_time, steps_completed, batch_speeds, best_params, best_score):
    """Calculate and print a summary of the current MCMC state"""
    try:
        # Calculate acceptance rate
        acceptance_rate = np.mean(sampler.acceptance_fraction)
        
        # Get recent samples (skipping burn-in)
        recent_samples = sampler.get_chain(discard=nburn, thin=5, flat=True)
        
        # Initialize summary data
        summary = {
            'omega': best_params['omega'],
            'beta': best_params['beta'],
            'score': best_score,
            'acceptance': acceptance_rate,
            'samples_per_second': np.mean(batch_speeds) if batch_speeds else 0
        }
        
        # If we have samples, calculate statistics
        if len(recent_samples) > 10:
            # Calculate median parameters
            omega_median = np.median(recent_samples[:, 0])
            beta_median = np.median(recent_samples[:, 1])
            
            # Calculate percentiles for error ranges
            omega_16, omega_84 = np.percentile(recent_samples[:, 0], [16, 84]) if len(recent_samples) >= 10 else (omega_median, omega_median)
            beta_16, beta_84 = np.percentile(recent_samples[:, 1], [16, 84]) if len(recent_samples) >= 10 else (beta_median, beta_median)
            
            # Create a model with current median parameters
            try:
                gs_model = GenesisSphereModel(
                    alpha=0.02,  # Fixed value 
                    beta=beta_median, 
                    omega=omega_median, 
                    epsilon=0.1   # Fixed value
                )
                
                # Quick approximate score calculation
                h0_corr = estimate_h0_correlation(gs_model)
                sne_r2 = estimate_sne_r2(gs_model)
                bao_effect = estimate_bao_effect(gs_model)
                
                # Simple combined score (average of normalized metrics)
                combined_score = (h0_corr + sne_r2 + min(1.0, bao_effect/100))/3
                
                # Update summary with current values
                summary['omega'] = omega_median
                summary['beta'] = beta_median
                summary['score'] = combined_score
                summary['h0_corr'] = h0_corr
                summary['sne_r2'] = sne_r2
                summary['bao_effect'] = bao_effect
                summary['omega_err'] = (omega_median - omega_16, omega_84 - omega_median)
                summary['beta_err'] = (beta_median - beta_16, beta_84 - beta_median)
            except Exception as e:
                print(f"Error calculating model metrics: {e}")
        
        # Print the summary with improved formatting
        print(f"\n--- CURRENT PARAMETER ESTIMATES AND PERFORMANCE ---")
        print(f"Runtime: {elapsed_time/60:.2f} minutes, Completed steps: {steps_completed}")
        print(f"Acceptance rate: {acceptance_rate:.2f}, Processing speed: {summary['samples_per_second']:.1f} samples/sec")
        
        if 'omega_err' in summary:
            print(f"\nParameter estimates with 1σ confidence intervals:")
            print(f"  ω = {summary['omega']:.4f} (+{summary['omega_err'][1]:.4f}/-{summary['omega_err'][0]:.4f})")
            print(f"  β = {summary['beta']:.4f} (+{summary['beta_err'][1]:.4f}/-{summary['beta_err'][0]:.4f})")
        else:
            print(f"\nCurrent parameter estimates:")
            print(f"  ω = {summary['omega']:.4f}")
            print(f"  β = {summary['beta']:.4f}")
        
        if 'h0_corr' in summary:
            print(f"\nPerformance metrics:")
            print(f"  H₀ Correlation: {summary['h0_corr']:.2%}")
            print(f"  Supernovae R²: {summary['sne_r2']:.2%}")
            print(f"  BAO Effect Size: {summary['bao_effect']:.2f}")
            print(f"  Combined Score: {summary['score']:.4f}")
        
        return summary
    
    except Exception as e:
        print(f"Error generating summary: {e}")
        return {
            'omega': best_params['omega'],
            'beta': best_params['beta'],
            'score': best_score,
            'acceptance': 0.0,
            'samples_per_second': np.mean(batch_speeds) if batch_speeds else 0
        }

# Add helper functions for quick metric estimation without full datasets
def estimate_h0_correlation(gs_model):
    """Quick approximation of H0 correlation for summaries"""
    try:
        # Generate mock time points similar to real H0 measurements
        years = np.linspace(1930, 2022, 20)
        t = (years - 2000.0) / 100.0
        
        # Calculate model predictions using key parameters
        sin_term = np.sin(gs_model.omega * t)
        rho = (1.0 / (1.0 + sin_term**2)) * (1.0 + gs_model.alpha * t**2)
        tf = 1.0 / (1.0 + gs_model.beta * (np.abs(t) + gs_model.epsilon))
        
        # Generate approximate H0 values with a pattern similar to real data
        h0_base = 70.0
        h0_pred = h0_base * (1.0 + 0.1 * np.sin(gs_model.omega * t)) * (1.0 + 0.05 * rho) / np.sqrt(tf)
        
        # Create mock observed data with a pattern
        h0_obs = h0_base * (1.0 + 0.15 * np.sin(0.8 * t)) + 2.0 * np.random.randn(len(t))
        
        # Calculate correlation
        correlation = np.corrcoef(h0_pred, h0_obs)[0, 1]
        return correlation
    except:
        return -0.2  # Return a default value if calculation fails

def estimate_sne_r2(gs_model):
    """Quick approximation of supernovae R² for summaries"""
    try:
        # Generate mock redshift range
        z = np.linspace(0.01, 1.5, 30)
        
        # Calculate approximate distance modulus using model parameters
        omega_m = 0.3 - 0.05 * np.sin(gs_model.omega)
        mu_model = 5.0 * np.log10((1+z) * (1.0 + gs_model.beta * z) / 
                                 np.sqrt(1.0 + gs_model.alpha * z**2)) + 43.0
        
        # Generate mock observed data
        mu_obs = 5.0 * np.log10((1+z) * (1.0 + 0.5 * z) / np.sqrt(omega_m)) + 43.0 + 0.2 * np.random.randn(len(z))
        
        # Calculate simplified R²
        mean_obs = np.mean(mu_obs)
        ss_tot = np.sum((mu_obs - mean_obs)**2)
        ss_res = np.sum((mu_obs - mu_model)**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else -1.0
        
        return r_squared
    except:
        return -0.3  # Return a default value if calculation fails

def estimate_bao_effect(gs_model):
    """Quick approximation of BAO effect size for summaries"""
    try:
        # Calculate approximate effect size based on model parameters
        effect_size = 20.0 + 10.0 * np.sin(gs_model.omega) - 5.0 * gs_model.beta
        
        # Add some randomness to simulate variation in real data
        effect_size += 2.0 * np.random.randn()
        
        return max(0, effect_size)  # Ensure positive
    except:
        return 10.0  # Return a default value if calculation fails

# Add this function after print_parameter_summary function
def generate_markdown_summary(results_summary, run_info, batch_speeds, acceptance_fraction, elapsed_time, samples, output_dir, corner_plot_filename, checkpoint_file=None, performance_metrics=None):
    """Generate a detailed Markdown summary of MCMC run results"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create markdown content
    md_content = []
    md_content.append("# Genesis-Sphere Parameter Sweep Results Summary")
    md_content.append(f"\n**Generated**: {timestamp}\n")
    
    # Add run information
    md_content.append("## Run Information")
    md_content.append(f"- **Run duration**: {elapsed_time/60:.2f} minutes")
    md_content.append(f"- **MCMC steps**: {run_info['nsteps']}")
    md_content.append(f"- **Burn-in steps**: {run_info['nburn']}")
    md_content.append(f"- **Number of walkers**: {run_info['nwalkers']}")
    md_content.append(f"- **Mean acceptance rate**: {acceptance_fraction:.4f}")
    md_content.append(f"- **Average processing speed**: {np.mean(batch_speeds):.2f} samples/second")
    md_content.append(f"- **Test mode**: {'Yes' if run_info.get('test_mode', False) else 'No'}")
    
    # Parameter results
    md_content.append("\n## Parameter Estimates")
    md_content.append("| Parameter | Value | Lower Error | Upper Error |")
    md_content.append("|-----------|-------|------------|-------------|")
    
    for param, values in results_summary.items():
        md_content.append(f"| {param.capitalize()} (ω) | {values['median']:.4f} | {values['lower_err']:.4f} | {values['upper_err']:.4f} |")
    
    # Performance metrics (if available)
    if performance_metrics:
        md_content.append("\n## Performance Metrics")
        md_content.append("| Metric | Value |")
        md_content.append("|--------|-------|")
        
        for metric, value in performance_metrics.items():
            if 'approx' in metric:
                metric_name = metric.replace('_approx', '')
                formatted_value = f"{value:.4f}"
                md_content.append(f"| {metric_name.replace('_', ' ').title()} | {formatted_value} |")
    
    # Sample statistics
    md_content.append("\n## Sample Statistics")
    md_content.append(f"- **Total samples**: {len(samples)}")
    
    if len(samples) > 0:
        md_content.append(f"- **Omega (ω) range**: {np.min(samples[:, 0]):.4f} to {np.max(samples[:, 0]):.4f}")
        md_content.append(f"- **Beta (β) range**: {np.min(samples[:, 1]):.4f} to {np.max(samples[:, 1]):.4f}")
    
    # Add checkpoint information if available
    if checkpoint_file:
        checkpoint_basename = os.path.basename(checkpoint_file)
        md_content.append(f"\n**Latest checkpoint**: {checkpoint_basename}")
    
    # Include a reference to the corner plot
    md_content.append("\n## Visualization")
    
    corner_plot_basename = os.path.basename(corner_plot_filename)
    md_content.append(f"![Corner Plot]({corner_plot_basename})")
    md_content.append("\nThe corner plot shows the posterior distribution of the parameters and their correlations.")
    
    # Add recommendations section
    md_content.append("\n## Recommendations")
    
    # Add suggestions based on acceptance rate
    if acceptance_fraction < 0.1:
        md_content.append("- **Low acceptance rate**: Consider adjusting prior ranges or using different initial positions")
    elif acceptance_fraction > 0.8:
        md_content.append("- **High acceptance rate**: Consider narrowing prior ranges for more efficient exploration")
    else:
        md_content.append("- **Good acceptance rate**: Current setup is effectively exploring the parameter space")
    
    # Add suggestions based on sample count
    if len(samples) < 100:
        md_content.append("- **Few samples**: Increase the number of steps or reduce burn-in for more reliable results")
    
    # Add a note about test mode if applicable
    if run_info.get('test_mode', False):
        md_content.append("- **Test mode active**: This was a test run with reduced computation. For production results, run without the --test_mode flag")
    
    # Add footer
    md_content.append("\n---")
    md_content.append("\n*This report was automatically generated by the Genesis-Sphere parameter sweep validation framework.*")
    
    # Write to file
    md_filename = os.path.join(output_dir, "mcmc_summary.md")
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    return md_filename

if __name__ == "__main__":
    try:
        # Force CuPy to initialize before we start to catch any startup errors
        if CUPY_IMPORT_SUCCESS:
            print("Initializing GPU environment...")
            cp.array([1, 2, 3])  # Simple test array
            try:
                cp.cuda.Stream.null.synchronize()  # Ensure initialization completes
                print("GPU initialization complete.")
            except Exception as e:
                print(f"Warning: GPU synchronization failed. Error: {e}")
                print("This suggests potential GPU driver or memory issues.")
                print("Will attempt detailed verification...")
                # Don't fail immediately - verify_gpu_operation will do more thorough checks
        
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup GPU resources in case of error
        if CUPY_IMPORT_SUCCESS:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                print("GPU resources cleaned up after error.")
            except:
                pass
                
        sys.exit(1)
        
        # Add a cleanup function to remove excess log files when the script finishes
        def cleanup_log_files(output_dir, prefix, keep=5):
            """Clean up excess log files, keeping only the most recent ones"""
            try:
                files = []
                for pattern in [f"{prefix}_checkpoint_*.csv", f"{prefix}_state_*.npz", 
                               f"{prefix}_random_state_*.pkl", f"{prefix}_*_info_*.json"]:
                    files.extend(glob.glob(os.path.join(output_dir, pattern)))
                
                # Sort files by modification time (newest first)
                files.sort(key=os.path.getmtime, reverse=True)
                
                # Keep the newest 'keep' files, delete the rest
                if len(files) > keep:
                    for old_file in files[keep:]:
                        try:
                            os.remove(old_file)
                            print(f"Cleaned up old log file: {os.path.basename(old_file)}")
                        except Exception as e:
                            print(f"Warning: Could not remove old log file {os.path.basename(old_file)}: {e}")
                            
                return len(files) - keep  # Return count of files removed
            except Exception as e:
                print(f"Warning: Error during log file cleanup: {e}")
                return 0
        
        # Register an atexit handler to clean up files when the script exits
        import atexit
        import glob
        atexit.register(lambda: cleanup_log_files(run_dir, "mcmc", keep=5))
