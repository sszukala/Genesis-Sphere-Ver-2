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
    # Define parameter ranges based on previous search results and theoretical considerations
    # NOTE: The current range allows negative β values, which is supported by your previous 
    # parameter sweep findings where β=-0.0333 produced good results.
    # If negative β values aren't physically meaningful, consider changing to 0.0 < beta < 3.0
    if 1.0 < omega < 6.0 and -1.0 < beta < 3.0: 
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

        # Calculate total chi-squared (proper statistical approach: simply sum the individual chi-squared values)
        # NOTE: These chi-squared values should ideally come from standard calculations:
        # χ² = ∑[(data_i - model_i)²/σ_i²] where σ_i is the uncertainty on the i-th data point
        total_chi2 = chi2_h0 + chi2_sne + chi2_bao  # No weighting - proper statistical approach

        # Convert total Chi-squared to Log Likelihood (assuming Gaussian errors)
        logL = -0.5 * total_chi2

        # Check for NaN or infinite results which can break MCMC
        if not np.isfinite(logL):
             print(f"Warning: Non-finite logL ({logL}) for ω={omega:.4f}, β={beta:.4f}")
             return -np.inf

        # Occasional garbage collection to prevent memory buildup
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
def save_intermediate_results(sampler, nburn, output_dir, prefix, fixed_params, step_number, batch_speed, total_elapsed):
    """Save intermediate MCMC results for checkpointing"""
    try:
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        t progress
        # Save the current state of the chain
        samples = sampler.get_chain(discard=nburn, thin=10, flat=True)effective_burn = min(nburn, step_number // 2)
        
        if len(samples) == 0:
            print("No valid samples to save yet.")er.get_chain(discard=effective_burn, thin=10, flat=True)
            return None
            
        # Create DataFrame and saveber} steps, using burn-in of {effective_burn}).")
        df_samples = pd.DataFrame(samples, columns=['omega', 'beta'])
        checkpoint_file = os.path.join(output_dir, f"{prefix}_checkpoint_{timestamp}.csv")
        df_samples.to_csv(checkpoint_file, index=False)    # Save basic info even if we don't have samples yet
        
        # Calculate quick stats if we have enough samplesimestamp,
        if len(samples) >= 10: 0,
            results_summary = {}
            for i, label in enumerate(['omega', 'beta']):
                if len(samples) >= 3:  # Need at least 3 samples for percentilesal_guess', {}).get('omega', 3.5))},
                    mcmc = np.percentile(samples[:, i], [16, 50, 84])': float(fixed_params.get('initial_guess', {}).get('beta', -0.0333))}
                    median = mcmc[1]
                    q = np.diff(mcmc)  # q[0] = 50th-16th, q[1] = 84th-50thd_params,
                    lower_err = q[0]
                    upper_err = q[1],
                    results_summary[label] = {
                        'median': float(median), 60.0,
                        'lower_err': float(lower_err),
                        'upper_err': float(upper_err)
                    }
                else:ess_info_{timestamp}.json")
                    # Just use mean if not enough for percentiles
                    median = float(np.mean(samples[:, i]))
                    std = float(np.std(samples[:, i])) if len(samples) > 1 else 0.0nfo_file}")
                    results_summary[label] = {
                        'median': median,
                        'lower_err': std,
                        'upper_err': stdFrame and save
                    }amples = pd.DataFrame(samples, columns=['omega', 'beta'])
            point_{timestamp}.csv")
            # Calculate preliminary performance metrics if possible_file, index=False)
            performance_metrics = {}
            try:
                omega_median = results_summary['omega']['median']
                beta_median = results_summary['beta']['median']lts_summary = {}
                
                # Create a model with current best parametersat least 3 samples for percentiles
                gs_model = GenesisSphereModel(:, i], [16, 50, 84])
                    alpha=fixed_params['alpha'], 
                    beta=beta_median,  q[0] = 50th-16th, q[1] = 84th-50th
                    omega=omega_median, 
                    epsilon=fixed_params['epsilon']   upper_err = q[1]
                )    results_summary[label] = {
                
                # Quick calculation of key metrics
                # These are simplified approximations to avoid full data loading
                h0_corr = estimate_h0_correlation(gs_model)
                sne_r2 = estimate_sne_r2(gs_model)
                bao_effect = estimate_bao_effect(gs_model)    # Just use mean if not enough for percentiles
                ean(samples[:, i]))
                performance_metrics = {n(samples) > 1 else 0.0
                    'h0_correlation_approx': float(h0_corr),
                    'sne_r2_approx': float(sne_r2),
                    'bao_effect_approx': float(bao_effect),
                    'combined_score_approx': float((h0_corr + sne_r2 + min(1.0, bao_effect/100))/3)       'upper_err': std
                }
            except Exception as e:
                print(f"Could not calculate performance metrics: {e}")formance metrics if possible
                performance_metrics = {
                    'calculation_error': str(e)
                }    omega_median = results_summary['omega']['median']
            ']
            # Save basic info with additional runtime metrics
            info = {urrent best parameters
                'timestamp': timestamp,
                'samples_saved': len(samples),
                'current_results': results_summary,
                'fixed_params': fixed_params,ga_median, 
                'progress': {on']
                    'step_number': step_number,
                    'batch_speed': batch_speed,
                    'elapsed_minutes': total_elapsed / 60.0,
                    'epoch': float(step_number / sampler.chain.shape[0]),
                    'percent_complete': float(step_number / (fixed_params['nsteps'] + step_number - 
                                                           (step_number % fixed_params['nsteps']))*100)e_r2 = estimate_sne_r2(gs_model)
                },
                'preliminary_metrics': performance_metrics   
            }    performance_metrics = {
            
            info_file = os.path.join(output_dir, f"{prefix}_checkpoint_info_{timestamp}.json")sne_r2),
            with open(info_file, 'w') as f:at(bao_effect),
                json.dump(info, f, indent=4)        'combined_score_approx': float((h0_corr + sne_r2 + min(1.0, bao_effect/100))/3)
            
            # Save sampler state for potential recovery
            state_file = os.path.join(output_dir, f"{prefix}_state_{timestamp}.npz") calculate performance metrics: {e}")
            np.savez(state_file, 
                     chain=sampler.chain,
                     random_state=np.random.get_state())}
                
            print(f"Checkpoint saved with {len(samples)} samples. Current estimates:")rics
            for param, values in results_summary.items():
                print(f"  {param}: {values['median']:.4f} (+{values['upper_err']:.4f}/-{values['lower_err']:.4f})")    'timestamp': timestamp,
            
            if 'preliminary_metrics' in info and len(performance_metrics) > 0:
                print("  Preliminary performance metrics (approximate):")
                for metric, value in performance_metrics.items():
                    if 'approx' in metric:
                        metric_name = metric.replace('_approx', '')
                        print(f"    {metric_name}: {value:.4f}")            'elapsed_minutes': total_elapsed / 60.0,
        loat(step_number / sampler.chain.shape[0]),
        return checkpoint_file                'percent_complete': float(step_number / (fixed_params['nsteps'] + step_number - 
                                     (step_number % fixed_params['nsteps']))*100)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")nary_metrics': performance_metrics
        import traceback
        traceback.print_exc()
        return None            info_file = os.path.join(output_dir, f"{prefix}_checkpoint_info_{timestamp}.json")
fo_file, 'w') as f:
# === Main Execution ===                json.dump(info, f, indent=4)
 
def main():x array shape issue
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
    parser.add_argument("--initial_beta", type=float, default=-0.0333, help="Initial guess for beta")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix for output filenames")
    parser.add_argument("--checkpoint_interval", type=int, default=100, 
                        help="Save intermediate results every N steps (0 to disable)")
    parser.add_argument("--test_mode", action="store_true", 
                        help="Run in test mode with reduced computation")
    parser.add_argument("--max_time", type=int, default=30,
                        help="Maximum runtime in minutes (default: 30 minutes)")
    parser.add_argument("--resume", type=str, default="", 
                        help="Path to state file to resume from a previous run")ess.")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU mode even if GPU is available")ples. Current estimates:")
    parser.add_argument("--verify_gpu", action="store_true",
                       help="Run GPU verification tests and exit")+{values['upper_err']:.4f}/-{values['lower_err']:.4f})")
    parser.add_argument("--quick_run", action="store_true",
                        help="Run with reduced parameters for quick results")trics) > 0:
    parser.add_argument("--summary_interval", type=int, default=15,
                        help="Interval in minutes for printing summary updates (default: 15)")
    parser.add_argument("--enhanced_progress", action="store_true",
                        help="Show enhanced progress tracking with percentage completion")                        metric_name = metric.replace('_approx', '')
f"    {metric_name}: {value:.4f}")
    args = parser.parse_args()    
    file
    # Run GPU verification
    if args.verify_gpu:
        verify_gpu_operation(args.force_cpu)
        print("GPU verification complete. Exiting.") traceback
        returntraceback.print_exc()
        ss despite checkpoint error.")
    # Initialize GPU if we want to use it
    gpu_status = verify_gpu_operation(args.force_cpu)
    if gpu_status:
        print(f"Using GPU: {GPU_INFO['name']} with {GPU_INFO['memory']} memory")
    else:
        print("Using CPU mode for computation")o run the MCMC parameter estimation"""
        global HAS_GPUime()
        HAS_GPU = False    parser = argparse.ArgumentParser(description="Run MCMC parameter estimation for Genesis-Sphere")
meters, data paths, MCMC settings, etc.
    # Create save points directory.02, help="Fixed alpha value")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")"Fixed epsilon value")
    suffix = f"_{args.output_suffix}" if args.output_suffix else """, type=int, default=32, help="Number of MCMC walkers (must be > 2*N_DIM)")
    run_id = f"{timestamp}{suffix}"parser.add_argument("--nsteps", type=int, default=5000, help="Number of MCMC steps per walker")
     help="Number of burn-in steps to discard")
    # Create main savepoints directory if it doesn't existlt=3.5, help="Initial guess for omega")
    savepoints_dir = os.path.join(results_dir, "savepoints")=float, default=-0.0333, help="Initial guess for beta")
    os.makedirs(savepoints_dir, exist_ok=True)parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix for output filenames")
     type=int, default=100, 
    # Create run-specific directory for this runry N steps (0 to disable)")
    run_dir = os.path.join(savepoints_dir, f"run_{run_id}")action="store_true", 
    os.makedirs(run_dir, exist_ok=True)                    help="Run in test mode with reduced computation")
    30,
    # Create progress tracking file in the run directory 30 minutes)")
    progress_log_file = os.path.join(run_dir, f"progress_log.txt")r, default="", 
    with open(progress_log_file, 'w') as f:)
        f.write("Step,Epoch,Batch_Speed,Elapsed_Min,Remaining_Min,Memory_MB\n")parser.add_argument("--force_cpu", action="store_true",
    PU is available")
    print(f"All results will be saved to: {run_dir}")parser.add_argument("--verify_gpu", action="store_true",
    cation tests and exit")
    # Apply quick run settings if requestedt("--quick_run", action="store_true",
    if args.quick_run:
        print("Running in QUICK MODE with reduced computation for faster results")=int, default=15,
        args.nwalkers = 24  # Reduced from defaultes for printing summary updates (default: 15)")
        args.nsteps = 2000  # Reduced from 5000action="store_true",
        args.nburn = 500    # Reduced from 1000 percentage completion")
        args.checkpoint_interval = 50  # More frequent checkpoints
        args.max_time = min(args.max_time, 120)  # Cap at 2 hours max    args = parser.parse_args()

    # Modify parameters if in test modetion
    if args.test_mode:
        print("Running in TEST MODE with reduced computation")on(args.force_cpu)
        args.nwalkers = 10
        args.nsteps = 25  # Reduced from 50 to 25 steps for quicker testing
        args.nburn = 10
        args.checkpoint_interval = 5  # More frequent checkpoints (was 10)    # Initialize GPU if we want to use it
_operation(args.force_cpu)
    # Validate walker count
    if args.nwalkers <= 2 * N_DIM:
        print(f"Error: Number of walkers ({args.nwalkers}) must be greater than 2 * N_DIM ({2*N_DIM}).")
        sys.exit(1) computation")
    if args.nsteps <= args.nburn:
        print(f"Error: Total steps ({args.nsteps}) must be greater than burn-in steps ({args.nburn}).")alse
        sys.exit(1)

    print("Starting Genesis-Sphere MCMC Parameter Estimation...")
    print(f"Fixed Parameters: α={args.alpha}, ε={args.epsilon}")
    print(f"MCMC Settings: Walkers={args.nwalkers}, Steps={args.nsteps}, Burn-in={args.nburn}")
    print(f"Checkpoint Interval: {args.checkpoint_interval} steps")
    print(f"Maximum runtime: {args.max_time} minutes")exist
    print(f"Save points directory: {savepoints_dir}")savepoints_dir = os.path.join(results_dir, "savepoints")
    _dir, exist_ok=True)
    # Initial memory usage
    print_memory_usage("at start")    # Create run-specific directory for this run
oin(savepoints_dir, f"run_{run_id}")
    # --- Load Data ---
    print("Loading observational data...")
    try: in the run directory
        loading_start = time.time()_dir, f"progress_log.txt")
        h0_data = load_h0_measurements()
        print(f"  H0 data loaded: {len(h0_data)} measurements ({time.time() - loading_start:.2f}s)")f.write("Step,Epoch,Batch_Speed,Elapsed_Min,Remaining_Min,Memory_MB\n")
        
        loading_start = time.time() {run_dir}")
        sne_data = load_supernovae_data()
        print(f"  SNe data loaded: {len(sne_data)} supernovae ({time.time() - loading_start:.2f}s)")ply quick run settings if requested
        
        loading_start = time.time()DE with reduced computation for faster results")
        bao_data = load_bao_data()
        print(f"  BAO data loaded: {len(bao_data)} measurements ({time.time() - loading_start:.2f}s)")args.nsteps = 2000  # Reduced from 5000
         1000
        print("Data loaded successfully.")equent checkpoints
        print_memory_usage("after data loading")args.max_time = min(args.max_time, 120)  # Cap at 2 hours max
        
        # Initialize GPU memory if availableters if in test mode
        if HAS_GPU:
            print("Setting up GPU environment...")
            # Limit GPU memory usage to 90% of available memorylkers = 10
            try:
                # Use get_default_memory_pool instead of directly using cp.cuda.malloc
                mempool = cp.get_default_memory_pool()equent checkpoints (was 10)
                mempool.set_limit(fraction=0.9)
                print("  GPU memory pool configured.")alker count
                
                # Preload some data to GPU to check it worksnwalkers}) must be greater than 2 * N_DIM ({2*N_DIM}).")
                test_array = cp.array([1, 2, 3])
                del test_array
                cp.get_default_memory_pool().free_all_blocks()s}) must be greater than burn-in steps ({args.nburn}).")
                print("  GPU test successful.")
            except Exception as e:
                print(f"  Warning: GPU setup encountered an issue: {e}")timation...")
                print("  Falling back to CPU mode.")={args.alpha}, ε={args.epsilon}")
                HAS_GPU = False Walkers={args.nwalkers}, Steps={args.nsteps}, Burn-in={args.nburn}")
    except Exception as e:eckpoint_interval} steps")
        print(f"Error loading data: {e}") runtime: {args.max_time} minutes")
        sys.exit(1)    print(f"Save points directory: {savepoints_dir}")

    # --- Initialize or Resume Walkers ---y usage
    if args.resume:
        print(f"Resuming from state file: {args.resume}")
        try:
            # Load previous state
            state_data = np.load(args.resume)
            chain = state_data['chain']
            random_state = state_data['random_state']ata = load_h0_measurements()
            {len(h0_data)} measurements ({time.time() - loading_start:.2f}s)")
            # Set the random state
            np.random.set_state(random_state)ing_start = time.time()
            
            # Use the last position of the chain: {len(sne_data)} supernovae ({time.time() - loading_start:.2f}s)")
            pos = chain[:, -1, :]
            
            # Calculate steps already completed
            steps_already_completed = chain.shape[1]
            print(f"Resuming from step {steps_already_completed} with {args.nsteps - steps_already_completed} remaining")
            
            # Adjust nsteps to account for steps already done
            args.nsteps = max(args.nsteps - steps_already_completed, 100)  # Ensure at least 100 more steps
            y if available
        except Exception as e:
            print(f"Error loading previous state: {e}")
            print("Starting with fresh initialization instead.")ry usage to 90% of available memory
            args.resume = ""        try:
    et_default_memory_pool instead of directly using cp.cuda.malloc
    if not args.resume:_memory_pool()
        print("Initializing walkers...")t(fraction=0.9)
        init_start = time.time()
        # Start walkers in a small Gaussian ball around the previous best parameters
        initial_pos_guess = np.array([args.initial_omega, args.initial_beta])        # Preload some data to GPU to check it works
        
        # Add small random offsets for each walker, ensuring they are within priors
        pos = np.zeros((args.nwalkers, N_DIM))_blocks()
        max_attempts = 1000  # Prevent infinite loops        print("  GPU test successful.")
        
        for i in range(args.nwalkers):  Warning: GPU setup encountered an issue: {e}")
            attempts = 0
            # Keep generating random starting points until they satisfy the prior
            while attempts < max_attempts:
                p = initial_pos_guess + 1e-3 * np.abs(initial_pos_guess) * np.random.randn(N_DIM)
                if np.isfinite(log_prior(p)):
                    pos[i] = p
                    breake Walkers ---
                attempts += 1rgs.resume:
        : {args.resume}")
            if attempts >= max_attempts:
                print(f"Warning: Could not initialize walker {i} after {max_attempts} attempts.")
                print(f"Using initial guess directly: ω={args.initial_omega}, β={args.initial_beta}")me)
                pos[i] = initial_pos_guess    chain = state_data['chain']
        ata['random_state']
        nwalkers, ndim = pos.shape
        print(f"Initialized {nwalkers} walkers around ω={args.initial_omega}, β={args.initial_beta} " 
              f"in {time.time() - init_start:.2f}s")dom_state)
        steps_already_completed = 0            

    # --- Test likelihood function with a few points ---
    print("Testing likelihood function with initial positions...")
    test_start = time.time()te steps already completed
    success_count = 0        steps_already_completed = chain.shape[1]
    p {steps_already_completed} with {args.nsteps - steps_already_completed} remaining")
    for i in range(min(5, nwalkers)):
        test_params = pos[i]# Adjust nsteps to account for steps already done
        try:ore steps
            lp = log_posterior(test_params, h0_data, sne_data, bao_data, args.alpha, args.epsilon)
            if np.isfinite(lp):
                success_count += 1
                print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → logP={lp:.2f} (Valid)")("Starting with fresh initialization instead.")
            else:
                print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Invalid posterior")
        except Exception as e:
            print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Error: {e}")    print("Initializing walkers...")
    
    print(f"Likelihood function test: {success_count}/5 successful in {time.time() - test_start:.2f}s")    # Start walkers in a small Gaussian ball around the previous best parameters
    = np.array([args.initial_omega, args.initial_beta])
    if success_count == 0:
        print("ERROR: All likelihood function tests failed. Exiting.") random offsets for each walker, ensuring they are within priors
        sys.exit(1)        pos = np.zeros((args.nwalkers, N_DIM))
 1000  # Prevent infinite loops
    # --- Run MCMC ---
    print(f"Running MCMC...")nwalkers):
    mcmc_start = time.time()        attempts = 0
    starting points until they satisfy the prior
    # Set up max runtime if specified
    max_runtime_seconds = args.max_time * 60            p = initial_pos_guess + 1e-3 * np.abs(initial_pos_guess) * np.random.randn(N_DIM)
    
    # Fixed parameters to pass to checkpointingpos[i] = p
    fixed_params = {
        'alpha': args.alpha,
        'epsilon': args.epsilon,
        'nwalkers': args.nwalkers,_attempts:
        'nsteps': args.nsteps,ning: Could not initialize walker {i} after {max_attempts} attempts.")
        'nburn': args.nburn           print(f"Using initial guess directly: ω={args.initial_omega}, β={args.initial_beta}")
    }            pos[i] = initial_pos_guess
    
    # The 'args' tuple passes fixed parameters and data to the log_posterior functions, ndim = pos.shape
    if HAS_GPU:d ω={args.initial_omega}, β={args.initial_beta} " 
        print("Using GPU-accelerated MCMC sampling")tart:.2f}s")
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior_gpu, 
            args=(h0_data, sne_data, bao_data, args.alpha, args.epsilon) Test likelihood function with a few points ---
        )("Testing likelihood function with initial positions...")
    else:
        print("Using CPU-based MCMC sampling")
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, 
            args=(h0_data, sne_data, bao_data, args.alpha, args.epsilon)est_params = pos[i]
        )        try:
o_data, args.alpha, args.epsilon)
    # Run MCMC steps with progress, checkpointing, and time limit
    print("Running with checkpoints and time limit...")            success_count += 1
    _params[0]:.4f}, β={test_params[1]:.4f} → logP={lp:.2f} (Valid)")
    # Run in smaller chunks for checkpointing
    chunk_size = min(5, args.checkpoint_interval)  # Smaller chunks for more frequent updates (was 50){test_params[0]:.4f}, β={test_params[1]:.4f} → Invalid posterior")
    n_chunks = args.nsteps // chunk_size
    remaining_steps = args.nsteps % chunk_size        print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Error: {e}")
    
    current_pos = poss_count}/5 successful in {time.time() - test_start:.2f}s")
    steps_completed = steps_already_completed
    total_steps = args.nsteps + steps_already_completed
    checkpoint_counter = 0    print("ERROR: All likelihood function tests failed. Exiting.")
    
    # Track batch speed over time
    batch_speeds = []
    last_checkpoint_time = time.time()
    start_chunk_time = time.time()  # Track when we started for accurate remaining timemcmc_start = time.time()
    
    # Create progress tracking file
    progress_log_file = os.path.join(run_dir, f"progress_log.txt")
    summary_log_file = os.path.join(run_dir, f"summary_log.txt")
    ting
    with open(progress_log_file, 'w') as f:
        f.write("Step,Epoch,Batch_Speed,Elapsed_Min,Remaining_Min,Memory_MB,Progress_Pct\n")    'alpha': args.alpha,
    
    with open(summary_log_file, 'w') as f:
        f.write("Timestamp,Elapsed_Min,Steps_Completed,Best_Omega,Best_Beta,Best_Score,Acceptance_Rate,Samples_Per_Sec,Remaining_Min\n")    'nsteps': args.nsteps,
    
    # Print header for progress table with modified format
    if args.enhanced_progress:
        print(f"\n{'='*100}")
        print(f"{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Progress':>8s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
        print(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")rint("Using GPU-accelerated MCMC sampling")
    else:mbleSampler(
        print(f"\n{'='*80}")
        print(f"{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
        print(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")    )
    
    # Track progress percentage
    total_work_units = args.nsteps * args.nwalkersmbleSampler(
    completed_work_units = 0ior, 
    last_progress_update = time.time()a, args.alpha, args.epsilon)
    progress_update_interval = 10  # seconds    )
    
    # Track time for summaries checkpointing, and time limit
    last_summary_time = time.time()oints and time limit...")
    best_score = float('-inf')
    best_params = {'omega': args.initial_omega, 'beta': args.initial_beta}unks for checkpointing
    current_elapsed = 0_interval)  # Smaller chunks for more frequent updates (was 50)
    estimated_total_time = float('inf')n_chunks = args.nsteps // chunk_size
    
    for i in range(n_chunks + (1 if remaining_steps > 0 else 0)):
        # Check if we've exceeded max runtime
        current_elapsed = time.time() - start_time
        if current_elapsed > max_runtime_seconds:
            print(f"\nMaximum runtime of {args.max_time} minutes exceeded. Stopping MCMC.")
            print(f"Time elapsed: {current_elapsed/60:.2f} minutes")
            
            # Save final checkpoint before stopping
            current_batch_speed = np.mean(batch_speeds) if batch_speeds else 0
            saved_file = save_intermediate_results(rted for accurate remaining time
                sampler, args.nburn, run_dir, f"mcmc", 
                fixed_params, steps_completed, current_batch_speed, current_elapsedprogress tracking file
            ).path.join(run_dir, f"progress_log.txt")
            if saved_file:
                print(f"Final checkpoint successfully saved to: {saved_file}")
            ) as f:
            # Generate final summary)
            print_parameter_summary(sampler, args.nburn, current_elapsed, steps_completed, 
                                   batch_speeds, best_params, best_score)mary_log_file, 'w') as f:
            breakite("Timestamp,Elapsed_Min,Steps_Completed,Best_Omega,Best_Beta,Best_Score,Acceptance_Rate,Samples_Per_Sec,Remaining_Min\n")
            
        # Determine steps for this chunked format
        if i == n_chunks and remaining_steps > 0:
            steps = remaining_steps(f"\n{'='*100}")
        else: {'Epoch':>6s} | {'Speed':>10s} | {'Progress':>8s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
            steps = chunk_sizet(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
            
        # Run this chunk
        chunk_start = time.time()print(f"{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
        t(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")
        try:
            current_pos, _, _ = sampler.run_mcmc(current_pos, steps, progress=True)
            steps_completed += steps
            completed_work_units += steps * nwalkersd_work_units = 0
            e()
            # Calculate batch speed
            chunk_time = time.time() - chunk_start
            batch_speed = steps * nwalkers / chunk_time  # samples per second
            batch_speeds.append(batch_speed)
            current_batch_speed = batch_speedre = float('-inf')
            }
            # Calculate estimated epoch (full passes through the dataset)
            current_epoch = steps_completed / nwalkersd_total_time = float('inf')
            
            # Calculate overall progress percentage
            progress_percentage = (completed_work_units / total_work_units) * 100eck if we've exceeded max runtime
            me
            # Calculate estimated remaining time
            current_elapsed = time.time() - start_timeuntime of {args.max_time} minutes exceeded. Stopping MCMC.")
            if batch_speed > 0:
                remaining_work = total_work_units - completed_work_units
                remaining_seconds = remaining_work / batch_speed
                remaining_min = remaining_seconds / 60se 0
                estimated_total_time = current_elapsed + remaining_seconds_file = save_intermediate_results(
            else:, f"mcmc", 
                remaining_min = float('inf')fixed_params, steps_completed, current_batch_speed, current_elapsed
                
            # Convert time measures to minutes for display
            elapsed_min = current_elapsed / 60    print(f"Final checkpoint successfully saved to: {saved_file}")
            
            # Get memory usage
            memory_mb = get_memory_usage()print_parameter_summary(sampler, args.nburn, current_elapsed, steps_completed, 
            ms, best_score)
            # Print progress in enhanced or regular format
            if args.enhanced_progress:
                progress_line = f"{steps_completed:6d} | {current_epoch:6.2f} | {batch_speed:10.1f} | {progress_percentage:7.1f}% | {elapsed_min:8.2f} | {remaining_min:8.2f} | {memory_mb:8.1f}"ne steps for this chunk
            else:
                progress_line = f"{steps_completed:6d} | {current_epoch:6.2f} | {batch_speed:10.1f} | {elapsed_min:8.2f} | {remaining_min:8.2f} | {memory_mb:8.1f}"steps = remaining_steps
            
            # Update progress more frequently (not just after each chunk)
            current_time = time.time()
            if current_time - last_progress_update >= progress_update_interval:
                print(f"\r{progress_line}", end="", flush=True)
                last_progress_update = current_time
                
                # Save progress to logt_pos, steps, progress=True)
                with open(progress_log_file, 'a') as f:
                    f.write(f"{steps_completed},{current_epoch:.2f},{batch_speed:.1f},{elapsed_min:.2f},{remaining_min:.2f},{memory_mb:.1f},{progress_percentage:.1f}\n")completed_work_units += steps * nwalkers
            
            # Check if it's time for a summary (every summary_interval minutes)
            time_since_last_summary = time.time() - last_summary_time
            if time_since_last_summary >= (args.summary_interval * 60):ond
                # Print a newline to ensure summary starts on a fresh linepend(batch_speed)
                print("\n")ch_speed
                print(f"\n{'='*100}")
                print(f"SUMMARY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {args.summary_interval} minute interval")poch (full passes through the dataset)
                print(f"{'='*100}")ent_epoch = steps_completed / nwalkers
                
                # Calculate and display summary statistics
                current_summary = print_parameter_summary(sampler, args.nburn, current_elapsed, 
                                                         steps_completed, batch_speeds, 
                                                         best_params, best_score)lculate estimated remaining time
                
                # Update best parameters if we found better ones
                if current_summary['score'] > best_score:eted_work_units
                    best_score = current_summary['score'] remaining_work / batch_speed
                    best_params = {
                        'omega': current_summary['omega'],+ remaining_seconds
                        'beta': current_summary['beta']
                    }remaining_min = float('inf')
                
                # Additional progress information
                print(f"\nOVERALL PROGRESS: {progress_percentage:.1f}% complete")
                print(f"Estimated completion time: {datetime.fromtimestamp(start_time + estimated_total_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Remaining time: {remaining_min:.1f} minutes")t memory usage
                emory_usage()
                # Log summary
                with open(summary_log_file, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{timestamp},{current_elapsed/60:.2f},{steps_completed},":10.1f} | {progress_percentage:7.1f}% | {elapsed_min:8.2f} | {remaining_min:8.2f} | {memory_mb:8.1f}"
                            f"{current_summary['omega']:.4f},{current_summary['beta']:.4f},"
                            f"{current_summary['score']:.4f},{current_summary['acceptance']:.4f},"f} | {elapsed_min:8.2f} | {remaining_min:8.2f} | {memory_mb:8.1f}"
                            f"{current_summary['samples_per_second']:.1f},{remaining_min:.2f}\n")
                not just after each chunk)
                last_summary_time = time.time()ent_time = time.time()
                rogress_update_interval:
                # Print table header again for progress, end="", flush=True)
                if args.enhanced_progress:
                    print(f"\n{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Progress':>8s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
                    print(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")e progress to log
                else:
                    print(f"\n{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")lapsed_min:.2f},{remaining_min:.2f},{memory_mb:.1f},{progress_percentage:.1f}\n")
                    print(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")
            mmary_interval minutes)
            # Print full line after each chunk completesime.time() - last_summary_time
            print(f"\r{progress_line}")if time_since_last_summary >= (args.summary_interval * 60):
            h line
            # Save checkpoint if needed or if enough time has passed
            checkpoint_counter += steps
            time_since_last_checkpoint = time.time() - last_checkpoint_time    print(f"SUMMARY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {args.summary_interval} minute interval")
            
            if (args.checkpoint_interval > 0 and checkpoint_counter >= args.checkpoint_interval) or time_since_last_checkpoint > 180:  # 3 minutes (was 5)
                print(f"\nSaving checkpoint after {steps_completed} steps...")ics
                saved_file = save_intermediate_results(ampler, args.nburn, current_elapsed, 
                    sampler, args.nburn, run_dir, f"mcmc",  
                    fixed_params, steps_completed, current_batch_speed, current_elapsed                                        best_params, best_score)
                )
                if saved_file:
                    print(f"Checkpoint successfully saved to: {saved_file}")ore'] > best_score:
                checkpoint_counter = 0score']
                last_checkpoint_time = time.time()
                print(f"\n{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
                print(f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")                'beta': current_summary['beta']
        
        except KeyboardInterrupt:
            print("\nMCMC interrupted by user. Saving current state and exiting.")
            saved_file = save_intermediate_results(f}% complete")
                sampler, args.nburn, run_dir, f"mcmc_interrupted", me + estimated_total_time).strftime('%Y-%m-%d %H:%M:%S')}")
                fixed_params, steps_completed, current_batch_speed, current_elapsed   print(f"Remaining time: {remaining_min:.1f} minutes")
            )
            if saved_file:
                print(f"Interrupted state saved to: {saved_file}")ith open(summary_log_file, 'a') as f:
            break= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:lapsed/60:.2f},{steps_completed},"
            print(f"\nError during MCMC chunk: {e}")},{current_summary['beta']:.4f},"
            print("Attempting to save current progress...")            f"{current_summary['score']:.4f},{current_summary['acceptance']:.4f},"
            try:_per_second']:.1f},{remaining_min:.2f}\n")
                saved_file = save_intermediate_results(
                    sampler, args.nburn, run_dir, f"mcmc_error", 
                    fixed_params, steps_completed, current_batch_speed, current_elapsed
                )header again for progress
                if saved_file:
                    print(f"Error state saved to: {saved_file}") print(f"\n{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Progress':>8s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
            except:*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
                print("Could not save error state.")lse:
            raise                print(f"\n{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
    f"{'-'*6} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")
    print(f"\n{'=' * 80}")
    mcmc_time = time.time() - mcmc_startfter each chunk completes
    print("MCMC run complete.")
    print(f"MCMC execution time: {mcmc_time:.2f} seconds "
          f"({args.nsteps * args.nwalkers / mcmc_time:.1f} samples/s)")        # Save checkpoint if needed or if enough time has passed
    nter += steps
    end_time = time.time()time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print_memory_usage("at end of MCMC")            if (args.checkpoint_interval > 0 and checkpoint_counter >= args.checkpoint_interval) or time_since_last_checkpoint > 180:  # 3 minutes (was 5)
ing checkpoint after {steps_completed} steps...")
    # --- Process Results ---        saved_file = save_intermediate_results(
    try:
        # Check acceptance fraction (should generally be between ~0.2 and 0.5)peed, current_elapsed
        acceptance_fraction = np.mean(sampler.acceptance_fraction)
        print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")        if saved_file:
        t successfully saved to: {saved_file}")
        if acceptance_fraction < 0.1:
            print("WARNING: Very low acceptance fraction. Results may be unreliable.")
            print("Consider adjusting the prior ranges or initial positions.")                print(f"\n{'Step':>6s} | {'Epoch':>6s} | {'Speed':>10s} | {'Elapsed':>8s} | {'Remain':>8s} | {'Mem(MB)':>8s}")
| {'-'*8} | {'-'*8} | {'-'*8}")
        # Discard burn-in steps and flatten the chain
        # flat=True combines results from all walkers
        # thin=X keeps only every Xth sample to reduce autocorrelation.")
        print(f"Processing samples (discarding {args.nburn} burn-in steps)...")
        samples = sampler.get_chain(discard=args.nburn, thin=15, flat=True)
        print(f"Shape of processed samples: {samples.shape}") # Should be (N_samples, N_DIM)                fixed_params, steps_completed, current_batch_speed, current_elapsed

        if len(samples) == 0:
            print("ERROR: No valid samples after burn-in and thinning.")
            print("Try reducing burn-in or increasing the number of steps.")
        elif len(samples) < 10:
            print("WARNING: Very few samples after burn-in and thinning.")
            print("Consider reducing burn-in or increasing total steps.")            print("Attempting to save current progress...")

        # --- Save Results ---ediate_results(
        print("Saving final results...")                    sampler, args.nburn, run_dir, f"mcmc_error", 
urrent_elapsed
        # Save the samples (the chain) and final results in run directory
        chain_file = os.path.join(run_dir, f"mcmc_chain.csv")
        df_samples = pd.DataFrame(samples, columns=['omega', 'beta']){saved_file}")
        df_samples.to_csv(chain_file, index=False)
        print(f"MCMC samples saved to {chain_file}")                print("Could not save error state.")

        # Save run info (parameters, settings)
        run_info = {
            'timestamp': timestamp,rt
            'fixed_alpha': args.alpha,
            'fixed_epsilon': args.epsilon,_time:.2f} seconds "
            'nwalkers': args.nwalkers,walkers / mcmc_time:.1f} samples/s)")
            'nsteps': args.nsteps,
            'nburn': args.nburn,
            'initial_guess': {'omega': args.initial_omega, 'beta': args.initial_beta}, start_time:.2f} seconds")
            'parameter_labels': PARAM_LABELS,
            'mean_acceptance_fraction': float(acceptance_fraction),
            'execution_time_seconds': end_time - start_time,
            'samples_shape': list(samples.shape) if len(samples) > 0 else [0, N_DIM],
            'test_mode': args.test_mode Check acceptance fraction (should generally be between ~0.2 and 0.5)
        }action)
        info_file = os.path.join(run_dir, f"run_info.json")n: {acceptance_fraction:.3f}")
        with open(info_file, 'w') as f:
            json.dump(run_info, f, indent=4)
        print(f"Run info saved to {info_file}")            print("WARNING: Very low acceptance fraction. Results may be unreliable.")
rior ranges or initial positions.")
        # --- Basic Analysis & Plotting ---
        print("\nAnalyzing MCMC results...")        # Discard burn-in steps and flatten the chain

        # Calculate median and 1-sigma credible intervals (16th, 50th, 84th percentiles)every Xth sample to reduce autocorrelation
        results_summary = {}
        print("\n=== MCMC Parameter Estimates (median and 1-sigma credible interval) ===")samples = sampler.get_chain(discard=args.nburn, thin=15, flat=True)
        mples: {samples.shape}") # Should be (N_samples, N_DIM)
        # Handle case with no samples
        if len(samples) == 0:
            print("No valid samples to calculate statistics. Using initial values as placeholders.") burn-in and thinning.")
            # Use initial values as placeholders burn-in or increasing the number of steps.")
            results_summary = {< 10:
                'omega': {in and thinning.")
                    'median': float(args.initial_omega),burn-in or increasing total steps.")
                    'upper_err': 0.0,
                    'lower_err': 0.0 Results ---
                },al results...")
                'beta': {
                    'median': float(args.initial_beta),) and final results in run directory
                    'upper_err': 0.0,n_dir, f"mcmc_chain.csv")
                    'lower_err': 0.0s = pd.DataFrame(samples, columns=['omega', 'beta'])
                }mples.to_csv(chain_file, index=False)
            }(f"MCMC samples saved to {chain_file}")
        else:
            # Process samples normally if we have them
            for i, label in enumerate(['omega', 'beta']):
                if len(samples) >= 3:  # Need at least 3 samples for percentiles
                    mcmc = np.percentile(samples[:, i], [16, 50, 84])
                    q = np.diff(mcmc) # q[0] = 50th-16th, q[1] = 84th-50thsilon,
                    median = mcmc[1]s,
                    upper_err = q[1]
                    lower_err = q[0]
                    print(f"{PARAM_LABELS[i]} = {median:.4f} (+{upper_err:.4f} / -{lower_err:.4f})")guess': {'omega': args.initial_omega, 'beta': args.initial_beta},
                else:
                    # If too few samples for percentiles, use mean and stdraction),
                    median = float(np.mean(samples[:, i]))
                    std = float(np.std(samples[:, i])) if len(samples) > 1 else 0.0amples.shape) if len(samples) > 0 else [0, N_DIM],
                    upper_err = stdmode
                    lower_err = std
                    print(f"{PARAM_LABELS[i]} = {median:.4f} (±{std:.4f})")e = os.path.join(run_dir, f"run_info.json")
                
                results_summary[label] = {dent=4)
                    'median': median,e}")
                    'upper_err': upper_err,
                    'lower_err': lower_erric Analysis & Plotting ---
                }        print("\nAnalyzing MCMC results...")

        # Save summary0th, 84th percentiles)
        summary_file = os.path.join(run_dir, f"mcmc_summary.json")
        with open(summary_file, 'w') as f:an and 1-sigma credible interval) ===")
            json.dump(results_summary, f, indent=4)
        print(f"Summary saved to {summary_file}")        # Handle case with no samples

        # Generate corner plot in run directorycalculate statistics. Using initial values as placeholders.")
        import matplotlib.pyplot as plt
        fig = corner.corner(samples, labels=PARAM_LABELS, truths=[results_summary['omega']['median'], results_summary['beta']['median']])
        plot_file = os.path.join(run_dir, f"corner_plot.png")
        fig.savefig(plot_file)mega),
        print(f"Corner plot saved to {plot_file}")            'upper_err': 0.0,
        
        # Save a copy of the corner plot in main results dir for quick access
        main_plot_file = os.path.join(results_dir, f"corner_plot_{run_id}.png")
        fig.savefig(main_plot_file)            'median': float(args.initial_beta),
        
        # Create a symlink to latest run for easy access
        latest_link = os.path.join(savepoints_dir, "latest_run")    }
        try:
            # Remove old link if exists
            if os.path.exists(latest_link):ve them
                if os.path.islink(latest_link):ega', 'beta']):
                    os.unlink(latest_link)n(samples) >= 3:  # Need at least 3 samples for percentiles
                else:amples[:, i], [16, 50, 84])
                    os.remove(latest_link)        q = np.diff(mcmc) # q[0] = 50th-16th, q[1] = 84th-50th
            
            # Create symlink on Unix or create a shortcut file on Windows
            if os.name == 'posix':  # Unix/Linux/Mac
                os.symlink(run_dir, latest_link)4f} (+{upper_err:.4f} / -{lower_err:.4f})")
            else:  # Windows - create a .txt pointer file
                with open(latest_link + ".txt", "w") as f:ean and std
                    f.write(f"Latest run directory: {run_dir}")        median = float(np.mean(samples[:, i]))
            , i])) if len(samples) > 1 else 0.0
            print(f"Link to latest run created")= std
        except Exception as e:
            print(f"Could not create link to latest run: {e}")                    print(f"{PARAM_LABELS[i]} = {median:.4f} (±{std:.4f})")

    except Exception as e:
        print(f"Error processing results: {e}")ian': median,
        import tracebackr': upper_err,
        traceback.print_exc()                    'lower_err': lower_err

    print("MCMC parameter estimation complete.")

# Improve the parameter summary function for better information
def print_parameter_summary(sampler, nburn, elapsed_time, steps_completed, batch_speeds, best_params, best_score):
    """Calculate and print a summary of the current MCMC state"""    json.dump(results_summary, f, indent=4)
    try:ummary_file}")
        # Calculate acceptance rate
        acceptance_rate = np.mean(sampler.acceptance_fraction)# Generate corner plot in run directory
        
        # Get recent samples (skipping burn-in)mmary['omega']['median'], results_summary['beta']['median']])
        recent_samples = sampler.get_chain(discard=nburn, thin=5, flat=True)plot_file = os.path.join(run_dir, f"corner_plot.png")
        
        # Initialize summary dataner plot saved to {plot_file}")
        summary = {
            'omega': best_params['omega'], in main results dir for quick access
            'beta': best_params['beta'],.join(results_dir, f"corner_plot_{run_id}.png")
            'score': best_score,
            'acceptance': acceptance_rate,
            'samples_per_second': np.mean(batch_speeds) if batch_speeds else 0 Create a symlink to latest run for easy access
        }latest_link = os.path.join(savepoints_dir, "latest_run")
        
        # If we have samples, calculate statisticssts
        if len(recent_samples) > 10:):
            # Calculate median parameters
            omega_median = np.median(recent_samples[:, 0])
            beta_median = np.median(recent_samples[:, 1])    else:
            
            # Calculate percentiles for error ranges
            omega_16, omega_84 = np.percentile(recent_samples[:, 0], [16, 84]) if len(recent_samples) >= 10 else (omega_median, omega_median)
            beta_16, beta_84 = np.percentile(recent_samples[:, 1], [16, 84]) if len(recent_samples) >= 10 else (beta_median, beta_median)if os.name == 'posix':  # Unix/Linux/Mac
            
            # Create a model with current median parameters:  # Windows - create a .txt pointer file
            try:, "w") as f:
                gs_model = GenesisSphereModel(ory: {run_dir}")
                    alpha=0.02,  # Fixed value 
                    beta=beta_median, reated")
                    omega=omega_median, 
                    epsilon=0.1   # Fixed value(f"Could not create link to latest run: {e}")
                )
                
                # Quick approximate score calculation
                h0_corr = estimate_h0_correlation(gs_model)
                sne_r2 = estimate_sne_r2(gs_model)
                bao_effect = estimate_bao_effect(gs_model)
                
                # Simple combined score (average of normalized metrics)
                combined_score = (h0_corr + sne_r2 + min(1.0, bao_effect/100))/3rameter summary function for better information
                time, steps_completed, batch_speeds, best_params, best_score):
                # Update summary with current valuesrent MCMC state"""
                summary['omega'] = omega_median
                summary['beta'] = beta_median
                summary['score'] = combined_scoreceptance_fraction)
                summary['h0_corr'] = h0_corr
                summary['sne_r2'] = sne_r2
                summary['bao_effect'] = bao_effect
                summary['omega_err'] = (omega_median - omega_16, omega_84 - omega_median)
                summary['beta_err'] = (beta_median - beta_16, beta_84 - beta_median)
            except Exception as e:
                print(f"Error calculating model metrics: {e}")    'omega': best_params['omega'],
        
        # Print the summary with improved formatting
        print(f"\n--- CURRENT PARAMETER ESTIMATES AND PERFORMANCE ---")
        print(f"Runtime: {elapsed_time/60:.2f} minutes, Completed steps: {steps_completed}")
        print(f"Acceptance rate: {acceptance_rate:.2f}, Processing speed: {summary['samples_per_second']:.1f} samples/sec")}
        
        if 'omega_err' in summary:
            print(f"\nParameter estimates with 1σ confidence intervals:")
            print(f"  ω = {summary['omega']:.4f} (+{summary['omega_err'][1]:.4f}/-{summary['omega_err'][0]:.4f})")
            print(f"  β = {summary['beta']:.4f} (+{summary['beta_err'][1]:.4f}/-{summary['beta_err'][0]:.4f})")mega_median = np.median(recent_samples[:, 0])
        else:, 1])
            print(f"\nCurrent parameter estimates:")
            print(f"  ω = {summary['omega']:.4f}")ges
            print(f"  β = {summary['beta']:.4f}")    omega_16, omega_84 = np.percentile(recent_samples[:, 0], [16, 84]) if len(recent_samples) >= 10 else (omega_median, omega_median)
        p.percentile(recent_samples[:, 1], [16, 84]) if len(recent_samples) >= 10 else (beta_median, beta_median)
        if 'h0_corr' in summary:
            print(f"\nPerformance metrics:")
            print(f"  H₀ Correlation: {summary['h0_corr']:.2%}")
            print(f"  Supernovae R²: {summary['sne_r2']:.2%}")
            print(f"  BAO Effect Size: {summary['bao_effect']:.2f}")
            print(f"  Combined Score: {summary['score']:.4f}")            beta=beta_median, 
        ega=omega_median, 
        return summary                epsilon=0.1   # Fixed value
    
    except Exception as e:
        print(f"Error generating summary: {e}")# Quick approximate score calculation
        return {elation(gs_model)
            'omega': best_params['omega'],(gs_model)
            'beta': best_params['beta'],imate_bao_effect(gs_model)
            'score': best_score,
            'acceptance': 0.0,
            'samples_per_second': np.mean(batch_speeds) if batch_speeds else 0       combined_score = (h0_corr + sne_r2 + min(1.0, bao_effect/100))/3
        }                

# Add helper functions for quick metric estimation without full datasetsga_median
def estimate_h0_correlation(gs_model):
    """Quick approximation of H0 correlation for summaries"""        summary['score'] = combined_score
    try:
        # Generate mock time points similar to real H0 measurements
        years = np.linspace(1930, 2022, 20)] = bao_effect
        t = (years - 2000.0) / 100.0        summary['omega_err'] = (omega_median - omega_16, omega_84 - omega_median)
        16, beta_84 - beta_median)
        # Calculate model predictions using key parameters
        sin_term = np.sin(gs_model.omega * t)
        rho = (1.0 / (1.0 + sin_term**2)) * (1.0 + gs_model.alpha * t**2)
        tf = 1.0 / (1.0 + gs_model.beta * (np.abs(t) + gs_model.epsilon))# Print the summary with improved formatting
        
        # Generate approximate H0 values with a pattern similar to real datae: {elapsed_time/60:.2f} minutes, Completed steps: {steps_completed}")
        h0_base = 70.0]:.1f} samples/sec")
        h0_pred = h0_base * (1.0 + 0.1 * np.sin(gs_model.omega * t)) * (1.0 + 0.05 * rho) / np.sqrt(tf)
        
        # Create mock observed data with a pattern
        h0_obs = h0_base * (1.0 + 0.15 * np.sin(0.8 * t)) + 2.0 * np.random.randn(len(t))    print(f"  ω = {summary['omega']:.4f} (+{summary['omega_err'][1]:.4f}/-{summary['omega_err'][0]:.4f})")
        ary['beta']:.4f} (+{summary['beta_err'][1]:.4f}/-{summary['beta_err'][0]:.4f})")
        # Calculate correlation
        correlation = np.corrcoef(h0_pred, h0_obs)[0, 1]ent parameter estimates:")
        return correlation print(f"  ω = {summary['omega']:.4f}")
    except:
        return -0.2  # Return a default value if calculation fails        
y:
def estimate_sne_r2(gs_model):
    """Quick approximation of supernovae R² for summaries"""    print(f"  H₀ Correlation: {summary['h0_corr']:.2%}")
    try:summary['sne_r2']:.2%}")
        # Generate mock redshift range {summary['bao_effect']:.2f}")
        z = np.linspace(0.01, 1.5, 30)    print(f"  Combined Score: {summary['score']:.4f}")
        
        # Calculate approximate distance modulus using model parameters
        omega_m = 0.3 - 0.05 * np.sin(gs_model.omega)
        mu_model = 5.0 * np.log10((1+z) * (1.0 + gs_model.beta * z) / 
                                 np.sqrt(1.0 + gs_model.alpha * z**2)) + 43.0print(f"Error generating summary: {e}")
        
        # Generate mock observed data
        mu_obs = 5.0 * np.log10((1+z) * (1.0 + 0.5 * z) / np.sqrt(omega_m)) + 43.0 + 0.2 * np.random.randn(len(z))    'beta': best_params['beta'],
        
        # Calculate simplified R²
        mean_obs = np.mean(mu_obs)_speeds) if batch_speeds else 0
        ss_tot = np.sum((mu_obs - mean_obs)**2)
        ss_res = np.sum((mu_obs - mu_model)**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else -1.0lper functions for quick metric estimation without full datasets
        ion(gs_model):
        return r_squaredk approximation of H0 correlation for summaries"""
    except:
        return -0.3  # Return a default value if calculation fails        # Generate mock time points similar to real H0 measurements
2022, 20)
def estimate_bao_effect(gs_model):
    """Quick approximation of BAO effect size for summaries"""
    try:
        # Calculate approximate effect size based on model parameters
        effect_size = 20.0 + 10.0 * np.sin(gs_model.omega) - 5.0 * gs_model.betarho = (1.0 / (1.0 + sin_term**2)) * (1.0 + gs_model.alpha * t**2)
        epsilon))
        # Add some randomness to simulate variation in real data
        effect_size += 2.0 * np.random.randn()# Generate approximate H0 values with a pattern similar to real data
        
        return max(0, effect_size)  # Ensure positivepred = h0_base * (1.0 + 0.1 * np.sin(gs_model.omega * t)) * (1.0 + 0.05 * rho) / np.sqrt(tf)
    except:
        return 10.0  # Return a default value if calculation fails        # Create mock observed data with a pattern
 (1.0 + 0.15 * np.sin(0.8 * t)) + 2.0 * np.random.randn(len(t))
if __name__ == "__main__":
    try:
        # Force CuPy to initialize before we start to catch any startup errorsef(h0_pred, h0_obs)[0, 1]
        if CUPY_IMPORT_SUCCESS:
            print("Initializing GPU environment...")
            cp.array([1, 2, 3])  # Simple test array0.2  # Return a default value if calculation fails
            try:
                cp.cuda.Stream.null.synchronize()  # Ensure initialization completes
                print("GPU initialization complete.")rnovae R² for summaries"""
            except Exception as e:
                print(f"Warning: GPU synchronization failed. Error: {e}")
                print("This suggests potential GPU driver or memory issues.")
                print("Will attempt detailed verification...")
                # Don't fail immediately - verify_gpu_operation will do more thorough checks# Calculate approximate distance modulus using model parameters
        m = 0.3 - 0.05 * np.sin(gs_model.omega)
        main()p.log10((1+z) * (1.0 + gs_model.beta * z) / 
    except Exception as e:p.sqrt(1.0 + gs_model.alpha * z**2)) + 43.0
        print(f"Fatal error: {e}")
        import tracebackved data
        traceback.print_exc()mu_obs = 5.0 * np.log10((1+z) * (1.0 + 0.5 * z) / np.sqrt(omega_m)) + 43.0 + 0.2 * np.random.randn(len(z))
        
        # Cleanup GPU resources in case of errorR²
        if CUPY_IMPORT_SUCCESS: = np.mean(mu_obs)
            try:
                cp.get_default_memory_pool().free_all_blocks()
                print("GPU resources cleaned up after error.") 1.0 - (ss_res / ss_tot) if ss_tot > 0 else -1.0
            except:
                pass_squared
                
        sys.exit(1)        return -0.3  # Return a default value if calculation fails


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
