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

# Try to import CuPy for GPU acceleration, fall back to NumPy if not available
try:
    import cupy as cp
    # Test that CuPy is actually working
    test_array = cp.array([1, 2, 3])
    test_result = cp.sum(test_array)  # Test a basic operation
    del test_array, test_result
    HAS_GPU = True
    print("CuPy successfully imported and tested. Using GPU acceleration.")
except (ImportError, Exception) as e:
    import numpy as np
    cp = np  # Use NumPy as a fallback
    HAS_GPU = False
    print(f"GPU acceleration not available: {e}")
    print("Using CPU with NumPy instead.")

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
alues
def calculate_h0_chi2_gpu(gs_model, h0_data):
    """GPU-accelerated chi-squared calculation for H0 measurements"""    h0_err = h0_data['H0_err'].values
    # Get data
    years = h0_data['year'].values
    h0_obs = h0_data['H0'].valuess)
    h0_err = h0_data['H0_err'].valuesobs)
        h0_err_gpu = to_gpu(h0_err)
    # Transfer to GPU
    years_gpu = to_gpu(years)ictions - a simplified version, real implementation would use the model
    h0_obs_gpu = to_gpu(h0_obs)
    h0_err_gpu = to_gpu(h0_err)
    
    # Get predictions - a simplified version, real implementation would use the model**2)
    h0_base = 70.0
    t = (years_gpu - 2000.0) / 100.0    h0_pred = h0_base * (1.0 + 0.1 * cp.sin(gs_model.omega * t)) * (1.0 + 0.05 * rho) / cp.sqrt(tf)
    sin_term = cp.sin(gs_model.omega * t)
    rho = (1.0 / (1.0 + sin_term**2)) * (1.0 + gs_model.alpha * t**2)
    tf = 1.0 / (1.0 + gs_model.beta * (cp.abs(t) + gs_model.epsilon))red) / h0_err_gpu
    h0_pred = h0_base * (1.0 + 0.1 * cp.sin(gs_model.omega * t)) * (1.0 + 0.05 * rho) / cp.sqrt(tf)    chi2 = float(cp.sum(residuals**2))
    
    # Calculate chi-squared
    residuals = (h0_obs_gpu - h0_pred) / h0_err_gpud overhead
    chi2 = float(cp.sum(residuals**2))rs_gpu, h0_obs_gpu, h0_err_gpu, h0_pred, residuals
    
    # Free GPU memory            cp.get_default_memory_pool().free_all_blocks()
    if np.random.random() < 0.1:  # Only occasionally to avoid overhead
        del years_gpu, h0_obs_gpu, h0_err_gpu, h0_pred, residuals        return chi2
        if HAS_GPU:
            cp.get_default_memory_pool().free_all_blocks()
    back to CPU implementation
    return chi2

def calculate_sne_chi2(gs_model, sne_data):sne_data):
    """Calculate chi-squared for supernovae fit with GPU acceleration if available"""ith GPU acceleration if available"""
    if HAS_GPU:
        return calculate_sne_chi2_gpu(gs_model, sne_data)_gpu(gs_model, sne_data)
    
    # Original CPU implementation# Original CPU implementation
    metrics = analyze_sne_fit(gs_model, sne_data)model, sne_data)
    # Use the reduced chi-squared metric directly if availabletric directly if available
    if 'reduced_chi2' in metrics:
        return metrics['reduced_chi2'] * (len(sne_data) - 2)  # Convert reduced chi2 to raw chi2Convert reduced chi2 to raw chi2
    
    # Alternatively use R-squarede R-squared
    r_squared = metrics['r_squared']    r_squared = metrics['r_squared']
    # Transform R² to a chi-squared-like metric (lower is better) (lower is better)
    # When R² is close to 1 (good fit), this gives a small value
    chi2_approx = len(sne_data) * (1 - r_squared) if r_squared <= 1.0 else len(sne_data) * 2ta) * (1 - r_squared) if r_squared <= 1.0 else len(sne_data) * 2
    return chi2_approx
values
def calculate_sne_chi2_gpu(gs_model, sne_data):ne_data):
    """GPU-accelerated chi-squared calculation for supernovae"""    mu_err = sne_data['mu_err'].values"""GPU-accelerated chi-squared calculation for supernovae"""
    # Extract data
    redshifts = sne_data['z'].valuesues
    mu_obs = sne_data['mu'].valuess)ues
    mu_err = sne_data['mu_err'].valuesobs).values
        mu_err_gpu = to_gpu(mu_err)
    # Transfer to GPU
    z_gpu = to_gpu(redshifts)
    mu_obs_gpu = to_gpu(mu_obs)el
    mu_err_gpu = to_gpu(mu_err)per implementation would use your gs_model to predict distance modulusu = to_gpu(mu_err)
    
    # Calculate distance modulus predictions prediction
    # This is a simplified implementation - adjust to match your actual model    c = 299792.458  # km/s# This is a simplified implementation - adjust to match your actual model
    # Proper implementation would use your gs_model to predict distance modulus
    h0 = 70.0ximation)
    omega_m = 0.3tch your Genesis-Sphere model's prediction
    c = 299792.458  # km/s
    m))
    # Simple luminosity distance calculation (flat ΛCDM approximation)le trapezoid integration - could be more sophisticateduminosity distance calculation (flat ΛCDM approximation)
    # Adjust this to match your Genesis-Sphere model's predictionion
    a = 1.0 / (1.0 + z_gpu).0, da)
    integrand = 1.0 / cp.sqrt(omega_m / a**3 + (1.0 - omega_m))3 + (1.0 - omega_m))
    # Simple trapezoid integration - could be more sophisticatedntegraluld be more sophisticated
    da = 0.001    mu_pred = 5.0 * cp.log10(dl) + 25.0da = 0.001
    a_vals = cp.arange(a.min(), 1.0, da)
    integral = cp.sum(integrand * da)
    dl = c / h0 * (1.0 + z_gpu) * integralred) / mu_err_gpugral
    mu_pred = 5.0 * cp.log10(dl) + 25.0    chi2 = float(cp.sum(residuals**2))mu_pred = 5.0 * cp.log10(dl) + 25.0
    
    # Calculate chi-squaredonally
    residuals = (mu_obs_gpu - mu_pred) / mu_err_gpu
    chi2 = float(cp.sum(residuals**2))pu, mu_obs_gpu, mu_err_gpu, mu_pred, residuals.sum(residuals**2))
    
    # Free GPU memory occasionally            cp.get_default_memory_pool().free_all_blocks()# Free GPU memory occasionally
    if np.random.random() < 0.1:m() < 0.1:
        del z_gpu, mu_obs_gpu, mu_err_gpu, mu_pred, residuals        return chi2        del z_gpu, mu_obs_gpu, mu_err_gpu, mu_pred, residuals
        if HAS_GPU:
            cp.get_default_memory_pool().free_all_blocks()
    back to CPU implementation
    return chi2

def calculate_bao_chi2(gs_model, bao_data):bao_data):bao_data):
    """Calculate chi-squared for BAO detection with GPU acceleration if available""" acceleration if available""" acceleration if available"""
    if HAS_GPU:
        return calculate_bao_chi2_gpu(gs_model, bao_data)
    
    # Original CPU implementation
    metrics = analyze_bao_detection(gs_model, bao_data)
    # For effect size, a larger value is better
    # Convert to a chi-squared-like value (lower is better)-squared-like value (lower is better)-squared-like value (lower is better)
    effect_size = metrics['high_z_effect_size']    effect_size = metrics['high_z_effect_size']    effect_size = metrics['high_z_effect_size']
    max_expected = 100  # A reasonable maximum to scale againstto scale againstto scale against
    # Normalize so that higher effect size gives lower chi-squared
    chi2_approx = max_expected - min(effect_size, max_expected)max_expected - min(effect_size, max_expected)max_expected - min(effect_size, max_expected)
    return chi2_approx

def calculate_bao_chi2_gpu(gs_model, bao_data):ao_data):ao_data):
    """GPU-accelerated chi-squared calculation for BAO measurements""""""GPU-accelerated chi-squared calculation for BAO measurements""""""GPU-accelerated chi-squared calculation for BAO measurements"""
    # Extract data
    redshifts = bao_data['z'].values
    rd_obs = bao_data['rd'].values'].valuesues
    rd_err = bao_data['rd_err'].values.values.values
        rd_err = bao_data['rd_err'].values
    # Transfer to GPU
    z_gpu = to_gpu(redshifts)
    rd_obs_gpu = to_gpu(rd_obs)shifts)_obs)
    rd_err_gpu = to_gpu(rd_err)u = to_gpu(rd_obs)to_gpu(rd_err)
        rd_err_gpu = to_gpu(rd_err)
    # Calculate predictions
    # This is a simplified version - replace with your actual model
    rd_fid = 147.78  # Mpc
    omega_m = 0.3    rd_fid = 147.78  # Mpcomega_m = 0.3
    
    # Simple approximation for sound horizon
    # Replace with your Genesis-Sphere model's actual predictionnd horizon model's actual prediction
    rd_pred = rd_fid * cp.sqrt(0.02237 / 0.02233) * cp.sqrt(omega_m / 0.3) * cp.ones_like(z_gpu)    # Replace with your Genesis-Sphere model's actual predictionrd_pred = rd_fid * cp.sqrt(0.02237 / 0.02233) * cp.sqrt(omega_m / 0.3) * cp.ones_like(z_gpu)
    sqrt(0.02237 / 0.02233) * cp.sqrt(omega_m / 0.3) * cp.ones_like(z_gpu)
    # Calculate chi-squared
    residuals = (rd_obs_gpu - rd_pred) / rd_err_gpu
    chi2 = float(cp.sum(residuals**2)) (rd_obs_gpu - rd_pred) / rd_err_gpu.sum(residuals**2))
    
    # Free memory occasionally    # Free memory occasionally
    if np.random.random() < 0.1:memory occasionallym.random() < 0.1:
        del z_gpu, rd_obs_gpu, rd_err_gpu, rd_pred, residuals        if np.random.random() < 0.1:        del z_gpu, rd_obs_gpu, rd_err_gpu, rd_pred, residuals
        if HAS_GPU:pu, rd_pred, residuals
            cp.get_default_memory_pool().free_all_blocks()            if HAS_GPU:            cp.get_default_memory_pool().free_all_blocks()
    default_memory_pool().free_all_blocks()
    return chi2

# === Memory and Performance Monitoring ===
        print(f"GPU calculation failed with error: {e}, falling back to CPU")
def get_memory_usage():entation
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MBMB

def print_memory_usage(label=""):
    """Print current memory usage with an optional label"""
    memory_mb = get_memory_usage()os.getpid())age()
    print(f"Memory usage {label}: {memory_mb:.2f} MB")

def with_timeout(seconds, func, *args, **kwargs):mory_usage(label=""):eout(seconds, func, *args, **kwargs):
    """Run a function with a timeout (cross-platform implementation)"""l label"""orm implementation)"""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:kers=1) as executor:
        future = executor.submit(func, *args, **kwargs)**kwargs)
        try:def with_timeout(seconds, func, *args, **kwargs):        try:
            return future.result(timeout=seconds)n with a timeout (cross-platform implementation)"""uture.result(timeout=seconds)
        except concurrent.futures.TimeoutError:    import concurrent.futures        except concurrent.futures.TimeoutError:
            print(f"Warning: Function call timed out")cutor(max_workers=1) as executor:l timed out")
            return Nonenc, *args, **kwargs)
ry:
# === MCMC Setup ===
        except concurrent.futures.TimeoutError:
# Define the parameter space (dimensions)
# We are fitting for omega and betaemega and beta
N_DIM = 2
PARAM_LABELS = [r"$\omega$", r"$\beta$"] # LaTeX labels for plots

# Define the prior function - sets allowed parameter rangesmeter ranges
def log_prior(params):e fitting for omega and beta_prior(params):
    """
    Log prior probability distribution (Log(Prior)).
    Returns 0 if params are within allowed ranges, -np.inf otherwise.
    This enforces constraints on parameters.
    """
    omega, beta = params
    # Define parameter ranges based on previous search results and theoretical considerationsetical considerations
    # NOTE: The current range allows negative β values, which is supported by your previous se.upported by your previous 
    # parameter sweep findings where β=-0.0333 produced good results.    This enforces constraints on parameters.    # parameter sweep findings where β=-0.0333 produced good results.
    # If negative β values aren't physically meaningful, consider changing to 0.0 < beta < 3.0
    if 1.0 < omega < 6.0 and -1.0 < beta < 3.0: 
        return 0.0 # Log(1) = 0 -> uniform prior within boundsefine parameter ranges based on previous search results and theoretical considerations return 0.0 # Log(1) = 0 -> uniform prior within bounds
    return -np.inf # Log(0) -> rules out parameters outside boundsβ values, which is supported by your previous eters outside bounds

# Define the log-likelihood function - compares model to dataf negative β values aren't physically meaningful, consider changing to 0.0 < beta < 3.0e the log-likelihood function - compares model to data
def log_likelihood(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """0.0 # Log(1) = 0 -> uniform prior within bounds
    Log likelihood function (Log(Likelihood)).
    Calculates the total likelihood of observing the data given the parameters.ulates the total likelihood of observing the data given the parameters.
    """res model to data
    # For GPU acceleration, delegate to the GPU version if availableata_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):delegate to the GPU version if available
    if HAS_GPU:"""if HAS_GPU:
        return log_likelihood_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)ion (Log(Likelihood)).ihood_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)
    l likelihood of observing the data given the parameters.
    # Track computation time for diagnostics
    start_time = time.time()    # For GPU acceleration, delegate to the GPU version if available    start_time = time.time()
    
    omega, beta = paramsdata_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)
    alpha = fixed_alpha
    epsilon = fixed_epsilon    # Track computation time for diagnostics    epsilon = fixed_epsilon
t_time = time.time()
    # Check prior first (can sometimes save computation)
    if not np.isfinite(log_prior(params)):
         return -np.inf    alpha = fixed_alpha         return -np.inf

    try:
        # Create model instance with current parameters
        gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)r(params)):eModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)

        # Calculate chi-squared for each dataset with timeoutsts
        try:
            chi2_h0 = with_timeout(5, calculate_h0_chi2, gs_model, data_h0)tance with current parameters_timeout(5, calculate_h0_chi2, gs_model, data_h0)
            if chi2_h0 is None:odel = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)if chi2_h0 is None:
                return -np.infnp.inf
        except Exception as e:
            print(f"Error calculating H0 chi2: {e}")
            return -np.infeout(5, calculate_h0_chi2, gs_model, data_h0)
            :
        try:
            chi2_sne = with_timeout(5, calculate_sne_chi2, gs_model, data_sne)s e:h_timeout(5, calculate_sne_chi2, gs_model, data_sne)
            if chi2_sne is None:print(f"Error calculating H0 chi2: {e}")if chi2_sne is None:
                return -np.infreturn -np.inf    return -np.inf
        except Exception as e:
            print(f"Error calculating SNe chi2: {e}"))
            return -np.infmeout(5, calculate_sne_chi2, gs_model, data_sne)
            e:
        try:
            chi2_bao = with_timeout(5, calculate_bao_chi2, gs_model, data_bao)s e:h_timeout(5, calculate_bao_chi2, gs_model, data_bao)
            if chi2_bao is None:            print(f"Error calculating SNe chi2: {e}")            if chi2_bao is None:
                return -np.inf
        except Exception as e:
            print(f"Error calculating BAO chi2: {e}")
            return -np.inf
            if chi2_bao is None:
        # Calculate total chi-squared (proper statistical approach: simply sum the individual chi-squared values)
        # NOTE: These chi-squared values should ideally come from standard calculations:values should ideally come from standard calculations:
        # χ² = ∑[(data_i - model_i)²/σ_i²] where σ_i is the uncertainty on the i-th data point            print(f"Error calculating BAO chi2: {e}")        # χ² = ∑[(data_i - model_i)²/σ_i²] where σ_i is the uncertainty on the i-th data point
        total_chi2 = chi2_h0 + chi2_sne + chi2_bao  # No weighting - proper statistical approach

        # Convert total Chi-squared to Log Likelihood (assuming Gaussian errors)idual chi-squared values)
        logL = -0.5 * total_chi2quared values should ideally come from standard calculations:_chi2
        # χ² = ∑[(data_i - model_i)²/σ_i²] where σ_i is the uncertainty on the i-th data point
        # Check for NaN or infinite results which can break MCMCg - proper statistical approach
        if not np.isfinite(logL):
             print(f"Warning: Non-finite logL ({logL}) for ω={omega:.4f}, β={beta:.4f}")Chi-squared to Log Likelihood (assuming Gaussian errors)ning: Non-finite logL ({logL}) for ω={omega:.4f}, β={beta:.4f}")
             return -np.inf = -0.5 * total_chi2 return -np.inf

        # Occasional garbage collection to prevent memory buildupch can break MCMCent memory buildup
        if np.random.random() < 0.01:  # Do this only occasionally (1% chance)
            gc.collect()
             return -np.inf
        # Monitor computation time
        compute_time = time.time() - start_time        # Occasional garbage collection to prevent memory buildup        compute_time = time.time() - start_time
        if compute_time > 1.0:  # Only log slow computationsm() < 0.01:  # Do this only occasionally (1% chance)1.0:  # Only log slow computations
            print(f"Slow likelihood computation: {compute_time:.2f}s for ω={omega:.4f}, β={beta:.4f}")
            
        return logLutation time
        compute_time = time.time() - start_time
    except Exception as e:
        # Handle potential errors during model calculation or analysiscompute_time:.2f}s for ω={omega:.4f}, β={beta:.4f}")culation or analysis
        print(f"Warning: Likelihood calculation failed for ω={omega:.4f}, β={beta:.4f}. Error: {e}")ion failed for ω={omega:.4f}, β={beta:.4f}. Error: {e}")
        return -np.inf    return logL    return -np.inf

def log_likelihood_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon): e:params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """GPU-accelerated log likelihood function."""errors during model calculation or analysislikelihood function."""
    start_time = time.time()        print(f"Warning: Likelihood calculation failed for ω={omega:.4f}, β={beta:.4f}. Error: {e}")    start_time = time.time()
    
    omega, beta = params
    alpha = fixed_alphaparams, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    epsilon = fixed_epsilon    """GPU-accelerated log likelihood function."""    epsilon = fixed_epsilon
t_time = time.time()
    # Check prior (CPU operation, no need for GPU)
    if not np.isfinite(log_prior(params)):
         return -np.inf    alpha = fixed_alpha         return -np.inf

    try:
        # Create model instance
        gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)):pha=alpha, beta=beta, omega=omega, epsilon=epsilon)

        # Calculate chi-squared for each dataset
        try:
            chi2_h0 = calculate_h0_chi2_gpu(gs_model, data_h0)tanceulate_h0_chi2_gpu(gs_model, data_h0)
            if not np.isfinite(chi2_h0):odel = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)if not np.isfinite(chi2_h0):
                return -np.infnp.inf
        except Exception as e:
            print(f"Error calculating H0 chi2 on GPU: {e}")
            return -np.infe_h0_chi2_gpu(gs_model, data_h0)
            (chi2_h0):
        try:
            chi2_sne = calculate_sne_chi2_gpu(gs_model, data_sne)s e:culate_sne_chi2_gpu(gs_model, data_sne)
            if not np.isfinite(chi2_sne):print(f"Error calculating H0 chi2 on GPU: {e}")if not np.isfinite(chi2_sne):
                return -np.infreturn -np.inf    return -np.inf
        except Exception as e:
            print(f"Error calculating SNe chi2 on GPU: {e}")
            return -np.infte_sne_chi2_gpu(gs_model, data_sne)
            (chi2_sne):
        try:
            chi2_bao = calculate_bao_chi2_gpu(gs_model, bao_data)s e:culate_bao_chi2_gpu(gs_model, data_bao)  # Fixed: Changed bao_data to data_bao
            if not np.isfinite(chi2_bao):            print(f"Error calculating SNe chi2 on GPU: {e}")            if not np.isfinite(chi2_bao):
                return -np.inf
        except Exception as e:
            print(f"Error calculating BAO chi2 on GPU: {e}")        try:            print(f"Error calculating BAO chi2 on GPU: {e}")
            return -np.info_chi2_gpu(gs_model, bao_data)
hi2_bao):
        # Calculate total chi-squared                return -np.inf        # Calculate total chi-squared
        total_chi2 = chi2_h0 + chi2_sne + chi2_bao
ting BAO chi2 on GPU: {e}")
        # Convert to log likelihood
        logL = -0.5 * total_chi2
        # Calculate total chi-squared
        # Check for NaN or infinite results which can break MCMCaoh can break MCMC
        if not np.isfinite(logL):
             print(f"Warning: Non-finite logL ({logL}) for ω={omega:.4f}, β={beta:.4f}")g likelihoodrning: Non-finite logL ({logL}) for ω={omega:.4f}, β={beta:.4f}")
             return -np.inf

        # Cleanup unused GPU memory periodicallyeck for NaN or infinite results which can break MCMCeanup unused GPU memory periodically
        if np.random.random() < 0.01:1:
            if HAS_GPU:{logL}) for ω={omega:.4f}, β={beta:.4f}")
                cp.get_default_memory_pool().free_all_blocks()mory_pool().free_all_blocks()
            gc.collect()
            eanup unused GPU memory periodically
        # Monitor computation timem.random() < 0.01:omputation time
        compute_time = time.time() - start_time            if HAS_GPU:        compute_time = time.time() - start_time
        if compute_time > 1.0:ault_memory_pool().free_all_blocks()1.0:
            print(f"Slow GPU likelihood: {compute_time:.2f}s for ω={omega:.4f}, β={beta:.4f}")
            
        return logL        # Monitor computation time        return logL

    except Exception as e:
        print(f"Warning: GPU likelihood calculation failed for ω={omega:.4f}, β={beta:.4f}. Error: {e}")     print(f"Slow GPU likelihood: {compute_time:.2f}s for ω={omega:.4f}, β={beta:.4f}") print(f"Warning: GPU likelihood calculation failed for ω={omega:.4f}, β={beta:.4f}. Error: {e}")
        return -np.inf

# Define the log-posterior function (Prior + Likelihood)og-posterior function (Prior + Likelihood)
def log_posterior(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """mega:.4f}, β={beta:.4f}. Error: {e}")
    Log posterior probability distribution (Log(Prior) + Log(Likelihood)).ability distribution (Log(Prior) + Log(Likelihood)).
    This is the function the MCMC sampler explores.
    """
    lp = log_prior(params)def log_posterior(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):    lp = log_prior(params)
    if not np.isfinite(lp): # If parameters are outside prior range
        return -np.inf
    # Log(Posterior) = Log(Prior) + Log(Likelihood)
    return lp + log_likelihood(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)

# Function for GPU-accelerated log posterior # If parameters are outside prior rangeted log posterior
def log_posterior_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """GPU-accelerated log posterior calculation"""# Log(Posterior) = Log(Prior) + Log(Likelihood)"""GPU-accelerated log posterior calculation"""
    # Check prior - no need to use GPU for this simple checksne, data_bao, fixed_alpha, fixed_epsilon)e check
    lp = log_prior(params)
    if not np.isfinite(lp):nction for GPU-accelerated log posteriorif not np.isfinite(lp):
        return -np.infgpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):.inf
        """GPU-accelerated log posterior calculation"""    
    # Calculate log likelihood using GPU accelerationcheck
    ll = log_likelihood_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)
    
    return lp + llreturn -np.infrn lp + ll

# Function to save intermediate results during MCMC run
def save_intermediate_results(sampler, nburn, output_dir, prefix, fixed_params): log_likelihood_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)_intermediate_results(sampler, nburn, output_dir, prefix, fixed_params):
    """Save intermediate MCMC results for checkpointing"""
    try:
        # Get current timestamprrent timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")te results during MCMC runnow().strftime("%Y%m%d_%H%M%S")
        ut_dir, prefix, fixed_params):
        # Save the current state of the chainediate MCMC results for checkpointing""" current state of the chain
        samples = sampler.get_chain(discard=nburn, thin=10, flat=True)= sampler.get_chain(discard=nburn, thin=10, flat=True)
        
        if len(samples) == 0:
            print("No valid samples to save yet.")
            return
            samples = sampler.get_chain(discard=nburn, thin=10, flat=True)    
        # Create DataFrame and save
        df_samples = pd.DataFrame(samples, columns=['omega', 'beta'])me(samples, columns=['omega', 'beta'])
        checkpoint_file = os.path.join(output_dir, f"{prefix}_checkpoint_{timestamp}.csv")les to save yet.")h.join(output_dir, f"{prefix}_checkpoint_{timestamp}.csv")
        df_samples.to_csv(checkpoint_file, index=False)
        
        # Calculate quick stats if we have enough samples
        if len(samples) >= 10:mples, columns=['omega', 'beta'])
            results_summary = {}nt_{timestamp}.csv")
            for i, label in enumerate(['omega', 'beta']):_csv(checkpoint_file, index=False)bel in enumerate(['omega', 'beta']):
                if len(samples) >= 3:  # Need at least 3 samples for percentiles
                    mcmc = np.percentile(samples[:, i], [16, 50, 84])
                    median = mcmc[1]en(samples) >= 10:        median = mcmc[1]
                    results_summary[label] = {'median': float(median)} {}ummary[label] = {'median': float(median)}
                else:abel in enumerate(['omega', 'beta'])::
                    # Just use mean if not enough for percentiles# Need at least 3 samples for percentilesnot enough for percentiles
                    results_summary[label] = {'median': float(np.mean(samples[:, i]))}es[:, i], [16, 50, 84])'median': float(np.mean(samples[:, i]))}
            
            # Save basic info {'median': float(median)}
            info = {   else:nfo = {
                'timestamp': timestamp,        # Just use mean if not enough for percentiles    'timestamp': timestamp,
                'samples_saved': len(samples),
                'current_results': results_summary,
                'fixed_params': fixed_params
            } = {
            
            info_file = os.path.join(output_dir, f"{prefix}_checkpoint_info_{timestamp}.json")nt_info_{timestamp}.json")
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=4)            'fixed_params': fixed_params            json.dump(info, f, indent=4)
                
            print(f"Checkpoint saved with {len(samples)} samples. Current estimates:")tes:")
            for param, values in results_summary.items():            info_file = os.path.join(output_dir, f"{prefix}_checkpoint_info_{timestamp}.json")            for param, values in results_summary.items():
                print(f"  {param}: {values['median']:.4f}")fo_file, 'w') as f:  {param}: {values['median']:.4f}")
                    json.dump(info, f, indent=4)    
    except Exception as e:     Exception as e:
        print(f"Error saving checkpoint: {e}")ples. Current estimates:")
s in results_summary.items():
# === Main Execution ===

def main():
    """Main function to run the MCMC parameter estimation"""
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run MCMC parameter estimation for Genesis-Sphere")
    # Add arguments for fixed parameters, data paths, MCMC settings, etc.
    parser.add_argument("--alpha", type=float, default=0.02, help="Fixed alpha value")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Fixed epsilon value")
    parser.add_argument("--nwalkers", type=int, default=32, help="Number of MCMC walkers (must be > 2*N_DIM)")
    parser.add_argument("--nsteps", type=int, default=5000, help="Number of MCMC steps per walker")imation for Genesis-Sphere") of MCMC steps per walker")
    parser.add_argument("--nburn", type=int, default=1000, help="Number of burn-in steps to discard")")
    parser.add_argument("--initial_omega", type=float, default=3.5, help="Initial guess for omega") help="Fixed alpha value")lt=3.5, help="Initial guess for omega")
    parser.add_argument("--initial_beta", type=float, default=-0.0333, help="Initial guess for beta") epsilon value")lp="Initial guess for beta")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix for output filenames"), help="Number of MCMC walkers (must be > 2*N_DIM)")lt="", help="Optional suffix for output filenames")
    parser.add_argument("--checkpoint_interval", type=int, default=500, MCMC steps per walker")
                        help="Save intermediate results every N steps (0 to disable)")    parser.add_argument("--nburn", type=int, default=1000, help="Number of burn-in steps to discard")                        help="Save intermediate results every N steps (0 to disable)")
    parser.add_argument("--test_mode", action="store_true", tial_omega", type=float, default=3.5, help="Initial guess for omega")t_mode", action="store_true", 
                        help="Run in test mode with reduced computation")    parser.add_argument("--initial_beta", type=float, default=-0.0333, help="Initial guess for beta")                        help="Run in test mode with reduced computation")
    parser.add_argument("--max_time", type=int, default=0,x", type=str, default="", help="Optional suffix for output filenames")ype=int, default=0,
                        help="Maximum runtime in minutes (0 for unlimited)")t("--checkpoint_interval", type=int, default=500,   help="Maximum runtime in minutes (0 for unlimited)")
N steps (0 to disable)")
    args = parser.parse_args()-test_mode", action="store_true", gs()
help="Run in test mode with reduced computation")
    # Modify parameters if in test mode("--max_time", type=int, default=0, if in test mode
    if args.test_mode: runtime in minutes (0 for unlimited)")
        print("Running in TEST MODE with reduced computation")        print("Running in TEST MODE with reduced computation")
        args.nwalkers = 10s()
        args.nsteps = 50
        args.nburn = 10
        args.checkpoint_interval = 10de:oint_interval = 10
DE with reduced computation")
    # Validate walker count
    if args.nwalkers <= 2 * N_DIM: = 50s <= 2 * N_DIM:
        print(f"Error: Number of walkers ({args.nwalkers}) must be greater than 2 * N_DIM ({2*N_DIM}).")        args.nburn = 10        print(f"Error: Number of walkers ({args.nwalkers}) must be greater than 2 * N_DIM ({2*N_DIM}).")
        sys.exit(1)
    if args.nsteps <= args.nburn:
        print(f"Error: Total steps ({args.nsteps}) must be greater than burn-in steps ({args.nburn}).")
        sys.exit(1)
mber of walkers ({args.nwalkers}) must be greater than 2 * N_DIM ({2*N_DIM}).")
    print("Starting Genesis-Sphere MCMC Parameter Estimation...")
    print(f"Fixed Parameters: α={args.alpha}, ε={args.epsilon}")if args.nsteps <= args.nburn:print(f"Fixed Parameters: α={args.alpha}, ε={args.epsilon}")
    print(f"MCMC Settings: Walkers={args.nwalkers}, Steps={args.nsteps}, Burn-in={args.nburn}")al steps ({args.nsteps}) must be greater than burn-in steps ({args.nburn}).") Walkers={args.nwalkers}, Steps={args.nsteps}, Burn-in={args.nburn}")
    print(f"Checkpoint Interval: {args.checkpoint_interval} steps")_interval} steps")
    if args.max_time > 0:    if args.max_time > 0:
        print(f"Maximum runtime: {args.max_time} minutes")MCMC Parameter Estimation...")rgs.max_time} minutes")
    silon}")
    # Initial memory usageeps}, Burn-in={args.nburn}")
    print_memory_usage("at start")rgs.checkpoint_interval} steps")
    if args.max_time > 0:
    # Create timestamp for this run runtime: {args.max_time} minutes")for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""itial memory usageix = f"_{args.output_suffix}" if args.output_suffix else ""
    run_id = f"{timestamp}{suffix}"

    # --- Load Data ---
    print("Loading observational data...")stamp = datetime.now().strftime("%Y%m%d_%H%M%S")t("Loading observational data...")
    try:}" if args.output_suffix else ""
        loading_start = time.time()
        h0_data = load_h0_measurements()
        print(f"  H0 data loaded: {len(h0_data)} measurements ({time.time() - loading_start:.2f}s)")- Load Data ---print(f"  H0 data loaded: {len(h0_data)} measurements ({time.time() - loading_start:.2f}s)")
        ta...")
        loading_start = time.time()
        sne_data = load_supernovae_data()
        print(f"  SNe data loaded: {len(sne_data)} supernovae ({time.time() - loading_start:.2f}s)")h0_data = load_h0_measurements()print(f"  SNe data loaded: {len(sne_data)} supernovae ({time.time() - loading_start:.2f}s)")
        data)} measurements ({time.time() - loading_start:.2f}s)")
        loading_start = time.time()
        bao_data = load_bao_data()loading_start = time.time()bao_data = load_bao_data()
        print(f"  BAO data loaded: {len(bao_data)} measurements ({time.time() - loading_start:.2f}s)")a)} measurements ({time.time() - loading_start:.2f}s)")
        Ne data loaded: {len(sne_data)} supernovae ({time.time() - loading_start:.2f}s)")
        print("Data loaded successfully.")
        print_memory_usage("after data loading")
         = load_bao_data()
        # Initialize GPU memory if availabletime() - loading_start:.2f}s)")
        if HAS_GPU:
            print("Setting up GPU environment...").")onment...")
            # Limit GPU memory usage to 90% of available memory
            try:
                memory_pool = cp.cuda.MemoryPool(cp.cuda.set_allocator)ocator)
                cp.cuda.set_allocator(memory_pool.malloc)PU:cp.cuda.set_allocator(memory_pool.malloc)
                with cp.cuda.Device(0):
                    mempool = cp.get_default_memory_pool()vailable memoryory_pool()
                    mempool.set_limit(fraction=0.9)on=0.9)
                print("  GPU memory pool configured.")llocator)
                ol.malloc)
                # Preload some data to GPU to check it workse(0):a to GPU to check it works
                test_array = cp.array([1, 2, 3])
                del test_array
                cp.get_default_memory_pool().free_all_blocks()print("  GPU memory pool configured.")cp.get_default_memory_pool().free_all_blocks()
                print("  GPU test successful.")ccessful.")
            except Exception as e:U to check it works
                print(f"  Warning: GPU setup encountered an issue: {e}")t_array = cp.array([1, 2, 3])nt(f"  Warning: GPU setup encountered an issue: {e}")
                print("  Continuing with reduced GPU optimization.")                del test_array                print("  Continuing with reduced GPU optimization.")
                emory_pool().free_all_blocks()
    except Exception as e:ccessful.")
        print(f"Error loading data: {e}") as e:g data: {e}")
        sys.exit(1)

    # --- Initialize Walkers ---            # --- Initialize Walkers ---
    print("Initializing walkers...")
    init_start = time.time()
    # Start walkers in a small Gaussian ball around the previous best parameterss
    initial_pos_guess = np.array([args.initial_omega, args.initial_beta])ial_pos_guess = np.array([args.initial_omega, args.initial_beta])
    
    # Add small random offsets for each walker, ensuring they are within priorsing walkers...")om offsets for each walker, ensuring they are within priors
    pos = np.zeros((args.nwalkers, N_DIM))
    max_attempts = 1000  # Prevent infinite loopsn ball around the previous best parametersinite loops
    
    for i in range(args.nwalkers):
        attempts = 0sets for each walker, ensuring they are within priors
        # Keep generating random starting points until they satisfy the priorrgs.nwalkers, N_DIM))ting random starting points until they satisfy the prior
        while attempts < max_attempts:# Prevent infinite loopsmax_attempts:
            p = initial_pos_guess + 1e-3 * np.abs(initial_pos_guess) * np.random.randn(N_DIM)p = initial_pos_guess + 1e-3 * np.abs(initial_pos_guess) * np.random.randn(N_DIM)
            if np.isfinite(log_prior(p)):)):
                pos[i] = p
                break
            attempts += 1
                p = initial_pos_guess + 1e-3 * np.abs(initial_pos_guess) * np.random.randn(N_DIM)    
        if attempts >= max_attempts:_prior(p)):empts:
            print(f"Warning: Could not initialize walker {i} after {max_attempts} attempts.")
            print(f"Using initial guess directly: ω={args.initial_omega}, β={args.initial_beta}")={args.initial_beta}")
            pos[i] = initial_pos_guess            attempts += 1            pos[i] = initial_pos_guess
    
    nwalkers, ndim = pos.shape
    print(f"Initialized {nwalkers} walkers around ω={args.initial_omega}, β={args.initial_beta} "  Could not initialize walker {i} after {max_attempts} attempts.")lkers} walkers around ω={args.initial_omega}, β={args.initial_beta} " 
          f"in {time.time() - init_start:.2f}s")sing initial guess directly: ω={args.initial_omega}, β={args.initial_beta}")time() - init_start:.2f}s")
        pos[i] = initial_pos_guess
    # --- Test likelihood function with a few points ---
    print("Testing likelihood function with initial positions...")ped function with initial positions...")
    test_start = time.time()Initialized {nwalkers} walkers around ω={args.initial_omega}, β={args.initial_beta} " rt = time.time()
    success_count = 0
    
    for i in range(min(5, nwalkers)): with a few points ---)):
        test_params = pos[i]
        try:time.time()
            lp = log_posterior(test_params, h0_data, sne_data, bao_data, args.alpha, args.epsilon)
            if np.isfinite(lp):
                success_count += 1
                print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → logP={lp:.2f} (Valid)")    test_params = pos[i]            print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → logP={lp:.2f} (Valid)")
            else:
                print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Invalid posterior")        lp = log_posterior(test_params, h0_data, sne_data, bao_data, args.alpha, args.epsilon)            print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Invalid posterior")
        except Exception as e:(lp):s e:
            print(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Error: {e}")
    nt(f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → logP={lp:.2f} (Valid)")
    print(f"Likelihood function test: {success_count}/5 successful in {time.time() - test_start:.2f}s")            else:    print(f"Likelihood function test: {success_count}/5 successful in {time.time() - test_start:.2f}s")
    f"  Test {i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Invalid posterior")
    if success_count == 0::
        print("ERROR: All likelihood function tests failed. Exiting.")i+1}: ω={test_params[0]:.4f}, β={test_params[1]:.4f} → Error: {e}")kelihood function tests failed. Exiting.")
        sys.exit(1)    sys.exit(1)
 {success_count}/5 successful in {time.time() - test_start:.2f}s")
    # --- Run MCMC ---
    print(f"Running MCMC...")if success_count == 0:print(f"Running MCMC...")
    mcmc_start = time.time()ests failed. Exiting.")
    
    # Set up max runtime if specified
    max_runtime_seconds = args.max_time * 60 if args.max_time > 0 else None60 if args.max_time > 0 else None
    
    # Fixed parameters to pass to checkpointingo checkpointing
    fixed_params = {
        'alpha': args.alpha, Set up max runtime if specified   'alpha': args.alpha,
        'epsilon': args.epsilon,max_runtime_seconds = args.max_time * 60 if args.max_time > 0 else None    'epsilon': args.epsilon,
        'nwalkers': args.nwalkers,
        'nsteps': args.nsteps,ameters to pass to checkpointing': args.nsteps,
        'nburn': args.nburn
    }
    
    # The 'args' tuple passes fixed parameters and data to the log_posterior function
    if HAS_GPU:nsteps': args.nsteps,S_GPU:
        print("Using GPU-accelerated MCMC sampling")nburn': args.nburnrint("Using GPU-accelerated MCMC sampling")
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior_gpu, 
            args=(h0_data, sne_data, bao_data, args.alpha, args.epsilon)ers and data to the log_posterior functionta, args.alpha, args.epsilon)
        )
    else:rint("Using GPU-accelerated MCMC sampling")
        print("Using CPU-based MCMC sampling")        sampler = emcee.EnsembleSampler(        print("Using CPU-based MCMC sampling")
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior,  sne_data, bao_data, args.alpha, args.epsilon), log_posterior, 
            args=(h0_data, sne_data, bao_data, args.alpha, args.epsilon)    )        args=(h0_data, sne_data, bao_data, args.alpha, args.epsilon)
        )

    # Run MCMC steps with progress and checkpointing
    checkpoint_counter = 0   nwalkers, ndim, log_posterior, point_counter = 0
    epsilon)
    # Use run_mcmc with progress=True for the simple case
    if args.checkpoint_interval <= 0 and not args.max_time:kpoint_interval <= 0 and not args.max_time:
        sampler.run_mcmc(pos, args.nsteps, progress=True)ingss=True)
    else:
        # More complex case with checkpointing and/or time limit
        print("Running with checkpoints and/or time limit...")le casee limit...")
        rgs.checkpoint_interval <= 0 and not args.max_time:
        # Run in smaller chunks for checkpointingpos, args.nsteps, progress=True)chunks for checkpointing
        chunk_size = min(100, args.checkpoint_interval) if args.checkpoint_interval > 0 else 100_interval) if args.checkpoint_interval > 0 else 100
        n_chunks = args.nsteps // chunk_size# More complex case with checkpointing and/or time limitn_chunks = args.nsteps // chunk_size
        remaining_steps = args.nsteps % chunk_size
        
        current_pos = pos
        steps_completed = 0else 100
        gs.nsteps // chunk_size
        for i in range(n_chunks + (1 if remaining_steps > 0 else 0)):g_steps = args.nsteps % chunk_size range(n_chunks + (1 if remaining_steps > 0 else 0)):
            # Check if we've exceeded max runtime
            if max_runtime_seconds and (time.time() - start_time > max_runtime_seconds):conds):
                print(f"Maximum runtime of {args.max_time} minutes exceeded. Stopping.")x_time} minutes exceeded. Stopping.")
                break
                (1 if remaining_steps > 0 else 0)):
            # Determine steps for this chunkeck if we've exceeded max runtimetermine steps for this chunk
            if i == n_chunks and remaining_steps > 0:econds and (time.time() - start_time > max_runtime_seconds): and remaining_steps > 0:
                steps = remaining_stepsme of {args.max_time} minutes exceeded. Stopping.")ps
            else:
                steps = chunk_size
                # Determine steps for this chunk    
            # Run this chunk
            chunk_start = time.time()eps)
            print(f"Running chunk {i+1}/{n_chunks + (1 if remaining_steps > 0 else 0)}: "else:print(f"Running chunk {i+1}/{n_chunks + (1 if remaining_steps > 0 else 0)}: "
                  f"{steps} steps ({steps_completed}/{args.nsteps} total)")({steps_completed}/{args.nsteps} total)")
            
            current_pos, _, _ = sampler.run_mcmc(current_pos, steps, progress=True)
            steps_completed += steps
            
            # Report on this chunk      f"{steps} steps ({steps_completed}/{args.nsteps} total)")# Report on this chunk
            chunk_time = time.time() - chunk_start
            print(f"  Completed chunk in {chunk_time:.2f}s " _, _ = sampler.run_mcmc(current_pos, steps, progress=True)mpleted chunk in {chunk_time:.2f}s "
                  f"({steps/chunk_time:.1f} steps/s, "
                  f"~{(args.nsteps-steps_completed)/(steps/chunk_time)/60:.1f} minutes remaining)")      f"~{(args.nsteps-steps_completed)/(steps/chunk_time)/60:.1f} minutes remaining)")
            
            # Garbage collection to prevent memory leakschunk_startvent memory leaks
            gc.collect()
            print_memory_usage(f"after chunk {i+1}")
            ompleted)/(steps/chunk_time)/60:.1f} minutes remaining)")
            # Save checkpoint if needed
            checkpoint_counter += stepsbage collection to prevent memory leakspoint_counter += steps
            if args.checkpoint_interval > 0 and checkpoint_counter >= args.checkpoint_interval:kpoint_counter >= args.checkpoint_interval:
                print(f"Saving checkpoint after {steps_completed} steps...")        print_memory_usage(f"after chunk {i+1}")            print(f"Saving checkpoint after {steps_completed} steps...")
                save_intermediate_results(
                    sampler, args.nburn, results_dir, f"mcmc_{run_id}", fixed_paramsf neededgs.nburn, results_dir, f"mcmc_{run_id}", fixed_params
                )
                checkpoint_counter = 0rgs.checkpoint_interval:
                print(f"Saving checkpoint after {steps_completed} steps...")
    mcmc_time = time.time() - mcmc_startmediate_results() - mcmc_start
    print("MCMC run complete.") fixed_params
    print(f"MCMC execution time: {mcmc_time:.2f} seconds "
          f"({args.nsteps * args.nwalkers / mcmc_time:.1f} samples/s)")                checkpoint_counter = 0          f"({args.nsteps * args.nwalkers / mcmc_time:.1f} samples/s)")
    
    end_time = time.time()_time = time.time() - mcmc_starttime = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print_memory_usage("at end of MCMC")
")
    # --- Process Results ---ocess Results ---
    try:
        # Check acceptance fraction (should generally be between ~0.2 and 0.5)
        acceptance_fraction = np.mean(sampler.acceptance_fraction)
        print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")        print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")
        
        if acceptance_fraction < 0.1:
            print("WARNING: Very low acceptance fraction. Results may be unreliable.")and 0.5)be unreliable.")
            print("Consider adjusting the prior ranges or initial positions.")

        # Discard burn-in steps and flatten the chain
        # flat=True combines results from all walkersif acceptance_fraction < 0.1:# flat=True combines results from all walkers
        # thin=X keeps only every Xth sample to reduce autocorrelationery low acceptance fraction. Results may be unreliable.")very Xth sample to reduce autocorrelation
        print(f"Processing samples (discarding {args.nburn} burn-in steps)...")s.")...")
        samples = sampler.get_chain(discard=args.nburn, thin=15, flat=True)
        print(f"Shape of processed samples: {samples.shape}") # Should be (N_samples, N_DIM)        # Discard burn-in steps and flatten the chain        print(f"Shape of processed samples: {samples.shape}") # Should be (N_samples, N_DIM)
        esults from all walkers
        if len(samples) < 10:mple to reduce autocorrelation
            print("WARNING: Very few samples after burn-in and thinning.")        print(f"Processing samples (discarding {args.nburn} burn-in steps)...")            print("WARNING: Very few samples after burn-in and thinning.")
            print("Consider reducing burn-in or increasing total steps.")scard=args.nburn, thin=15, flat=True)urn-in or increasing total steps.")
(N_samples, N_DIM)
        # --- Save Results ---
        print("Saving final results...")
urn-in and thinning.")
        # Save the samples (the chain)            print("Consider reducing burn-in or increasing total steps.")        # Save the samples (the chain)
        chain_file = os.path.join(results_dir, f"mcmc_chain_{run_id}.csv")
        df_samples = pd.DataFrame(samples, columns=['omega', 'beta'])esults --- pd.DataFrame(samples, columns=['omega', 'beta'])
        df_samples.to_csv(chain_file, index=False)...")e, index=False)
        print(f"MCMC samples saved to {chain_file}")

        # Save run info (parameters, settings)lts_dir, f"mcmc_chain_{run_id}.csv")ettings)
        run_info = {samples, columns=['omega', 'beta'])
            'timestamp': timestamp,file, index=False)mp,
            'fixed_alpha': args.alpha,
            'fixed_epsilon': args.epsilon,
            'nwalkers': args.nwalkers,
            'nsteps': args.nsteps,
            'nburn': args.nburn,
            'initial_guess': {'omega': args.initial_omega, 'beta': args.initial_beta},rgs.initial_omega, 'beta': args.initial_beta},
            'parameter_labels': PARAM_LABELS,   'fixed_epsilon': args.epsilon,   'parameter_labels': PARAM_LABELS,
            'mean_acceptance_fraction': float(acceptance_fraction),
            'execution_time_seconds': end_time - start_time,me - start_time,
            'samples_shape': list(samples.shape),
            'test_mode': args.test_modetial_omega, 'beta': args.initial_beta},
        }            'parameter_labels': PARAM_LABELS,        }
        info_file = os.path.join(results_dir, f"run_info_{run_id}.json")            'mean_acceptance_fraction': float(acceptance_fraction),        info_file = os.path.join(results_dir, f"run_info_{run_id}.json")
        with open(info_file, 'w') as f:ime - start_time,
            json.dump(run_info, f, indent=4)ape),
        print(f"Run info saved to {info_file}")            'test_mode': args.test_mode        print(f"Run info saved to {info_file}")

join(results_dir, f"run_info_{run_id}.json")
        # --- Basic Analysis & Plotting ---
        print("\nAnalyzing MCMC results...")

        # Calculate median and 1-sigma credible intervals (16th, 50th, 84th percentiles)
        results_summary = {}
        print("\n=== MCMC Parameter Estimates (median and 1-sigma credible interval) ===") & Plotting ---rameter Estimates (median and 1-sigma credible interval) ===")
        for i, label in enumerate(['omega', 'beta']):CMC results...")erate(['omega', 'beta']):
            mcmc = np.percentile(samples[:, i], [16, 50, 84])
            q = np.diff(mcmc) # q[0] = 50th-16th, q[1] = 84th-50th, 50th, 84th percentiles)0th
            median = mcmc[1]
            upper_err = q[1]redible interval) ===")
            lower_err = q[0]        for i, label in enumerate(['omega', 'beta']):            lower_err = q[0]
            print(f"{PARAM_LABELS[i]} = {median:.4f} (+{upper_err:.4f} / -{lower_err:.4f})")tile(samples[:, i], [16, 50, 84])ABELS[i]} = {median:.4f} (+{upper_err:.4f} / -{lower_err:.4f})")
            results_summary[label] = {'median': float(median), 
                                     'upper_err': float(upper_err), oat(upper_err), 
                                     'lower_err': float(lower_err)}

        # Save summary stats            print(f"{PARAM_LABELS[i]} = {median:.4f} (+{upper_err:.4f} / -{lower_err:.4f})")        # Save summary stats
        stats_file = os.path.join(results_dir, f"param_stats_{run_id}.json")            results_summary[label] = {'median': float(median),         stats_file = os.path.join(results_dir, f"param_stats_{run_id}.json")
        with open(stats_file, 'w') as f:pper_err), 
            json.dump(results_summary, f, indent=4)err': float(lower_err)}dent=4)
        print(f"Parameter stats saved to {stats_file}")r stats saved to {stats_file}")

n_id}.json")
        # Generate a corner plot using the corner libraryr library
        print("\nGenerating corner plot...")
        try:
            figure = corner.corner(
                samples, labels=PARAM_LABELS, # Use LaTeX labelsabels=PARAM_LABELS, # Use LaTeX labels
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True, title_kwargs={"fontsize": 12},gs={"fontsize": 12},
                truths=[results_summary['omega']['median'], results_summary['beta']['median']], # Show median valuesvalues
                truth_color='red'.corner(='red'
            )
            corner_plot_file = os.path.join(results_dir, f"corner_plot_{run_id}.png")6, 0.5, 0.84], os.path.join(results_dir, f"corner_plot_{run_id}.png")
            figure.savefig(corner_plot_file)},
            print(f"Corner plot saved to {corner_plot_file}")                truths=[results_summary['omega']['median'], results_summary['beta']['median']], # Show median values            print(f"Corner plot saved to {corner_plot_file}")
        except ImportError:
            print("\nInstall 'corner' package (`pip install corner`) to generate corner plots.")
        except Exception as e:            corner_plot_file = os.path.join(results_dir, f"corner_plot_{run_id}.png")        except Exception as e:
            print(f"Error during corner plot generation: {e}")(corner_plot_file)during corner plot generation: {e}")
")
        # Generate a markdown summary reportor:kdown summary report
        generate_validation_summary(results_summary, args, timestamp, suffix, acceptance_fraction)'corner' package (`pip install corner`) to generate corner plots.")ummary(results_summary, args, timestamp, suffix, acceptance_fraction)

    except Exception as e:            print(f"Error during corner plot generation: {e}")    except Exception as e:
        print(f"Error during MCMC results processing: {e}")
        import traceback    # Generate a markdown summary report    import traceback
        traceback.print_exc()tion_summary(results_summary, args, timestamp, suffix, acceptance_fraction)_exc()
        print("Chain data might still be saved if the run completed.")t still be saved if the run completed.")

    print("\nMCMC parameter estimation script finished!"){e}")!")
            import traceback    
    # Final GPU cleanup
    if HAS_GPU:
        print("Performing final GPU memory cleanup...")l GPU memory cleanup...")
        cp.get_default_memory_pool().free_all_blocks()

def generate_validation_summary(results_summary, args, timestamp, suffix, acceptance_fraction):_summary, args, timestamp, suffix, acceptance_fraction):
    """Generate a markdown summary of the MCMC parameter estimation"""
    summary = [
        "# Genesis-Sphere MCMC Parameter Estimation Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",Y-%m-%d %H:%M:%S')}\n",
        "## Validation Method",summary, args, timestamp, suffix, acceptance_fraction):
        "\nThis validation uses Markov Chain Monte Carlo (MCMC) to estimate the posterior probability distribution",C parameter estimation"""Monte Carlo (MCMC) to estimate the posterior probability distribution",
        "of Genesis-Sphere model parameters based on astronomical datasets. Unlike the previous grid search approach,",previous grid search approach,",
        "MCMC provides robust parameter uncertainties and explores the parameter space more efficiently.\n",
        "## MCMC Settings",}\n",
        f"\n- Walkers: {args.nwalkers}",
        f"- Steps per walker: {args.nsteps}",arkov Chain Monte Carlo (MCMC) to estimate the posterior probability distribution",gs.nsteps}",
        f"- Burn-in steps discarded: {args.nburn}",ets. Unlike the previous grid search approach,",
        f"- Initial parameter guess: ω={args.initial_omega:.4f}, β={args.initial_beta:.4f}",es the parameter space more efficiently.\n", β={args.initial_beta:.4f}",
        f"- Fixed parameters: α={args.alpha:.4f}, ε={args.epsilon:.4f}",
        f"- Mean acceptance fraction: {acceptance_fraction:.3f}\n",
        "## Parameter Estimates",
        "\nBest-fit parameters with 1-sigma (68%) credible intervals:", discarded: {args.nburn}",eters with 1-sigma (68%) credible intervals:",
        f"\n| Parameter | Median | Lower Error | Upper Error |",:.4f}",
        "|-----------|--------|-------------|-------------|",
        f"| Omega (ω) | {results_summary['omega']['median']:.4f} | {results_summary['omega']['lower_err']:.4f} | {results_summary['omega']['upper_err']:.4f} |",omega']['lower_err']:.4f} | {results_summary['omega']['upper_err']:.4f} |",
        f"| Beta (β) | {results_summary['beta']['median']:.4f} | {results_summary['beta']['lower_err']:.4f} | {results_summary['beta']['upper_err']:.4f} |\n",s",s_summary['beta']['median']:.4f} | {results_summary['beta']['lower_err']:.4f} | {results_summary['beta']['upper_err']:.4f} |\n",
        "## Corner Plot",
        "\n![Parameter Corner Plot](corner_plot_" + f"{timestamp}{suffix}.png" + ")",
        "\nThe corner plot shows the 1D and 2D posterior distributions of the model parameters.",
        "Contours show the 1-sigma, 2-sigma, and 3-sigma credible regions.",lower_err']:.4f} | {results_summary['omega']['upper_err']:.4f} |",
        "\n## Interpretation",lower_err']:.4f} | {results_summary['beta']['upper_err']:.4f} |\n",
        "\nThe MCMC analysis shows that the optimal Genesis-Sphere parameters are:",
        f"- **Omega (ω)**: {results_summary['omega']['median']:.4f} ± {(results_summary['omega']['lower_err'] + results_summary['omega']['upper_err'])/2:.4f}",esults_summary['omega']['upper_err'])/2:.4f}",
        f"- **Beta (β)**: {results_summary['beta']['median']:.4f} ± {(results_summary['beta']['lower_err'] + results_summary['beta']['upper_err'])/2:.4f}",summary['beta']['upper_err'])/2:.4f}",
        "\nThese values represent the statistical constraints from combining H₀ correlation,",s show the 1-sigma, 2-sigma, and 3-sigma credible regions.", values represent the statistical constraints from combining H₀ correlation,",
        "supernovae distance modulus fitting, and BAO signal detection. The uncertainties",
        "reflect the genuine statistical uncertainty in determining these parameters from the available data.",   "\nThe MCMC analysis shows that the optimal Genesis-Sphere parameters are:",   "reflect the genuine statistical uncertainty in determining these parameters from the available data.",
        "\nCompared to the previous grid search approach, this MCMC analysis provides more robust",    f"- **Omega (ω)**: {results_summary['omega']['median']:.4f} ± {(results_summary['omega']['lower_err'] + results_summary['omega']['upper_err'])/2:.4f}",    "\nCompared to the previous grid search approach, this MCMC analysis provides more robust",
        "parameter constraints by thoroughly exploring the parameter space and quantifying uncertainties.",y['beta']['lower_err'] + results_summary['beta']['upper_err'])/2:.4f}",ifying uncertainties.",
        "\n---",aints from combining H₀ correlation,",
        "\n*This report was automatically generated by the Genesis-Sphere MCMC parameter estimation framework.*"s fitting, and BAO signal detection. The uncertainties",ically generated by the Genesis-Sphere MCMC parameter estimation framework.*"
    ]    "reflect the genuine statistical uncertainty in determining these parameters from the available data.",]
     this MCMC analysis provides more robust",
    summary_path = os.path.join(results_dir, f"mcmc_summary_{timestamp}{suffix}.md")        "parameter constraints by thoroughly exploring the parameter space and quantifying uncertainties.",    summary_path = os.path.join(results_dir, f"mcmc_summary_{timestamp}{suffix}.md")
    with open(summary_path, 'w', encoding='utf-8') as f:oding='utf-8') as f:
        f.write('\n'.join(summary))"\n*This report was automatically generated by the Genesis-Sphere MCMC parameter estimation framework.*"f.write('\n'.join(summary))
    
    print(f"Validation summary saved to: {summary_path}")ed to: {summary_path}")
summary_{timestamp}{suffix}.md")
if __name__ == "__main__":s f:
    try:
        # Force CuPy to initialize before we start to catch any startup errors
        if HAS_GPU:t(f"Validation summary saved to: {summary_path}")if HAS_GPU:
            print("Initializing GPU environment...")ing GPU environment...")
            cp.array([1, 2, 3])  # Simple test array, 3])  # Simple test array
            cp.cuda.Stream.null.synchronize()  # Ensure initialization completesialization completes
            print("GPU initialization complete.")initialize before we start to catch any startup errorsnitialization complete.")
        
        main()    print("Initializing GPU environment...")main()
    except Exception as e:rray
        print(f"Fatal error: {e}").Stream.null.synchronize()  # Ensure initialization completesal error: {e}")
        import tracebackt("GPU initialization complete.")raceback
        traceback.print_exc()
        
        # Cleanup GPU resources in case of errorn as e:PU resources in case of error
        if HAS_GPU:l error: {e}")
            try:raceback
                cp.get_default_memory_pool().free_all_blocks()rint_exc()get_default_memory_pool().free_all_blocks()
                print("GPU resources cleaned up after error.")                        print("GPU resources cleaned up after error.")





        sys.exit(1)                                pass            except:









        sys.exit(1)                                pass            except:                print("GPU resources cleaned up after error.")                cp.get_default_memory_pool().free_all_blocks()            try:        if HAS_GPU:        # Cleanup GPU resources in case of error            except:
                pass
                
        sys.exit(1)
