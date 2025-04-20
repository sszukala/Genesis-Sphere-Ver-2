"""
Example implementation of GPU-accelerated likelihood calculation for Genesis-Sphere model.
This demonstrates how to use CuPy to accelerate chi-squared calculations.

Use this as a reference for updating your parameter_sweep_validation.py file.
"""

# Try to import CuPy for GPU acceleration, fall back to NumPy if not available
try:
    import cupy as cp
    HAS_GPU = True
    print("CuPy successfully imported. Using GPU acceleration.")
except ImportError:
    import numpy as np
    cp = np  # Use NumPy as a fallback
    HAS_GPU = False
    print("CuPy not found. Using CPU with NumPy instead.")

import time
import pandas as pd

# Helper functions for GPU data transfer
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

# Example class mimicking your GenesisSphereModel
class GenesisSphereModel:
    def __init__(self, alpha=0.02, beta=1.2, omega=2.0, epsilon=0.1):
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.epsilon = epsilon

# GPU-accelerated prediction functions
def gs_predicted_h0_variations_gpu(gs_model, years_gpu):
    """GPU-accelerated version of the H0 prediction function"""
    # Get the base H0 value (typically around 70 km/s/Mpc)
    h0_base = 70.0
    
    # Get Genesis-Sphere parameters
    omega = gs_model.omega
    alpha = gs_model.alpha
    beta = gs_model.beta
    epsilon = gs_model.epsilon
    
    # Normalized time values (map years to model time)
    t = (years_gpu - 2000.0) / 100.0  # Example: center at year 2000
    
    # Calculate time-density and temporal flow on GPU
    sin_term = cp.sin(omega * t)
    rho = (1.0 / (1.0 + sin_term**2)) * (1.0 + alpha * t**2)
    tf = 1.0 / (1.0 + beta * (cp.abs(t) + epsilon))
    
    # Calculate H0 variations based on Genesis-Sphere model
    h0_pred = h0_base * (1.0 + 0.1 * cp.sin(omega * t)) * (1.0 + 0.05 * rho) / cp.sqrt(tf)
    
    return h0_pred

def gs_predicted_distance_modulus_gpu(gs_model, redshifts_gpu):
    """GPU-accelerated version of the distance modulus prediction function"""
    # Get Genesis-Sphere parameters
    omega = gs_model.omega
    alpha = gs_model.alpha
    beta = gs_model.beta
    epsilon = gs_model.epsilon
    
    # Convert redshift to time in the model (simplified example)
    t = 13.8 - 13.8 / (1.0 + redshifts_gpu)  # Approximate age of universe minus lookback time
    
    # Calculate time-density and temporal flow on GPU
    sin_term = cp.sin(omega * t)
    rho = (1.0 / (1.0 + sin_term**2)) * (1.0 + alpha * t**2)
    tf = 1.0 / (1.0 + beta * (cp.abs(t) + epsilon))
    
    # Hubble distance (simplified)
    d_H = 3000.0  # Mpc
    
    # Calculate distance modulus based on Genesis-Sphere model
    # Using simplified luminosity distance formula
    d_L = d_H * (1.0 + redshifts_gpu) * cp.sqrt(rho / tf)
    mu = 5.0 * cp.log10(d_L) + 25.0
    
    return mu

def gs_predicted_bao_feature_gpu(gs_model, redshifts_gpu):
    """GPU-accelerated version of the BAO prediction function"""
    # Get Genesis-Sphere parameters
    omega = gs_model.omega
    alpha = gs_model.alpha
    beta = gs_model.beta
    epsilon = gs_model.epsilon
    
    # Convert redshift to time in the model (simplified)
    t = 13.8 - 13.8 / (1.0 + redshifts_gpu)
    
    # Calculate time-density and temporal flow on GPU
    sin_term = cp.sin(omega * t)
    rho = (1.0 / (1.0 + sin_term**2)) * (1.0 + alpha * t**2)
    tf = 1.0 / (1.0 + beta * (cp.abs(t) + epsilon))
    
    # Base sound horizon (rd) value
    rd_base = 147.0  # Mpc
    
    # Calculate BAO scale based on Genesis-Sphere model
    rd = rd_base * cp.sqrt(rho) * (1.0 + 0.15 * (1.0 - tf)) * (1.0 + 0.1 * cp.sin(omega * t))
    
    return rd

# GPU-accelerated chi-squared calculation functions
def calculate_h0_chi2_gpu(gs_model, h0_data):
    """Calculate proper chi-squared for H0 measurements with GPU acceleration"""
    # Get years and H0 values with uncertainties
    years = h0_data['year'].values
    h0_obs = h0_data['H0'].values
    h0_err = h0_data['H0_err'].values
    
    # Transfer to GPU if available
    years_gpu = to_gpu(years)
    h0_obs_gpu = to_gpu(h0_obs)
    h0_err_gpu = to_gpu(h0_err)
    
    # Get Genesis-Sphere predictions on GPU
    h0_pred_gpu = gs_predicted_h0_variations_gpu(gs_model, years_gpu)
    
    # Calculate chi-squared on GPU
    residuals_gpu = (h0_obs_gpu - h0_pred_gpu) / h0_err_gpu
    chi2 = float(cp.sum(residuals_gpu**2))  # Convert back to Python float
    
    return chi2

def calculate_sne_chi2_gpu(gs_model, sne_data):
    """Calculate proper chi-squared for supernovae distance modulus with GPU acceleration"""
    # Extract redshifts and distance moduli with uncertainties
    redshifts = sne_data['z'].values
    mu_obs = sne_data['mu'].values
    mu_err = sne_data['mu_err'].values
    
    # Transfer to GPU if available
    redshifts_gpu = to_gpu(redshifts)
    mu_obs_gpu = to_gpu(mu_obs)
    mu_err_gpu = to_gpu(mu_err)
    
    # Get predictions on GPU
    mu_pred_gpu = gs_predicted_distance_modulus_gpu(gs_model, redshifts_gpu)
    
    # Calculate chi-squared on GPU - much faster for large SNe datasets
    residuals_gpu = (mu_obs_gpu - mu_pred_gpu) / mu_err_gpu
    chi2 = float(cp.sum(residuals_gpu**2))
    
    return chi2

def calculate_bao_chi2_gpu(gs_model, bao_data):
    """Calculate proper chi-squared for BAO measurements with GPU acceleration"""
    # Extract redshifts and sound horizon with uncertainties
    redshifts = bao_data['z'].values
    rd_obs = bao_data['rd'].values
    rd_err = bao_data['rd_err'].values
    
    # Transfer to GPU if available
    redshifts_gpu = to_gpu(redshifts)
    rd_obs_gpu = to_gpu(rd_obs)
    rd_err_gpu = to_gpu(rd_err)
    
    # Get predictions on GPU
    rd_pred_gpu = gs_predicted_bao_feature_gpu(gs_model, redshifts_gpu)
    
    # Calculate chi-squared on GPU
    residuals_gpu = (rd_obs_gpu - rd_pred_gpu) / rd_err_gpu
    chi2 = float(cp.sum(residuals_gpu**2))
    
    return chi2

def log_likelihood_gpu(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """
    Log likelihood function using GPU acceleration.
    Calculates the total likelihood of observing the data given the parameters.
    """
    omega, beta = params
    
    # Create model instance with current parameters
    gs_model = GenesisSphereModel(alpha=fixed_alpha, beta=beta, omega=omega, epsilon=fixed_epsilon)
    
    # Calculate chi-squared for each dataset
    chi2_h0 = calculate_h0_chi2_gpu(gs_model, data_h0)
    chi2_sne = calculate_sne_chi2_gpu(gs_model, data_sne)
    chi2_bao = calculate_bao_chi2_gpu(gs_model, data_bao)
    
    # Calculate total chi-squared (proper statistical approach: sum without weighting)
    total_chi2 = chi2_h0 + chi2_sne + chi2_bao
    
    # Convert to log-likelihood
    logL = -0.5 * total_chi2
    
    return logL

# Example for comparing CPU vs GPU performance
def compare_performance(n_points=1000):
    """Compare performance between CPU and GPU implementations"""
    # Create mock datasets
    np.random.seed(42)
    
    # Mock H0 data
    years = np.linspace(1930, 2022, 20)
    h0_values = 70.0 + 5.0 * np.sin(0.1 * (years - 1970)) + np.random.normal(0, 2, size=len(years))
    h0_errors = np.ones_like(years) * 2.0
    h0_data = pd.DataFrame({'year': years, 'H0': h0_values, 'H0_err': h0_errors})
    
    # Mock SNe data (larger dataset)
    redshifts = np.random.uniform(0.01, 1.5, n_points)
    mu_values = 5.0 * np.log10(redshifts * 3000.0) + 25.0 + np.random.normal(0, 0.2, size=n_points)
    mu_errors = np.ones_like(redshifts) * 0.15
    sne_data = pd.DataFrame({'z': redshifts, 'mu': mu_values, 'mu_err': mu_errors})
    
    # Mock BAO data
    bao_z = np.array([0.1, 0.35, 0.57, 0.7, 1.5, 2.3])
    bao_rd = np.array([147.0, 148.5, 147.5, 147.9, 147.8, 146.2])
    bao_err = np.array([1.7, 2.8, 1.9, 2.4, 3.2, 4.5])
    bao_data = pd.DataFrame({'z': bao_z, 'rd': bao_rd, 'rd_err': bao_err})
    
    # Parameters to test
    params = (2.0, 1.2)  # omega, beta
    fixed_alpha = 0.02
    fixed_epsilon = 0.1
    
    # Model instance for CPU version
    gs_model = GenesisSphereModel(alpha=fixed_alpha, beta=params[1], omega=params[0], epsilon=fixed_epsilon)
    
    # Warm-up for more accurate timing
    if HAS_GPU:
        # Warm up GPU
        _ = log_likelihood_gpu(params, h0_data, sne_data, bao_data, fixed_alpha, fixed_epsilon)
        if HAS_GPU:
            cp.cuda.runtime.deviceSynchronize()
    
    # Time GPU version
    start_gpu = time.time()
    log_likelihood_gpu_result = log_likelihood_gpu(params, h0_data, sne_data, bao_data, fixed_alpha, fixed_epsilon)
    if HAS_GPU:
        cp.cuda.runtime.deviceSynchronize()  # Ensure all GPU computations are done
    gpu_time = time.time() - start_gpu
    
    # Print results
    print(f"\nPerformance comparison (n_points={n_points}):")
    print(f"Using: {'GPU (CuPy)' if HAS_GPU else 'CPU (NumPy)'}")
    print(f"Log-likelihood: {log_likelihood_gpu_result:.4f}")
    print(f"Runtime: {gpu_time:.6f} seconds")
    
    # Memory management - free GPU memory if used
    if HAS_GPU:
        cp.get_default_memory_pool().free_all_blocks()
        print("GPU memory pool cleared")

if __name__ == "__main__":
    # Small dataset
    compare_performance(n_points=100)
    
    # Large dataset
    compare_performance(n_points=10000)
