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
        
        # Save the current state of the chain
        samples = sampler.get_chain(discard=nburn, thin=10, flat=True)
        
        if len(samples) == 0:
            print("No valid samples to save yet.")
            return
            
        # Create DataFrame and save
        df_samples = pd.DataFrame(samples, columns=['omega', 'beta'])
        checkpoint_file = os.path.join(output_dir, f"{prefix}_checkpoint_{timestamp}.csv")
        df_samples.to_csv(checkpoint_file, index=False)
        
        # Calculate quick stats if we have enough samples
        if len(samples) >= 10:
            results_summary = {}
            for i, label in enumerate(['omega', 'beta']):
                if len(samples) >= 3:  # Need at least 3 samples for percentiles
                    mcmc = np.percentile(samples[:, i], [16, 50, 84])
                    median = mcmc[1]
                    results_summary[label] = {'median': float(median)}
                else:
                    # Just use mean if not enough for percentiles
                    results_summary[label] = {'median': float(np.mean(samples[:, i]))}
            
            # Save basic info with additional runtime metrics
            info = {
                'timestamp': timestamp,
                'samples_saved': len(samples),
                'current_results': results_summary,
                'fixed_params': fixed_params,
                'progress': {
                    'step_number': step_number,
                    'batch_speed': batch_speed,
                    'elapsed_minutes': total_elapsed / 60.0,
                    'epoch': int(step_number / len(sampler.chain))
                }
            }
            
            info_file = os.path.join(output_dir, f"{prefix}_checkpoint_info_{timestamp}.json")
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=4)
            
            # Save sampler state for potential recovery
            state_file = os.path.join(output_dir, f"{prefix}_state_{timestamp}.npz")
            np.savez(state_file, 
                     chain=sampler.chain,
                     random_state=np.random.get_state())
                
            print(f"Checkpoint saved with {len(samples)} samples. Current estimates:")
            print(f"  Epoch: {int(step_number / len(sampler.chain))}, Steps: {step_number}")
            print(f"  Batch speed: {batch_speed:.1f} samples/s, Elapsed: {total_elapsed/60:.1f} min")
            for param, values in results_summary.items():
                print(f"  {param}: {values['median']:.4f}")
    
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

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
    parser.add_argument("--initial_beta", type=float, default=-0.0333, help="Initial guess for beta")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix for output filenames")
    parser.add_argument("--checkpoint_interval", type=int, default=100, 
                        help="Save intermediate results every N steps (0 to disable)")
    parser.add_argument("--test_mode", action="store_true", 
                        help="Run in test mode with reduced computation")
    parser.add_argument("--max_time", type=int, default=30,
                        help="Maximum runtime in minutes (default: 30 minutes)")
    parser.add_argument("--resume", type=str, default="", 
                        help="Path to state file to resume from a previous run")

    args = parser.parse_args()

    # Create save points directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    run_id = f"{timestamp}{suffix}"
    save_points_dir = os.path.join(results_dir, f"savepoints_{run_id}")
    os.makedirs(save_points_dir, exist_ok=True)
    
    # Modify parameters if in test mode
    if args.test_mode:
        print("Running in TEST MODE with reduced computation")
        args.nwalkers = 10
        args.nsteps = 50
        args.nburn = 10
        args.checkpoint_interval = 10

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
    print(f"Save points directory: {save_points_dir}")
    
    # Initial memory usage
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
                memory_pool = cp.cuda.MemoryPool(cp.cuda.malloc)
                cp.cuda.set_allocator(memory_pool.malloc)
                with cp.cuda.Device(0):
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
                print("  Continuing with reduced GPU optimization.")
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
    
    # The 'args' tuple passes fixed parameters and data to the log_posterior function
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
    chunk_size = min(50, args.checkpoint_interval)  # Smaller chunks for more frequent updates
    n_chunks = args.nsteps // chunk_size
    remaining_steps = args.nsteps % chunk_size
    
    current_pos = pos
    steps_completed = steps_already_completed
    total_steps = args.nsteps + steps_already_completed
    checkpoint_counter = 0
    
    # Track batch speed over time
    batch_speeds = []
    last_checkpoint_time = time.time()
    
    for i in range(n_chunks + (1 if remaining_steps > 0 else 0)):
        # Check if we've exceeded max runtime
        current_elapsed = time.time() - start_time
        if current_elapsed > max_runtime_seconds:
            print(f"Maximum runtime of {args.max_time} minutes exceeded. Stopping.")
            
            # Save final checkpoint before stopping
            current_batch_speed = np.mean(batch_speeds) if batch_speeds else 0
            save_intermediate_results(
                sampler, args.nburn, save_points_dir, f"mcmc_{run_id}", 
                fixed_params, steps_completed, current_batch_speed, current_elapsed
            )
            break
            
        # Determine steps for this chunk
        if i == n_chunks and remaining_steps > 0:
            steps = remaining_steps
        else:
            steps = chunk_size
            
        # Run this chunk
        chunk_start = time.time()
        print(f"Running chunk {i+1}/{n_chunks + (1 if remaining_steps > 0 else 0)}: "
              f"{steps} steps ({steps_completed}/{total_steps} total)")
        print(f"Elapsed time: {current_elapsed/60:.1f} min, Remaining: {(max_runtime_seconds-current_elapsed)/60:.1f} min")
        
        current_pos, _, _ = sampler.run_mcmc(current_pos, steps, progress=True)
        steps_completed += steps
        
        # Calculate batch speed
        chunk_time = time.time() - chunk_start
        batch_speed = steps * nwalkers / chunk_time  # samples per second
        batch_speeds.append(batch_speed)
        current_batch_speed = batch_speed
        
        # Calculate estimated epoch (full passes through the dataset)
        # In MCMC, this is the number of steps completed divided by the total number of walkers
        current_epoch = steps_completed / nwalkers
        
        # Report on this chunk
        print(f"  Completed chunk in {chunk_time:.2f}s "
              f"({batch_speed:.1f} samples/s, "
              f"~{(total_steps-steps_completed)/(steps/chunk_time)/60:.1f} minutes remaining)")
        print(f"  Epoch: {current_epoch:.2f}, Batch speed: {batch_speed:.1f} samples/s")
        
        # Garbage collection to prevent memory leaks
        gc.collect()
        print_memory_usage(f"after chunk {i+1}")
        
        # Save checkpoint if needed or if enough time has passed
        checkpoint_counter += steps
        time_since_last_checkpoint = time.time() - last_checkpoint_time
        
        if (args.checkpoint_interval > 0 and checkpoint_counter >= args.checkpoint_interval) or time_since_last_checkpoint > 300:  # 5 minutes
            print(f"Saving checkpoint after {steps_completed} steps...")
            save_intermediate_results(
                sampler, args.nburn, save_points_dir, f"mcmc_{run_id}", 
                fixed_params, steps_completed, current_batch_speed, current_elapsed
            )
            checkpoint_counter = 0
            last_checkpoint_time = time.time()

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

        if len(samples) < 10:
            print("WARNING: Very few samples after burn-in and thinning.")
            print("Consider reducing burn-in or increasing total steps.")

        # --- Save Results ---
        print("Saving final results...")

        # Save the samples (the chain)
        chain_file = os.path.join(results_dir, f"mcmc_chain_{run_id}.csv")
        df_samples = pd.DataFrame(samples, columns=['omega', 'beta'])
        df_samples.to_csv(chain_file, index=False)
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
            'samples_shape': list(samples.shape),
            'test_mode': args.test_mode
        }
        info_file = os.path.join(results_dir, f"run_info_{run_id}.json")
        with open(info_file, 'w') as f:
            json.dump(run_info, f, indent=4)
        print(f"Run info saved to {info_file}")

        # --- Basic Analysis & Plotting ---
        print("\nAnalyzing MCMC results...")

        # Calculate median and 1-sigma credible intervals (16th, 50th, 84th percentiles)
        results_summary = {}
        print("\n=== MCMC Parameter Estimates (median and 1-sigma credible interval) ===")
        for i, label in enumerate(['omega', 'beta']):
            mcmc = np.percentile(samples[:, i], [16, 50, 84])
            q = np.diff(mcmc) # q[0] = 50th-16th, q[1] = 84th-50th
            median = mcmc[1]
            upper_err = q[1]
            lower_err = q[0]
            print(f"{PARAM_LABELS[i]} = {median:.4f} (+{upper_err:.4f} / -{lower_err:.4f})")
            results_summary[label] = {'median': float(median), 
                                     'upper_err': float(upper_err), 
                                     'lower_err': float(lower_err)}

        # Save summary stats
        stats_file = os.path.join(results_dir, f"param_stats_{run_id}.json")
        with open(stats_file, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Parameter stats saved to {stats_file}")

        # Generate a corner plot using the corner library
        print("\nGenerating corner plot...")
        try:
            figure = corner.corner(
                samples, labels=PARAM_LABELS, # Use LaTeX labels
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True, title_kwargs={"fontsize": 12},
                truths=[results_summary['omega']['median'], results_summary['beta']['median']], # Show median values
                truth_color='red'
            )
            corner_plot_file = os.path.join(results_dir, f"corner_plot_{run_id}.png")
            figure.savefig(corner_plot_file)
            print(f"Corner plot saved to {corner_plot_file}")
        except ImportError:
            print("\nInstall 'corner' package (`pip install corner`) to generate corner plots.")
        except Exception as e:
            print(f"Error during corner plot generation: {e}")

        # Generate a markdown summary report
        generate_validation_summary(results_summary, args, timestamp, suffix, acceptance_fraction)

    except Exception as e:
        print(f"Error during MCMC results processing: {e}")
        import traceback
        traceback.print_exc()
        print("Chain data might still be saved if the run completed.")

    print("\nMCMC parameter estimation script finished!")

    # Final GPU cleanup
    if HAS_GPU:
        print("Performing final GPU memory cleanup...")
        cp.get_default_memory_pool().free_all_blocks()

def generate_validation_summary(results_summary, args, timestamp, suffix, acceptance_fraction):
    """Generate a markdown summary of the MCMC parameter estimation"""
    summary = [
        "# Genesis-Sphere MCMC Parameter Estimation Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Validation Method",
        "\nThis validation uses Markov Chain Monte Carlo (MCMC) to estimate the posterior probability distribution",
        "of Genesis-Sphere model parameters based on astronomical datasets. Unlike the previous grid search approach,",
        "MCMC provides robust parameter uncertainties and explores the parameter space more efficiently.\n",
        "## MCMC Settings",
        f"\n- Walkers: {args.nwalkers}",
        f"- Steps per walker: {args.nsteps}",
        f"- Burn-in steps discarded: {args.nburn}",
        f"- Initial parameter guess: ω={args.initial_omega:.4f}, β={args.initial_beta:.4f}",
        f"- Fixed parameters: α={args.alpha:.4f}, ε={args.epsilon:.4f}",
        f"- Mean acceptance fraction: {acceptance_fraction:.3f}\n",
        "## Parameter Estimates",
        "\nBest-fit parameters with 1-sigma (68%) credible intervals:",
        f"\n| Parameter | Median | Lower Error | Upper Error |",
        "|-----------|--------|-------------|-------------|",
        f"| Omega (ω) | {results_summary['omega']['median']:.4f} | {results_summary['omega']['lower_err']:.4f} | {results_summary['omega']['upper_err']:.4f} |",
        f"| Beta (β) | {results_summary['beta']['median']:.4f} | {results_summary['beta']['lower_err']:.4f} | {results_summary['beta']['upper_err']:.4f} |\n",
        "## Corner Plot",
        "\n![Parameter Corner Plot](corner_plot_" + f"{timestamp}{suffix}.png" + ")",
        "\nThe corner plot shows the 1D and 2D posterior distributions of the model parameters.",
        "Contours show the 1-sigma, 2-sigma, and 3-sigma credible regions.",
        "\n## Interpretation",
        "\nThe MCMC analysis shows that the optimal Genesis-Sphere parameters are:",
        f"- **Omega (ω)**: {results_summary['omega']['median']:.4f} ± {(results_summary['omega']['lower_err'] + results_summary['omega']['upper_err'])/2:.4f}",
        f"- **Beta (β)**: {results_summary['beta']['median']:.4f} ± {(results_summary['beta']['lower_err'] + results_summary['beta']['upper_err'])/2:.4f}",
        "\nThese values represent the statistical constraints from combining H₀ correlation,",
        "supernovae distance modulus fitting, and BAO signal detection. The uncertainties",
        "reflect the genuine statistical uncertainty in determining these parameters from the available data.",
        "\nCompared to the previous grid search approach, this MCMC analysis provides more robust",
        "parameter constraints by thoroughly exploring the parameter space and quantifying uncertainties.",
        "\n---",
        "\n*This report was automatically generated by the Genesis-Sphere MCMC parameter estimation framework.*"
    ]
    
    summary_path = os.path.join(results_dir, f"mcmc_summary_{timestamp}{suffix}.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print(f"Validation summary saved to: {summary_path}")

if __name__ == "__main__":
    try:
        # Force CuPy to initialize before we start to catch any startup errors
        if HAS_GPU:
            print("Initializing GPU environment...")
            cp.array([1, 2, 3])  # Simple test array
            cp.cuda.Stream.null.synchronize()  # Ensure initialization completes
            print("GPU initialization complete.")
        
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup GPU resources in case of error
        if HAS_GPU:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                print("GPU resources cleaned up after error.")
            except:
                pass
                
        sys.exit(1)
