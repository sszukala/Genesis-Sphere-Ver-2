"""
Tests all validation scripts with datasets to ensure they work correctly.
"""

import os
import subprocess
import sys
import time

def run_command(command, description):
    """Run a command and print the output"""
    print(f"\n{'-'*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'-'*80}\n")
    
    start_time = time.time()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Get any errors
    stderr = process.communicate()[1]
    if stderr:
        print(f"\nErrors:\n{stderr}")
    
    elapsed = time.time() - start_time
    print(f"\n{'-'*80}")
    print(f"Finished: {description}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"{'-'*80}\n")
    
    return process.returncode

def main():
    """Run all validation scripts"""
    print("Genesis-Sphere Validation Test Suite")
    print("===================================")
    
    # Get all dataset files
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    sne_files = [f for f in os.listdir(datasets_dir) if f.startswith('SNe_') and f.endswith('.csv')]
    
    # Run observational validation for each dataset type
    for dataset_type in ['supernovae', 'cmb', 'bao']:
        if dataset_type == 'supernovae' and sne_files:
            data_file = sne_files[0]
        else:
            data_file = None
        
        # Run with default parameters
        command = [
            sys.executable,
            'observational_validation.py',
            f'--dataset={dataset_type}'
        ]
        
        if data_file:
            command.append(f'--data-file={data_file}')
        
        run_command(command, f"Observational validation with {dataset_type} dataset")
        
        # Run with optimization
        command.append('--optimize')
        run_command(command, f"Observational validation with {dataset_type} dataset (optimized)")
    
    # Run NED validation
    run_command(
        [sys.executable, 'ned_validation.py'],
        "NED validation"
    )
    
    # Run Astropy validation with different models and comparison types
    for model in ['Planck18', 'WMAP9']:
        for compare in ['hubble_evolution', 'density_evolution', 'gs_functions']:
            run_command(
                [sys.executable, 'astropy_validation.py', f'--model={model}', f'--compare={compare}'],
                f"Astropy validation with {model} model, {compare} comparison"
            )
    
    print("All validation tests completed!")

if __name__ == "__main__":
    main()
