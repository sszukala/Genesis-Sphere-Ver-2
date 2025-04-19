"""
Overview of validation files in Genesis-Sphere that use astronomical datasets.
This module provides documentation on which validation scripts use the datasets
stored in the 'datasets' directory.
"""

import os
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Datasets directory
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

def list_available_datasets():
    """
    List all datasets available in the datasets directory
    
    Returns:
    --------
    list:
        List of dataset filenames
    """
    if not os.path.exists(datasets_dir):
        print(f"Datasets directory does not exist: {datasets_dir}")
        return []
    
    datasets = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    return sorted(datasets)

def describe_validation_files():
    """
    Provides description of all validation files that use datasets
    
    Returns:
    --------
    dict:
        Dictionary mapping validation files to the datasets they use
    """
    validation_files = {
        'comprehensive_validation.py': [
            'SNe_Pantheon_Plus_simplified.csv', 
            'BAO_compilation.csv',
            'cmb_priors.csv',
            'bbn_abundances.csv'
        ],
        
        'observational_validation.py': [
            'SNe_gold_sample.csv',
            'SNe_Union2.1.csv',
            'SNe_Pantheon_Plus_simplified.csv'
        ],
        
        'cyclic_behavior_validation.py': [
            'phantom_divide_data.csv',
            'cycle_period_data.csv', 
            'temporal_flow_transition_data.csv',
            'cyclic_model_comparison.csv'
        ],
        
        'celestial_correlation_validation.py': [
            'hubble_measurements.csv',
            'SNe_Pantheon_Plus_simplified.csv',
            'BAO_compilation.csv'
        ],
        
        'inflationary_validation.py': [
            'slow_roll_V1.csv',
            'chaotic_inflation.csv',
            'quadratic_potential.csv'
        ],
        
        'black_hole_validation.py': [
            'schwarzschild_time_dilation.csv',
            'kerr_time_dilation.csv',
            'kerr_newman_time_dilation.csv'
        ],
        
        'astropy_validation.py': [
            'astropy_comparison_planck18.csv',
            'astropy_comparison_wmap9.csv'
        ],
        
        'ned_validation.py': [
            'ned_cosmology_data.csv'
        ]
    }
    
    return validation_files

def generate_validation_dataset_report(output_file=None):
    """
    Generate a comprehensive report of which validation scripts use which datasets
    
    Parameters:
    -----------
    output_file : str, optional
        Path to save the report to
        
    Returns:
    --------
    str:
        Markdown formatted report
    """
    report = [
        "# Genesis-Sphere Validation Datasets Report",
        "",
        "This report documents which validation scripts use which datasets in the Genesis-Sphere validation framework.",
        "These datasets are referenced in the Genesis-Sphere whitepaper as empirical validation sources.",
        "",
        "## Available Datasets",
        ""
    ]
    
    # List available datasets
    datasets = list_available_datasets()
    for dataset in datasets:
        report.append(f"- `{dataset}`")
    
    report.extend([
        "",
        "## Validation Files",
        "",
        "The following validation scripts use these datasets:",
        ""
    ])
    
    # List validation files and their datasets
    validation_files = describe_validation_files()
    for validation_file, used_datasets in validation_files.items():
        report.append(f"### {validation_file}")
        report.append("")
        report.append(f"Located in `validation/{validation_file}`")
        report.append("")
        report.append("Uses datasets:")
        for dataset in used_datasets:
            report.append(f"- `{dataset}`")
        report.append("")
    
    report.extend([
        "## Dataset Usage in Whitepaper",
        "",
        "All datasets in the `datasets` directory are referenced in the Genesis-Sphere whitepaper as empirical validation sources. "
        "These datasets provide the observational foundation for validating the mathematical models presented in the paper.",
        "",
        "The whitepaper includes:",
        "",
        "- Detailed comparison of Genesis-Sphere predictions with Type Ia supernovae data",
        "- Analysis of how the model explains Hubble constant tensions",
        "- BAO pattern detection and its implications for cyclic cosmology",
        "- CMB constraints and their interpretation within the Genesis-Sphere framework",
        "",
        "## Validation Methods",
        "",
        "Each validation script employs different statistical methods appropriate to the dataset being analyzed:",
        "",
        "- Chi-squared statistical tests for goodness of fit",
        "- Correlation analysis for time-series data",
        "- Bayesian Information Criterion (BIC) for model comparison",
        "- Effect size measurements for feature detection",
        ""
    ])
    
    # If output file specified, save the report
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
    
    return '\n'.join(report)

if __name__ == "__main__":
    print("Genesis-Sphere Validation Datasets Overview")
    print("==========================================")
    
    # List available datasets
    print("\nAvailable datasets:")
    datasets = list_available_datasets()
    for dataset in datasets:
        print(f"- {dataset}")
    
    # Print validation files
    print("\nValidation files that use datasets:")
    validation_files = describe_validation_files()
    for validation_file in validation_files:
        print(f"- validation/{validation_file}")
    
    # Generate report
    output_path = os.path.join(parent_dir, 'validation', 'validation_datasets_report.md')
    generate_validation_dataset_report(output_path)
    print(f"\nGenerated validation datasets report: {output_path}")
