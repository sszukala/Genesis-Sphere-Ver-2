# Genesis-Sphere Model Validation

This directory contains tools and scripts for validating the Genesis-Sphere model against established cosmological data and models.

## Data Sources

The validation framework uses the following cosmological data sources:

1. **NASA/IPAC Extragalactic Database (NED)** - Uses the [NED Cosmology Calculator](https://ned.ipac.caltech.edu/help/cosmology_calc.html) for standard cosmological measurements
2. **Astropy Cosmology Models** - Leverages standard models like Planck18, WMAP9 from the astropy.cosmology package
3. **Observational Datasets** - Type Ia supernovae data, CMB measurements, and BAO constraints

## Validation Approaches

### 1. Parameter Mapping

Maps Genesis-Sphere parameters (α, β, ω, ε) to standard cosmological parameters:
- Hubble constant (H₀)
- Matter density (Ωₘ)
- Dark energy density (ΩΛ)
- Equation of state parameter (w)

### 2. Observable Predictions

Compares predictions made by Genesis-Sphere against established observables:
- Distance-redshift relation
- Age of the universe
- Expansion history
- Time dilation effects

### 3. Model Consistency Checks

Validates internal consistency and compatibility with:
- General relativity in appropriate limits
- Standard ΛCDM model predictions
- Observational constraints from multiple sources

## Usage

### NED Cosmology Calculator Validation

```bash
python ned_validation.py --alpha 0.02 --beta 0.8 --omega 1.0 --epsilon 0.1
```

### Astropy Cosmology Validation

```bash
python astropy_validation.py --model Planck18 --compare density_evolution
```

### Observational Dataset Comparison

```bash
python observational_validation.py --dataset supernovae --data-file datasets/SNe_gold_sample.csv
```

## AI Validation Summaries

Each validation run automatically generates an intelligent summary that interprets the results, providing:

- Assessment of model fit quality
- Interpretation of statistical metrics
- Dataset-specific insights
- Recommendations for model improvements

These summaries are saved as Markdown files in the results directory and also displayed in the console output. They help translate complex statistical results into actionable insights for model development.

Example summary sections include:
- Model configuration details
- Statistical fit analysis
- Overall performance assessment 
- Detailed metric analysis
- Recommendations for next steps

To skip generating the summary (for automated batch processing):
```bash
python observational_validation.py --dataset supernovae --no-summary
```

## Required Dependencies

- astropy
- numpy
- scipy
- matplotlib
- pandas
- requests

Install these dependencies with:
```bash
pip install astropy numpy scipy matplotlib pandas requests
```

## Validation Results

Validation results are stored in the `/results` subdirectory with the following organization:
- Parameter validation plots: `results/parameter_space/`
- Observable comparisons: `results/observables/`
- Statistical analysis: `results/statistics/`
- Consistency metrics: `results/consistency/`
