# Cyclic Universe and Black Hole Mapping

This directory contains visualizations demonstrating the connection between cyclic universe models and black hole physics using the Genesis-Sphere framework.

## Overview

The Genesis-Sphere model demonstrates a mathematical formulation that naturally describes both:
- The temporal dynamics of cyclic universe models
- The time dilation effects near black holes

This connection reveals how the same underlying mathematical structure can govern seemingly different cosmic phenomena, providing insight into the nature of time, space-time density, and gravity.

## Key Concepts

- **Cycle Period**: Determined by the omega (ω) parameter, controls the frequency of cosmic cycles
- **Temporal Damping**: Determined by the beta (β) parameter, controls time dilation magnitude
- **Phase Mapping**: Maps black hole radial distance to position in cosmic cycle
- **Temporal Flow Equivalence**: Shows how black hole time dilation relates to cosmic temporal flow

## Command-Line Interface

### Basic Commands

```bash
# Run the core model directly
python models/cyclic_bh_mapping.py

# Run the simulation with default parameters
python simulations/cyclic_cosmology_simulation.py
```

### Parameter Exploration

```bash
# Run parameter exploration mode (tests multiple parameter values)
python simulations/cyclic_cosmology_simulation.py --param-exploration
```

### Customizing Parameters

```bash
# Vary omega (angular frequency) parameter
python simulations/cyclic_cosmology_simulation.py --omega 0.5
python simulations/cyclic_cosmology_simulation.py --omega 1.0
python simulations/cyclic_cosmology_simulation.py --omega 2.0

# Vary beta (temporal damping) parameter
python simulations/cyclic_cosmology_simulation.py --beta 0.4
python simulations/cyclic_cosmology_simulation.py --beta 0.8
python simulations/cyclic_cosmology_simulation.py --beta 1.5

# Specify cycle period directly (instead of using omega)
python simulations/cyclic_cosmology_simulation.py --cycle-period 8

# Combine multiple parameters
python simulations/cyclic_cosmology_simulation.py --alpha 0.03 --beta 0.6 --omega 1.5 --epsilon 0.05
```

### Example Parameter Sets

```bash
# Fast oscillating universe
python simulations/cyclic_cosmology_simulation.py --omega 2.0 --beta 0.4

# Strong time dilation near singularities
python simulations/cyclic_cosmology_simulation.py --beta 1.5

# Long cycle period
python simulations/cyclic_cosmology_simulation.py --cycle-period 20
```

## Output Files

This directory contains the following output files:

- **bh_cyclic_mapping.png**: Static visualization showing the relationship between black hole physics and cyclic universe models
- **cyclic_universe_animation.mp4**: Animation showing the evolution of cyclic universe properties and their correspondence to black hole states
- **cycle_frames/**: Directory containing individual animation frames (if FFmpeg is not available)
- **omega_parameter_effect.png**: Visualization of how different omega values affect cycle frequency
- **beta_parameter_effect.png**: Visualization of how different beta values affect time dilation

## Theoretical Implications

The equivalence demonstrated by this model suggests:

1. **Common Mechanism**: Both black hole time dilation and cyclic universe dynamics may emerge from the same underlying space-time geometry
2. **Universal Time Behavior**: Similar mathematical descriptions of temporal flow can apply at different scales and contexts
3. **Singularity Connection**: Black hole event horizons and cosmic singularities may be related phenomena
4. **Predictive Power**: Understanding one context may inform predictions in the other

## Further Reading

For more details, see the full theoretical explanation in the Genesis-Sphere whitepaper and related publications.
