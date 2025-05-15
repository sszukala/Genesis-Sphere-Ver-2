@echo off
echo Starting IBM Quantum Parameter Sweep Optimization
echo ---------------------------------------------
echo This will use your 10-minute IBM Quantum allocation
echo to optimize training parameters.
echo.
echo Method: QAOA (Quantum Approximate Optimization Algorithm)
echo Note: This approach does NOT use MCMC (Markov Chain Monte Carlo)
echo but leverages quantum superposition to explore the parameter space.
echo.
echo The optimizer uses:
echo - QUBO problem formulation
echo - QAOA quantum algorithm
echo - COBYLA classical optimizer for variational parameters
echo.

:: Activate your Python environment if needed
:: call activate your_environment_name

:: Run the quantum optimizer
python quantum_optimizer.py

echo.
echo Optimization complete! Check the results directory for optimized parameters.
pause