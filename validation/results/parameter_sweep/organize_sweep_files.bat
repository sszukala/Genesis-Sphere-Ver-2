@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo  PARAMETER SWEEP FILES ORGANIZATION UTILITY
echo ===================================================
echo.
echo  This script will organize all parameter sweep files
echo  into a structured directory layout.
echo.
echo  Current directory: %CD%
echo.
echo  Press any key to continue or Ctrl+C to cancel...
pause > nul
echo.

set "BASE_DIR=%CD%"
echo Creating directory structure...

:: Create main directories
mkdir "runs" 2>nul
mkdir "figures\corner_plots" 2>nul
mkdir "figures\heatmaps" 2>nul
mkdir "summaries" 2>nul
mkdir "data" 2>nul

echo Directory structure created successfully.
echo.
echo ===================================================
echo  ORGANIZING RUN INFO FILES
echo ===================================================

:: Find and organize run_info files
for %%f in (run_info_*.json) do (
    echo Processing: %%f
    set "file=%%f"
    set "timestamp=!file:run_info_=!"
    set "timestamp=!timestamp:.json=!"
    
    mkdir "runs\run_!timestamp!" 2>nul
    echo   Creating run directory: runs\run_!timestamp!
    copy "%%f" "runs\run_!timestamp!\run_info.json" > nul
    echo   Copied to: runs\run_!timestamp!\run_info.json
    echo.
)

echo ===================================================
echo  ORGANIZING PROGRESS LOG FILES
echo ===================================================

:: Find and organize progress log files
for %%f in (progress_log_*.txt) do (
    echo Processing: %%f
    set "file=%%f"
    set "timestamp=!file:progress_log_=!"
    set "timestamp=!timestamp:.txt=!"
    
    if exist "runs\run_!timestamp!" (
        copy "%%f" "runs\run_!timestamp!\progress_log.txt" > nul
        echo   Copied to: runs\run_!timestamp!\progress_log.txt
    ) else (
        mkdir "runs\run_!timestamp!" 2>nul
        echo   Creating run directory: runs\run_!timestamp!
        copy "%%f" "runs\run_!timestamp!\progress_log.txt" > nul
        echo   Copied to: runs\run_!timestamp!\progress_log.txt
    )
    echo.
)

echo ===================================================
echo  ORGANIZING MCMC CHAIN FILES
echo ===================================================

:: Find and organize MCMC chain files
for %%f in (mcmc_chain_*.csv) do (
    echo Processing: %%f
    set "file=%%f"
    set "timestamp=!file:mcmc_chain_=!"
    set "timestamp=!timestamp:.csv=!"
    
    if exist "runs\run_!timestamp!" (
        copy "%%f" "runs\run_!timestamp!\mcmc_chain.csv" > nul
        echo   Copied to: runs\run_!timestamp!\mcmc_chain.csv
    ) else (
        mkdir "runs\run_!timestamp!" 2>nul
        echo   Creating run directory: runs\run_!timestamp!
        copy "%%f" "runs\run_!timestamp!\mcmc_chain.csv" > nul
        echo   Copied to: runs\run_!timestamp!\mcmc_chain.csv
    )
    echo.
)

echo ===================================================
echo  ORGANIZING CORNER PLOTS
echo ===================================================

:: Find and organize corner plot images
for %%f in (corner_plot_*.png) do (
    echo Processing: %%f
    set "file=%%f"
    set "timestamp=!file:corner_plot_=!"
    set "timestamp=!timestamp:.png=!"
    
    if exist "runs\run_!timestamp!" (
        copy "%%f" "runs\run_!timestamp!\corner_plot.png" > nul
        echo   Copied to: runs\run_!timestamp!\corner_plot.png
    ) else (
        mkdir "runs\run_!timestamp!" 2>nul
        echo   Creating run directory: runs\run_!timestamp!
        copy "%%f" "runs\run_!timestamp!\corner_plot.png" > nul
        echo   Copied to: runs\run_!timestamp!\corner_plot.png
    )
    
    :: Also copy to figures directory
    copy "%%f" "figures\corner_plots\" > nul
    echo   Copied to: figures\corner_plots\%%f
    echo.
)

echo ===================================================
echo  ORGANIZING MCMC SUMMARIES
echo ===================================================

:: Find and organize MCMC summary files
for %%f in (mcmc_summary_*.json) do (
    echo Processing: %%f
    set "file=%%f"
    set "timestamp=!file:mcmc_summary_=!"
    set "timestamp=!timestamp:.json=!"
    
    if exist "runs\run_!timestamp!" (
        copy "%%f" "runs\run_!timestamp!\mcmc_summary.json" > nul
        echo   Copied to: runs\run_!timestamp!\mcmc_summary.json
    ) else (
        mkdir "runs\run_!timestamp!" 2>nul
        echo   Creating run directory: runs\run_!timestamp!
        copy "%%f" "runs\run_!timestamp!\mcmc_summary.json" > nul
        echo   Copied to: runs\run_!timestamp!\mcmc_summary.json
    )
    echo.
)

echo ===================================================
echo  ORGANIZING PARAMETER SWEEP FILES
echo ===================================================

:: Find and organize parameter sweep CSV files
for %%f in (parameter_sweep_*.csv) do (
    echo Processing: %%f
    set "file=%%f"
    set "timestamp=!file:parameter_sweep_=!"
    set "timestamp=!timestamp:.csv=!"
    
    if exist "runs\run_!timestamp!" (
        copy "%%f" "runs\run_!timestamp!\parameter_sweep.csv" > nul
        echo   Copied to: runs\run_!timestamp!\parameter_sweep.csv
    ) else (
        mkdir "runs\run_!timestamp!" 2>nul
        echo   Creating run directory: runs\run_!timestamp!
        copy "%%f" "runs\run_!timestamp!\parameter_sweep.csv" > nul
        echo   Copied to: runs\run_!timestamp!\parameter_sweep.csv
    )
    
    :: Also copy to data directory
    copy "%%f" "data\" > nul
    echo   Copied to: data\%%f
    echo.
)

echo ===================================================
echo  ORGANIZING CHECKPOINTS
echo ===================================================

:: Find and organize checkpoint files
for %%f in (mcmc_*_checkpoint_*.csv) do (
    echo Processing: %%f
    set "file=%%f"
    
    :: Extract run ID and checkpoint time using for loop for string parsing
    for /f "tokens=1-3 delims=_" %%a in ("!file!") do (
        set "prefix=%%a"
        set "runid=%%b"
        set "rest=%%c"
        
        :: Extract checkpoint time from the rest of the filename
        for /f "tokens=1,* delims=_" %%x in ("!rest!") do (
            set "checkpointtime=%%y"
            set "checkpointtime=!checkpointtime:.csv=!"
            
            if exist "runs\run_!runid!" (
                mkdir "runs\run_!runid!\checkpoints" 2>nul
                copy "%%f" "runs\run_!runid!\checkpoints\checkpoint_!checkpointtime!.csv" > nul
                echo   Copied to: runs\run_!runid!\checkpoints\checkpoint_!checkpointtime!.csv
            ) else (
                mkdir "runs\run_!runid!" 2>nul
                echo   Creating run directory: runs\run_!runid!
                mkdir "runs\run_!runid!\checkpoints" 2>nul
                copy "%%f" "runs\run_!runid!\checkpoints\checkpoint_!checkpointtime!.csv" > nul
                echo   Copied to: runs\run_!runid!\checkpoints\checkpoint_!checkpointtime!.csv
            )
        )
    )
    echo.
)

echo ===================================================
echo  ORGANIZING CHECKPOINT INFO FILES
echo ===================================================

:: Find and organize checkpoint info files
for %%f in (mcmc_*_checkpoint_info_*.json) do (
    echo Processing: %%f
    set "file=%%f"
    
    :: Extract run ID and checkpoint time using for loop for string parsing
    for /f "tokens=1-5 delims=_" %%a in ("!file!") do (
        set "prefix=%%a"
        set "runid=%%b"
        set "checkpoint=%%c"
        set "info=%%d"
        set "checkpointtime=%%e"
        set "checkpointtime=!checkpointtime:.json=!"
        
        if exist "runs\run_!runid!" (
            mkdir "runs\run_!runid!\checkpoints" 2>nul
            copy "%%f" "runs\run_!runid!\checkpoints\checkpoint_info_!checkpointtime!.json" > nul
            echo   Copied to: runs\run_!runid!\checkpoints\checkpoint_info_!checkpointtime!.json
        ) else (
            mkdir "runs\run_!runid!" 2>nul
            echo   Creating run directory: runs\run_!runid!
            mkdir "runs\run_!runid!\checkpoints" 2>nul
            copy "%%f" "runs\run_!runid!\checkpoints\checkpoint_info_!checkpointtime!.json" > nul
            echo   Copied to: runs\run_!runid!\checkpoints\checkpoint_info_!checkpointtime!.json
        )
    )
    echo.
)

echo ===================================================
echo  ORGANIZING HEATMAPS AND SUMMARIES
echo ===================================================

:: Process heatmap images
for %%f in (combined_score_heatmap.png h0_correlation_heatmap.png sne_r_squared_heatmap.png bao_high_z_effect_size_heatmap.png) do (
    if exist "%%f" (
        echo Processing: %%f
        copy "%%f" "figures\heatmaps\" > nul
        echo   Copied to: figures\heatmaps\%%f
        echo.
    )
)

:: Process summary files
if exist "parameter_sweep_summary.md" (
    echo Processing: parameter_sweep_summary.md
    copy "parameter_sweep_summary.md" "summaries\" > nul
    echo   Copied to: summaries\parameter_sweep_summary.md
    echo.
)

echo ===================================================
echo  CREATING REFERENCES TO LATEST RUN
echo ===================================================

:: Find the most recent run by timestamp
set "latest_run="
set "latest_timestamp=0"

for /d %%d in (runs\run_*) do (
    set "current=%%~nxd"
    set "timestamp=!current:run_=!"
    
    if !timestamp! GTR !latest_timestamp! (
        set "latest_timestamp=!timestamp!"
        set "latest_run=!current!"
    )
)

:: Create a reference file to the latest run
if not "!latest_run!"=="" (
    echo Found latest run: !latest_run!
    
    echo Latest run: !latest_run! > "runs\latest_run.txt"
    echo Full path: %CD%\runs\!latest_run! >> "runs\latest_run.txt"
    echo Created: %date% %time% >> "runs\latest_run.txt"
    
    echo Created reference: runs\latest_run.txt
    echo.
)

echo ===================================================
echo  ORGANIZATION COMPLETE
echo ===================================================
echo.
echo All parameter sweep files have been organized successfully.
echo.
echo Files have been COPIED to their new locations.
echo Original files remain in place for safety.
echo.
echo New structure:
echo.
echo   runs\               - Organized run directories by timestamp
echo   figures\            - Visualizations and plots
echo   summaries\          - Analysis summary documents
echo   data\               - Combined data files
echo.
echo After verifying the organization is correct, you may
echo delete the original files if desired.
echo.
echo ===================================================

endlocal
