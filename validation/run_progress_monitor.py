"""
Progress Monitor for Parameter Sweep Validation

This script monitors the progress of a parameter sweep validation run by reading
the progress log file and displaying a real-time progress bar and statistics.
"""

import os
import sys
import time
import pandas as pd
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import numpy as np
import re

def read_latest_progress(log_file):
    """Read the latest progress information from the log file."""
    try:
        if not os.path.exists(log_file):
            return None
        
        df = pd.read_csv(log_file)
        if len(df) == 0:
            return None
            
        latest = df.iloc[-1]
        return latest
    except Exception as e:
        print(f"Error reading progress log: {e}")
        return None

def read_latest_summary(log_file):
    """Read the latest summary information from the log file."""
    try:
        if not os.path.exists(log_file):
            return None
        
        df = pd.read_csv(log_file)
        if len(df) == 0:
            return None
            
        latest = df.iloc[-1]
        return latest
    except Exception as e:
        print(f"Error reading summary log: {e}")
        return None

def format_time(minutes):
    """Format minutes as HH:MM:SS."""
    if pd.isna(minutes) or not isinstance(minutes, (int, float)):
        return "Unknown"
    
    seconds = int(minutes * 60)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def monitor_text_mode(run_dir, refresh_rate=5):
    """Monitor progress using text-based display with TQDM progress bar."""
    progress_log = os.path.join(run_dir, "progress_log.txt")
    summary_log = os.path.join(run_dir, "summary_log.txt")
    
    print(f"Monitoring parameter sweep progress in {run_dir}")
    print(f"Press Ctrl+C to exit monitoring\n")
    
    progress_bar = None
    last_progress = None
    
    try:
        while True:
            progress = read_latest_progress(progress_log)
            
            if progress is not None:
                progress_pct = float(progress.get('Progress_Pct', 0))
                steps = int(progress.get('Step', 0))
                elapsed = float(progress.get('Elapsed_Min', 0))
                remaining = float(progress.get('Remaining_Min', 0))
                epoch = float(progress.get('Epoch', 0))
                speed = float(progress.get('Batch_Speed', 0))
                
                # Initialize progress bar if first time or if the total has changed
                if progress_bar is None or (last_progress is not None and int(last_progress.get('Step', 0)) != steps):
                    if progress_bar:
                        progress_bar.close()
                    
                    progress_bar = tqdm(total=100, unit="%", desc="Progress", 
                                       bar_format="{desc}: {percentage:3.1f}%|{bar}| {n:.1f}/{total:.1f} [ETA: {remaining}]")
                    progress_bar.update(progress_pct)
                
                # Only update if there's been a change
                if last_progress is None or progress_pct > float(last_progress.get('Progress_Pct', 0)):
                    progress_bar.update(progress_pct - progress_bar.n)
                
                # Print additional statistics
                est_completion = datetime.now() + timedelta(minutes=remaining)
                
                # Clear the previous lines if needed
                if last_progress is not None:
                    sys.stdout.write("\033[F" * 8)  # Move cursor up 8 lines
                
                print(f"\nCurrent Step: {steps}")
                print(f"Epoch: {epoch:.2f}")
                print(f"Processing Speed: {speed:.1f} samples/sec")
                print(f"Elapsed Time: {format_time(elapsed)}")
                print(f"Remaining Time: {format_time(remaining)}")
                print(f"Est. Completion: {est_completion.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Read latest summary if available
                summary = read_latest_summary(summary_log)
                if summary is not None:
                    omega = float(summary.get('Best_Omega', 0))
                    beta = float(summary.get('Best_Beta', 0))
                    score = float(summary.get('Best_Score', 0))
                    
                    print(f"Best Parameters: ω={omega:.4f}, β={beta:.4f}, Score={score:.4f}")
                else:
                    print("No summary data available yet")
                
                last_progress = progress
            
            time.sleep(refresh_rate)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        if progress_bar:
            progress_bar.close()

def monitor_plot_mode(run_dir, refresh_rate=5):
    """Monitor progress using an animated plot."""
    progress_log = os.path.join(run_dir, "progress_log.txt")
    summary_log = os.path.join(run_dir, "summary_log.txt")
    
    print(f"Launching graphical progress monitor for {run_dir}")
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Parameter Sweep Progress Monitor - {os.path.basename(run_dir)}", fontsize=14)
    
    # Progress plot
    progress_line, = ax1.plot([], [], 'b-', label='Progress (%)')
    speed_line, = ax1.plot([], [], 'r-', label='Speed (samples/s)', alpha=0.7)
    ax1.set_xlim(0, 10)  # Will be adjusted
    ax1.set_ylim(0, 110)  # 0-100% with a little headroom
    ax1.set_ylabel('Progress (%) / Speed')
    ax1.set_xlabel('Time (minutes)')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Create a second y-axis for the remaining time
    ax1_twin = ax1.twinx()
    remaining_line, = ax1_twin.plot([], [], 'g-', label='Remaining Time (min)')
    ax1_twin.set_ylabel('Remaining Time (min)')
    ax1_twin.set_ylim(0, 120)  # Initial scale, will adjust
    ax1_twin.legend(loc='upper right')
    
    # Memory usage plot
    memory_line, = ax2.plot([], [], 'g-', label='Memory (MB)')
    ax2.set_xlim(0, 10)  # Will be adjusted
    ax2.set_ylim(0, 100)  # Will be adjusted
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # Add a text box for summary information
    info_text = ax1.text(0.02, 0.97, '', transform=ax1.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', 
                         facecolor='wheat', alpha=0.5), fontsize=9)
    
    # Add estimation quality indicator
    estimate_quality = ax1_twin.text(0.98, 0.97, '', transform=ax1_twin.transAxes,
                                  verticalalignment='top', horizontalalignment='right',
                                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                                  fontsize=8)

    # Initialize variables
    max_remaining = 120  # Initial scale for remaining time (minutes)
    target_epochs = 100   # Default value, will be updated from run_info.json if available
    
    # Function to update plot
    def update(frame):
        nonlocal max_remaining, target_epochs
        
        try:
            if not os.path.exists(progress_log):
                return progress_line, speed_line, remaining_line, memory_line, info_text, estimate_quality
            
            # Read data
            try:
                df = pd.read_csv(progress_log)
                if len(df) == 0:
                    return progress_line, speed_line, remaining_line, memory_line, info_text, estimate_quality
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                return progress_line, speed_line, remaining_line, memory_line, info_text, estimate_quality
                
            # Calculate elapsed time from first to last entry
            times = df['Elapsed_Min'].values
            progress = df['Epoch'].values * 100 / target_epochs  # Convert to percentage
            speeds = df['Batch_Speed'].values
            memory = df['Memory_MB'].values if 'Memory_MB' in df.columns else np.zeros_like(times)
            remaining = df['Remaining_Min'].values
            
            # Apply a moving average to the remaining time to smooth out spikes
            window_size = min(10, len(remaining))
            if window_size > 0:
                # Calculate weighted moving average (recent values have more weight)
                weights = np.linspace(0.5, 1.0, window_size)
                weights = weights / np.sum(weights)
                
                # Apply weighted moving average with a cap on maximum time
                MAX_REASONABLE_HOURS = 24  # Cap at 24 hours
                for i in range(len(remaining)):
                    # For early estimates, use more aggressive smoothing
                    if progress[i] < 5:  # First 5% of progress
                        # Cap initial estimates more aggressively
                        remaining[i] = min(remaining[i], MAX_REASONABLE_HOURS * 30)
                    elif progress[i] < 15:  # First 15% of progress
                        # Still cap, but less aggressively
                        remaining[i] = min(remaining[i], MAX_REASONABLE_HOURS * 60)
                    else:
                        # Regular cap for established estimates
                        remaining[i] = min(remaining[i], MAX_REASONABLE_HOURS * 120)
                    
                # Apply smoothing to the whole array
                if len(remaining) >= window_size:
                    smoothed = np.zeros_like(remaining)
                    for i in range(window_size - 1, len(remaining)):
                        smoothed[i] = np.sum(weights * remaining[i-window_size+1:i+1])
                    # Only replace values after we have enough data for smoothing
                    remaining[window_size-1:] = smoothed[window_size-1:]
            
            # Update the max remaining time for y-axis scaling
            current_max = np.max(remaining) if len(remaining) > 0 else 120
            max_remaining = max(min(current_max * 1.2, 24*60), 120)  # Cap at 24 hours, min 2 hours
            
            # Update the lines
            progress_line.set_data(times, progress)
            speed_line.set_data(times, speeds)
            remaining_line.set_data(times, remaining)
            memory_line.set_data(times, memory)
            
            # Adjust axes limits
            ax1.set_xlim(0, max(10, np.max(times) * 1.1))
            ax2.set_xlim(0, max(10, np.max(times) * 1.1))
            
            # Update memory axis limit if needed
            if len(memory) > 0 and np.max(memory) > 0:
                ax2.set_ylim(0, max(100, np.max(memory) * 1.1))
            
            # Update remaining time axis
            ax1_twin.set_ylim(0, max_remaining)
            
            # Read summary log for additional info
            summary_info = "No summary data available"
            estimation_quality = "Initializing..."
            estimation_color = "yellow"
            
            try:
                if os.path.exists(summary_log) and os.path.getsize(summary_log) > 0:
                    summary_df = pd.read_csv(summary_log)
                    if len(summary_df) > 0:
                        last_row = summary_df.iloc[-1]
                        
                        # Format the summary information
                        summary_info = (
                            f"Steps: {int(last_row['Steps_Completed'])}\n"
                            f"Best params: ω={last_row['Best_Omega']:.3f}, β={last_row['Best_Beta']:.3f}\n"
                            f"Acc. rate: {last_row['Acceptance_Rate']:.2f}\n"
                            f"Est. completion: {estimated_completion(df)}"
                        )
                        
                        # Determine estimation quality based on progress
                        latest_progress = progress[-1] if len(progress) > 0 else 0
                        if latest_progress < 5:
                            estimation_quality = "Initializing estimate..."
                            estimation_color = "yellow" 
                        elif latest_progress < 15:
                            estimation_quality = "Estimate stabilizing..."
                            estimation_color = "khaki"
                        elif latest_progress < 30:
                            estimation_quality = "Estimate improving"
                            estimation_color = "palegreen"
                        else:
                            estimation_quality = "Estimate reliable"
                            estimation_color = "lightgreen"
            except Exception as e:
                summary_info = f"Error reading summary: {str(e)}"
            
            info_text.set_text(summary_info)
            estimate_quality.set_text(estimation_quality)
            estimate_quality.set_bbox(dict(facecolor=estimation_color, alpha=0.5, boxstyle='round'))
            
            # Extract run info if available to get target epochs
            try:
                run_info_path = os.path.join(run_dir, "run_info.json")
                if os.path.exists(run_info_path):
                    with open(run_info_path, 'r') as f:
                        run_info = json.load(f)
                        walkers = run_info.get('nwalkers', 32)
                        steps = run_info.get('nsteps', 5000)
                        target_epochs = steps / walkers if walkers > 0 else 100
            except Exception:
                pass
                
        except Exception as e:
            print(f"Error updating plot: {e}")
            import traceback
            traceback.print_exc()
        
        return progress_line, speed_line, remaining_line, memory_line, info_text, estimate_quality
    
    def estimated_completion(df):
        """Calculate estimated completion time based on current progress."""
        if len(df) < 2:
            return "Calculating..."
        
        try:
            last_row = df.iloc[-1]
            remaining_min = last_row['Remaining_Min']
            
            # Apply sanity check to remaining time
            MAX_HOURS = 48
            remaining_min = min(remaining_min, MAX_HOURS * 60)
            
            # Calculate estimated completion time
            now = datetime.now()
            eta = now + timedelta(minutes=remaining_min)
            return eta.strftime("%Y-%m-%d %H:%M")
        except Exception as e:
            return f"Calculating... ({str(e)})"
    
    # Add a progress bar title
    fig.suptitle(f"Parameter Sweep Progress Monitor - {os.path.basename(run_dir)}", fontsize=14)
    
    # Try to load run info to get target epochs
    try:
        run_info_path = os.path.join(run_dir, "run_info.json")
        if os.path.exists(run_info_path):
            with open(run_info_path, 'r') as f:
                run_info = json.load(f)
                walkers = run_info.get('nwalkers', 32)
                steps = run_info.get('nsteps', 5000)
                target_epochs = steps / walkers if walkers > 0 else 100
    except Exception as e:
        print(f"Could not load run info: {e}")
    
    # Create animation
    ani = FuncAnimation(
        fig, update, frames=None, 
        blit=True, interval=refresh_rate*1000, 
        cache_frame_data=False  # Add this parameter to fix the warning
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nExiting monitor...")
        sys.exit(0)

def monitor_terminal_mode(run_dir, refresh_rate=5):
    """Monitor progress in terminal mode."""
    progress_log = os.path.join(run_dir, "progress_log.txt")
    summary_log = os.path.join(run_dir, "summary_log.txt")
    
    print(f"Launching terminal progress monitor for {run_dir}")
    print("Press Ctrl+C to exit")
    
    # Initialize tracking variables
    last_step = 0
    start_time = time.time()
    last_update = 0
    
    # ANSI color codes for terminal output
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    try:
        while True:
            current_time = time.time()
            
            # Only update at the specified interval
            if current_time - last_update < refresh_rate:
                time.sleep(0.5)  # Check more frequently but only update at interval
                continue
                
            last_update = current_time
            
            # Clear terminal (works on most terminals)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"{BOLD}===== Parameter Sweep Monitor ====={RESET}")
            print(f"{BOLD}Run directory:{RESET} {run_dir}")
            print(f"{BOLD}Elapsed time:{RESET} {(current_time - start_time) / 60:.1f} minutes")
            print(f"{BOLD}{'='*35}{RESET}")
            
            try:
                # Check if logs exist
                if not os.path.exists(progress_log):
                    print(f"{YELLOW}Waiting for progress log to be created...{RESET}")
                    time.sleep(refresh_rate)
                    continue
                    
                # Read progress log
                try:
                    df_progress = pd.read_csv(progress_log)
                    if len(df_progress) == 0:
                        print(f"{YELLOW}Progress log exists but contains no data yet.{RESET}")
                        time.sleep(refresh_rate)
                        continue
                except Exception as e:
                    print(f"{YELLOW}Error reading progress log: {e}{RESET}")
                    time.sleep(refresh_rate)
                    continue
                
                # Extract latest progress
                latest = df_progress.iloc[-1]
                
                # Calculate progress percentage
                if 'Epoch' in latest:
                    # Try to get target epochs from run_info.json
                    target_epochs = 100  # Default
                    try:
                        run_info_path = os.path.join(run_dir, "run_info.json")
                        if os.path.exists(run_info_path):
                            with open(run_info_path, 'r') as f:
                                run_info = json.load(f)
                                walkers = run_info.get('nwalkers', 32)
                                steps = run_info.get('nsteps', 5000)
                                target_epochs = steps / walkers if walkers > 0 else 100
                    except Exception:
                        pass
                        
                    progress_pct = (latest['Epoch'] / target_epochs) * 100
                else:
                    progress_pct = 0
                
                # Format and display progress information
                progress_bar = "█" * int(progress_pct / 2) + "░" * (50 - int(progress_pct / 2))
                print(f"{BOLD}Progress:{RESET} {progress_bar} {progress_pct:.1f}%")
                
                # Calculate time estimates with sanity checks
                remaining_min = min(latest.get('Remaining_Min', float('inf')), 24*60)  # Cap at 24 hours
                eta = datetime.now() + timedelta(minutes=remaining_min)
                eta_str = eta.strftime("%Y-%m-%d %H:%M")
                
                # Determine reliability of estimate
                reliability = ""
                if progress_pct < 5:
                    reliability = f"{YELLOW}(initializing){RESET}"
                elif progress_pct < 15:
                    reliability = f"{YELLOW}(stabilizing){RESET}"
                
                print(f"{BOLD}Step:{RESET} {int(latest.get('Step', 0))}")
                print(f"{BOLD}Epoch:{RESET} {latest.get('Epoch', 0):.2f}")
                print(f"{BOLD}Processing speed:{RESET} {latest.get('Batch_Speed', 0):.1f} samples/s")
                print(f"{BOLD}Memory usage:{RESET} {latest.get('Memory_MB', 0):.1f} MB")
                print(f"{BOLD}Est. remaining time:{RESET} {int(remaining_min//60)}h {int(remaining_min%60)}m {reliability}")
                print(f"{BOLD}Est. completion:{RESET} {eta_str} {reliability}")
                
                # Check if summary log exists
                if os.path.exists(summary_log) and os.path.getsize(summary_log) > 0:
                    try:
                        df_summary = pd.read_csv(summary_log)
                        if len(df_summary) > 0:
                            latest_summary = df_summary.iloc[-1]
                            
                            # Extract best parameters
                            best_omega = latest_summary.get('Best_Omega', 'N/A')
                            best_beta = latest_summary.get('Best_Beta', 'N/A')
                            best_score = latest_summary.get('Best_Score', 'N/A')
                            acceptance = latest_summary.get('Acceptance_Rate', 'N/A')
                            
                            print(f"\n{BOLD}===== Current Results ====={RESET}")
                            print(f"{BOLD}Best parameters:{RESET} ω={best_omega}, β={best_beta}")
                            print(f"{BOLD}Best score:{RESET} {best_score}")
                            print(f"{BOLD}Acceptance rate:{RESET} {acceptance}")
                    except Exception as e:
                        print(f"\n{YELLOW}Error reading summary log: {e}{RESET}")
                else:
                    print(f"\n{YELLOW}Waiting for summary information...{RESET}")
                
                # Update last known step
                last_step = int(latest.get('Step', 0))
                
            except Exception as e:
                print(f"{YELLOW}Error updating progress display: {e}{RESET}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{BOLD}Press Ctrl+C to exit monitor{RESET}")
            
            # Sleep until next update
            time.sleep(1)  # Short sleep to allow for Ctrl+C
            
    except KeyboardInterrupt:
        print("\nExiting monitor...")
        sys.exit(0)
    except Exception as e:
        print(f"Error in monitor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def list_run_directories():
    """List available run directories."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results', 'parameter_sweep', 'runs')
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return []
    
    run_dirs = []
    for d in os.listdir(results_dir):
        full_path = os.path.join(results_dir, d)
        if os.path.isdir(full_path) and d.startswith('run_'):
            run_dirs.append((d, full_path))
    
    return sorted(run_dirs, reverse=True)  # Most recent first

def select_run_directory():
    """Let the user select a run directory from the list."""
    run_dirs = list_run_directories()
    
    if not run_dirs:
        print("No run directories found.")
        return None
    
    print("\nAvailable Parameter Sweep Run Directories:")
    for i, (dirname, _) in enumerate(run_dirs):
        # Extract date and time from directory name
        if dirname.startswith('run_') and len(dirname) > 4:
            date_time = dirname[4:]  # Remove 'run_' prefix
            print(f"[{i+1}] {date_time}")
        else:
            print(f"[{i+1}] {dirname}")
    
    choice = input("\nEnter number to monitor (or 'q' to quit): ")
    if choice.lower() == 'q':
        return None
    
    try:
        index = int(choice) - 1
        if 0 <= index < len(run_dirs):
            return run_dirs[index][1]  # Return the full path
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None

def find_latest_run(base_dir=None):
    """Find the most recent run directory."""
    if base_dir is None:
        # Try to locate the default directory structure
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "results", "parameter_sweep", "runs")
        savepoints_dir = os.path.join(script_dir, "results", "parameter_sweep", "savepoints")
        
        # Check if either directory exists
        if not os.path.exists(base_dir) and os.path.exists(savepoints_dir):
            base_dir = savepoints_dir
        elif not os.path.exists(base_dir) and not os.path.exists(savepoints_dir):
            print("Could not find default run directories.")
            return None
    
    # Find all directories that match the run pattern
    run_pattern = re.compile(r'run_\d{8}_\d{6}')
    run_dirs = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and run_pattern.match(item):
            run_dirs.append(item_path)
    
    if not run_dirs:
        # Try to find in savepoints directory if main directory has no runs
        if base_dir.endswith("runs"):
            savepoints_dir = os.path.join(os.path.dirname(base_dir), "savepoints")
            if os.path.exists(savepoints_dir):
                for item in os.listdir(savepoints_dir):
                    item_path = os.path.join(savepoints_dir, item)
                    if os.path.isdir(item_path) and run_pattern.match(item):
                        run_dirs.append(item_path)
    
    if not run_dirs:
        print("No run directories found.")
        return None
    
    # Sort by creation time (most recent first)
    latest_dir = max(run_dirs, key=os.path.getctime)
    return latest_dir

def main():
    parser = argparse.ArgumentParser(description="Monitor parameter sweep validation progress")
    parser.add_argument("--run_dir", type=str, help="Directory containing progress logs (if not specified, will prompt)")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh rate in seconds (default: 5)")
    parser.add_argument("--plot", action="store_true", help="Use graphical plot mode instead of text mode")
    
    args = parser.parse_args()
    
    run_dir = args.run_dir
    
    # If no run directory specified, let user select one
    if not run_dir:
        run_dir = select_run_directory()
        if not run_dir:
            print("No run directory selected. Exiting.")
            return
    
    # Check if the directory exists and contains progress logs
    if not os.path.exists(run_dir):
        print(f"Run directory not found: {run_dir}")
        return
    
    progress_log = os.path.join(run_dir, "progress_log.txt")
    if not os.path.exists(progress_log):
        print(f"Progress log not found in {run_dir}")
        return
    
    # Monitor in the appropriate mode
    if args.plot:
        try:
            import matplotlib
            monitor_plot_mode(run_dir, args.refresh)
        except ImportError:
            print("Matplotlib not available. Using text mode instead.")
            monitor_text_mode(run_dir, args.refresh)
    else:
        monitor_text_mode(run_dir, args.refresh)

if __name__ == "__main__":
    main()
