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
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Progress (%)')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Create a second y-axis for speed
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Samples/second')
    ax1_twin.set_ylim(0, 100)  # Will be adjusted
    
    # Parameter convergence plot
    omega_line, = ax2.plot([], [], 'g-', label='ω (Omega)')
    beta_line, = ax2.plot([], [], 'm-', label='β (Beta)')
    score_line, = ax2.plot([], [], 'k--', label='Score', alpha=0.5)
    ax2.set_xlim(0, 10)  # Will be adjusted
    ax2.set_ylim(-1, 4)  # Will be adjusted
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Parameter Values')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # Create a second y-axis for score
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Score')
    ax2_twin.set_ylim(-1, 1)  # Will be adjusted
    
    # Progress data
    time_data = []
    progress_data = []
    speed_data = []
    
    # Parameter data
    summary_time = []
    omega_data = []
    beta_data = []
    score_data = []
    
    # Text annotations
    progress_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, va='top')
    eta_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes, va='top')
    param_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, va='top')
    
    def init():
        progress_line.set_data([], [])
        speed_line.set_data([], [])
        omega_line.set_data([], [])
        beta_line.set_data([], [])
        score_line.set_data([], [])
        return progress_line, speed_line, omega_line, beta_line, score_line, progress_text, eta_text, param_text
    
    def update(frame):
        # Read progress data
        try:
            if os.path.exists(progress_log):
                df = pd.read_csv(progress_log)
                if not df.empty:
                    # Extract data
                    time_data.clear()
                    progress_data.clear()
                    speed_data.clear()
                    
                    time_data.extend(df['Elapsed_Min'].values)
                    progress_data.extend(df['Progress_Pct'].values if 'Progress_Pct' in df.columns else [0] * len(df))
                    speed_data.extend(df['Batch_Speed'].values)
                    
                    # Update progress plot
                    progress_line.set_data(time_data, progress_data)
                    speed_line.set_data(time_data, speed_data)
                    
                    # Adjust axis limits if needed
                    if time_data:
                        ax1.set_xlim(0, max(10, max(time_data) * 1.1))
                    if speed_data:
                        max_speed = max(speed_data) * 1.1
                        ax1_twin.set_ylim(0, max_speed)
                    
                    # Update progress text
                    latest = df.iloc[-1]
                    progress_pct = latest.get('Progress_Pct', 0) if 'Progress_Pct' in latest else 0
                    elapsed = latest.get('Elapsed_Min', 0)
                    remaining = latest.get('Remaining_Min', 0)
                    
                    progress_text.set_text(f'Progress: {progress_pct:.1f}% - Elapsed: {format_time(elapsed)}')
                    est_completion = datetime.now() + timedelta(minutes=remaining)
                    eta_text.set_text(f'ETA: {est_completion.strftime("%Y-%m-%d %H:%M:%S")}')
        except Exception as e:
            print(f"Error updating progress plot: {e}")
        
        # Read summary data
        try:
            if os.path.exists(summary_log):
                df = pd.read_csv(summary_log)
                if not df.empty:
                    # Extract data
                    summary_time.clear()
                    omega_data.clear()
                    beta_data.clear()
                    score_data.clear()
                    
                    summary_time.extend(df['Elapsed_Min'].values)
                    omega_data.extend(df['Best_Omega'].values)
                    beta_data.extend(df['Best_Beta'].values)
                    score_data.extend(df['Best_Score'].values)
                    
                    # Update parameter plot
                    omega_line.set_data(summary_time, omega_data)
                    beta_line.set_data(summary_time, beta_data)
                    score_line.set_data(summary_time, score_data)
                    
                    # Adjust axis limits if needed
                    if summary_time:
                        ax2.set_xlim(0, max(10, max(summary_time) * 1.1))
                    
                    if omega_data and beta_data:
                        min_param = min(min(omega_data), min(beta_data)) * 1.1
                        max_param = max(max(omega_data), max(beta_data)) * 1.1
                        ax2.set_ylim(min_param, max_param)
                    
                    if score_data:
                        min_score = min(score_data) * 1.1
                        max_score = max(score_data) * 1.1
                        ax2_twin.set_ylim(min_score, max_score)
                    
                    # Update parameter text
                    if omega_data and beta_data and score_data:
                        param_text.set_text(f'Latest: ω={omega_data[-1]:.4f}, β={beta_data[-1]:.4f}, Score={score_data[-1]:.4f}')
        except Exception as e:
            print(f"Error updating parameter plot: {e}")
        
        return progress_line, speed_line, omega_line, beta_line, score_line, progress_text, eta_text, param_text
    
    # Fix the animation warning by setting cache_frame_data=False
    # This prevents building up an unbounded cache during long monitoring sessions
    ani = FuncAnimation(fig, update, frames=None, init_func=init, blit=True, 
                        interval=refresh_rate*1000, cache_frame_data=False, save_count=100)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.show()

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
