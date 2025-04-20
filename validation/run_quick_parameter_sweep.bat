@echo off
echo Running parameter sweep validation with quick mode and enhanced progress tracking...
python parameter_sweep_validation.py --quick_run --max_time 120 --summary_interval 5 --enhanced_progress
echo Parameter sweep completed.
pause
