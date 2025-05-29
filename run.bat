@echo off

:: Deactivate any active env
call conda deactivate

:: Activate your environment
call conda activate facefusion

:: Run your script
python facefusion.py run --open-browser --execution-providers cuda --execution-thread-count 32 --execution-queue-count 2

:: Optional: Pause to see output
pause
