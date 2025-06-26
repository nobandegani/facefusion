@echo off

:: Deactivate any active env
call conda deactivate

:: Activate your environment
call conda activate facefusion

:: Run your script
python install.py --onnxruntime cuda

:: Optional: Pause to see output
pause
