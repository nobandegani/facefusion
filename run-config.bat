@echo off

:: Deactivate any active env
call conda deactivate

:: Activate your environment
call conda activate facefusion

:: Run your script
python facefusion.py run --open-browser --config-path "E:\\NIL\\spark\\python\\InoGenie\\configs\\facefusion.ini"

:: Optional: Pause to see output
pause
