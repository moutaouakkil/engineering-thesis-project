@echo off
echo === LLM Analysis Visualization Setup ===

:: Install Python packages
echo Installing Python dependencies...
pip install -r requirements.txt && echo [OK] Dependencies installed || echo [FAIL] Dependencies installation failed

echo === Setup complete ===
echo Installed: pandas, numpy, matplotlib, seaborn

echo.
set /p run_script=Would you like to proceed with running 'visualize.py'? (Y/N): 
if /i "%run_script%"=="Y" (
    python visualize.py
) else (
    echo Script execution skipped
)
pause 