@echo off
echo === Additional Results and Visualizations Setup ===

:: Install Python packages
echo Installing Python dependencies...
pip install -r requirements.txt && echo [OK] Dependencies installed || echo [FAIL] Dependencies installation failed

echo === Setup complete ===
echo Installed: pandas, numpy, matplotlib, seaborn, scikit-learn

echo.
set /p run_script=Would you like to proceed with running 'analyze_consistency.py'? (Y/N): 
if /i "%run_script%"=="Y" (
    python analyze_consistency.py
) else (
    echo Script execution skipped
)
pause 