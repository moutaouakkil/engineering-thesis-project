@echo off
echo === LLM Analysis Setup ===

:: Install Python packages
echo Installing Python dependencies...
pip install -r requirements.txt && echo [OK] Dependencies installed || echo [FAIL] Dependencies installation failed

echo === Setup complete ===
echo Installed: pandas, numpy, scikit-learn

echo.
set /p run_script=Would you like to proceed with running 'llm_analysis.py'? (Y/N): 
if /i "%run_script%"=="Y" (
    python llm_analysis.py
) else (
    echo Script execution skipped
)
pause 