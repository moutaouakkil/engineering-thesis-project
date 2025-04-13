@echo off
echo === LLM Query Tool Setup ===

:: Install Python packages
echo Installing Python dependencies...
pip install "tqdm>=4.66.0" && echo [OK] tqdm installed || echo [FAIL] tqdm installation failed
pip install ollama && echo [OK] ollama installed || echo [FAIL] ollama installation failed

:: Install Ollama models
echo Installing Ollama models...
set models=aya deepseek-r1 llama3.2 falcon3 phi qwen gemma

for %%m in (%models%) do (
    echo Pulling: %%m
    ollama pull %%m && echo [OK] %%m pulled || echo [FAIL] %%m pull failed
)

echo === Setup complete ===
echo Installed: tqdm, ollama, models (aya, deepseek-r1, llama3.2, falcon3, phi, qwen, gemma)

echo.
set /p run_script=Would you like to proceed with running 'query_llms.py'? (Y/N): 
if /i "%run_script%"=="Y" (
    python query_llms.py
) else (
    echo Script execution skipped
)
pause