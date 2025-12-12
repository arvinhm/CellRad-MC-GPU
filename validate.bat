@echo off
REM ============================================================================
REM CellRad-MC Validation Suite (Windows)
REM ============================================================================
REM Run all physics and isotope validations
REM Outputs saved to validation_results/

echo ==============================================
echo CellRad-MC Validation Suite
echo ==============================================
echo.

REM Create output directory
if not exist validation_results mkdir validation_results

REM Run publication validation
echo [1/4] Running Publication Validation...
python publication_validation.py
echo       Done!
echo.

REM Run isotope validation
echo [2/4] Running Isotope Validation...
python isotope_validation.py
echo       Done!
echo.

REM Run DNA damage validation
if exist dna_damage_validation.py (
    echo [3/4] Running DNA Damage Validation...
    python dna_damage_validation.py
    echo       Done!
    echo.
) else (
    echo [3/4] Skipping DNA Damage Validation
    echo.
)

REM Run track structure validation
if exist track_structure_validation.py (
    echo [4/4] Running Track Structure Validation...
    python track_structure_validation.py
    echo       Done!
    echo.
) else (
    echo [4/4] Skipping Track Structure Validation
    echo.
)

echo ==============================================
echo Validation Complete!
echo ==============================================
echo.
echo Results saved to: validation_results\
echo.
dir /b validation_results\*.png 2>nul
dir /b validation_results\*.csv 2>nul
dir /b validation_results\*.json 2>nul
echo.
pause

