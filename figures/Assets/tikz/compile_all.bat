@echo off
REM Compile all standalone TikZ figures to PDF
REM Requires pdflatex (TeX Live or MiKTeX)

cd /d "%~dp0"

echo Compiling standalone TikZ figures...
echo.

for %%f in (*.tex) do (
    echo Compiling %%f...
    pdflatex -interaction=nonstopmode "%%f" > nul 2>&1
    if errorlevel 1 (
        echo   [FAILED] %%f
    ) else (
        echo   [OK] %%f
    )
)

echo.
echo Cleaning up auxiliary files...
del /q *.aux *.log 2>nul

echo.
echo Done! PDF files created.
dir /b *.pdf
echo.
pause
