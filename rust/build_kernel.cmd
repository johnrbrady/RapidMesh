@echo off
REM Build and install the RapidMesh collapse kernel - WP-3.4b, Round 11.
REM ASCII only.
REM
REM   rust\build_kernel.cmd            build a wheel into rust\kernel\target\wheels
REM   rust\build_kernel.cmd develop    build and install into the project venv
REM
REM Requires a Rust toolchain and, on Windows, the MSVC linker (Visual Studio
REM Build Tools, "Desktop development with C++"). NEITHER IS REQUIRED TO RUN
REM RAPIDMESH: with no kernel built, `rapidmesh.decimate` uses its own Python
REM sweep and the tree behaves exactly as it did at Round 10.
REM
REM Run this from cmd.exe, not from Git Bash. Git Bash puts a GNU `link` on
REM PATH which shadows MSVC's `link.exe`, and rustc invokes the linker by name;
REM the failure is an obscure "extra operand" error from coreutils rather than
REM anything that mentions Rust.

setlocal
set REPO=%~dp0..
set PYTHON=%REPO%\.venv\Scripts\python.exe
set MANIFEST=%REPO%\rust\kernel\Cargo.toml

if not exist "%PYTHON%" (
    echo ERROR: project venv not found at %PYTHON%
    exit /b 1
)

where cargo >nul 2>&1
if errorlevel 1 (
    echo ERROR: cargo not on PATH. Install rustup, then reopen the shell.
    exit /b 1
)

if /I "%~1"=="develop" (
    echo Building and installing rapidmesh_kernel into the project venv...
    "%PYTHON%" -m maturin develop --release -m "%MANIFEST%"
) else (
    echo Building a wheel for rapidmesh_kernel...
    "%PYTHON%" -m maturin build --release -m "%MANIFEST%"
)
if errorlevel 1 (
    echo.
    echo BUILD FAILED. RapidMesh still runs: the Python sweep is the fallback.
    exit /b 1
)

echo.
echo Done. Check which sweep the tree will use with:
echo     "%PYTHON%" -c "from rapidmesh import decimate_kernel; print(decimate_kernel.describe())"
endlocal
