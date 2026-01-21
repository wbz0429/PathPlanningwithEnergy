@echo off
REM 自动启动AirSim并运行可视化脚本

echo ========================================
echo   AirSim + Visualization Launcher
echo ========================================
echo.

REM 检查AirSim是否已经在运行
tasklist /FI "IMAGENAME eq blocks.exe" 2>NUL | find /I /N "blocks.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] AirSim is already running
) else (
    echo [1] Starting AirSim Blocks environment...
    start "" "E:\Sim\Blocks\Blocks\WindowsNoEditor\blocks.exe"
    echo     Waiting 15 seconds for AirSim to initialize...
    timeout /t 15 /nobreak >nul
    echo     [OK] AirSim should be ready now
)

echo.
echo [2] Activating conda environment...
call conda activate drone
if %ERRORLEVEL% NEQ 0 (
    echo [X] Failed to activate conda environment 'drone'
    echo     Please run: conda create -n drone python=3.9
    pause
    exit /b 1
)

echo.
echo [3] Testing AirSim connection...
python connect_test.py
if %ERRORLEVEL% NEQ 0 (
    echo [X] Failed to connect to AirSim
    echo     Please make sure AirSim is running
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Choose visualization mode:
echo ========================================
echo   1. Real AirSim Flight (Recommended)
echo   2. Simulated Data (Quick Test)
echo   3. Exit
echo ========================================
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo [4] Running real AirSim flight with visualization...
    python fly_with_simple_visualization.py
) else if "%choice%"=="2" (
    echo.
    echo [4] Running simulated visualization...
    python generate_simple_visualization.py
) else if "%choice%"=="3" (
    echo.
    echo Exiting...
    exit /b 0
) else (
    echo.
    echo [X] Invalid choice
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Visualization Complete!
echo ========================================
echo.
echo Output files:
if "%choice%"=="1" (
    echo   - airsim_flight_visualization.mp4
    echo   - airsim_flight_final.png
    echo   - airsim_flight_frames/
) else (
    echo   - simple_3d_visualization.mp4
    echo   - simple_3d_final.png
    echo   - simple_3d_frames/
)
echo.

set /p open="Open visualization video? (y/n): "
if /i "%open%"=="y" (
    if "%choice%"=="1" (
        start airsim_flight_visualization.mp4
    ) else (
        start simple_3d_visualization.mp4
    )
)

echo.
echo Done!
pause
