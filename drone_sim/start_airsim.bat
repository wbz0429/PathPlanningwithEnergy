@echo off
REM 快速启动AirSim

echo Starting AirSim Blocks environment...
start "" "E:\Sim\Blocks\Blocks\WindowsNoEditor\blocks.exe"

echo.
echo AirSim is starting...
echo Please wait for the environment to load completely.
echo.
echo Once you see the drone in the environment, you can:
echo   1. Run: launch_visualization.bat
echo   2. Or manually run: python fly_with_simple_visualization.py
echo.

pause
