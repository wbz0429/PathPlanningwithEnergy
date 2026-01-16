@echo off
echo Installing dependencies for drone_sim...
echo.

echo [1/5] Installing numpy...
pip install numpy

echo.
echo [2/5] Installing airsim...
pip install airsim

echo.
echo [3/5] Installing opencv-python...
pip install opencv-python

echo.
echo [4/5] Installing matplotlib pandas...
pip install matplotlib pandas

echo.
echo [5/5] Installing scipy...
pip install scipy

echo.
echo ============================================
echo Installation completed!
echo ============================================
echo.
echo You can now run: python fly_planned_path.py
pause
