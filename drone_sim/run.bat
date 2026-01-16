@echo off
echo ========================================
echo    无人机仿真项目 - Phase 1
echo ========================================
echo.

REM 检查仿真器是否运行
echo [1] 请确保 Colosseum/AirSim 仿真器已经启动
echo     (运行 CityEnviron.exe 或其他环境)
echo.
pause

REM 运行主程序
echo [2] 正在启动主控制程序...
echo.
cd /d "E:\毕业设计\drone_sim"
python main_control.py

echo.
echo ========================================
echo 程序已结束，数据已保存到 logs 文件夹
echo ========================================
pause
