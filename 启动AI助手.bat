@echo off
chcp 65001 > nul
echo ========================================
echo       斗地主AI助手 启动中...
echo ========================================
echo.

cd /d "%~dp0"
"C:\Users\30330\miniconda3\envs\doudizhu\python.exe" main.py

if %errorlevel% neq 0 (
    echo.
    echo 程序异常退出，错误代码: %errorlevel%
    pause
)
