@echo off
title BlackbirdSDK Keepalive Backend
echo Starting BlackbirdSDK Keepalive Backend...
echo Port: 5012
echo Keepalive Code: 013093f7-a76b-40ab-8388-b37957f031de
echo.
"C:\decompute-app\.venv_obf_test\Scripts\python.exe" "C:\decompute-app\sdk\blackbird_sdk\backends\windows\decompute.py"
echo.
echo Backend stopped. Press any key to close this window...
pause >nul
