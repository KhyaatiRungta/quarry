@echo off
rem Quarry UI launcher - starts the server if needed, then opens the browser.
rem Double-click this file any time you want the local site.
cd /d "%~dp0"

rem Already running? Just open it.
curl -s -o nul http://127.0.0.1:8477/api/health
if %errorlevel%==0 (
    start "" http://127.0.0.1:8477
    goto :done
)

rem Start the server in its own window (keep it open so you can stop it later).
start "Quarry server - close this window to stop" cmd /k "uvicorn quarry.api:app --host 127.0.0.1 --port 8477"

set /a tries=0
:wait
timeout /t 1 /nobreak >nul
set /a tries+=1
curl -s -o nul http://127.0.0.1:8477/api/health
if %errorlevel%==0 (
    start "" http://127.0.0.1:8477
    goto :done
)
if %tries% lss 20 goto wait
echo Server did not start - check the "Quarry server" window for errors.
pause
goto :eof

:done
echo Quarry UI opened at http://127.0.0.1:8477
