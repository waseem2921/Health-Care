@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo ========================================
echo PulseAnalysis AI - One Click Startup
echo ========================================
echo Project: %CD%
echo.

if not exist "manage.py" (
	echo [ERROR] manage.py not found. Run this file from the project root folder.
	goto :fail
)

set "PY_EXE="
if exist "D:\Python\python.exe" set "PY_EXE=D:\Python\python.exe"
if not defined PY_EXE (
	where py >nul 2>&1
	if %ERRORLEVEL%==0 set "PY_EXE=py -3"
)
if not defined PY_EXE (
	where python >nul 2>&1
	if %ERRORLEVEL%==0 set "PY_EXE=python"
)
if not defined PY_EXE (
	echo [ERROR] Python not found. Install Python 3.10+ and rerun.
	goto :fail
)

echo [1/7] Python runtime
call %PY_EXE% --version
if %ERRORLEVEL% neq 0 (
	echo [ERROR] Python runtime check failed.
	goto :fail
)
echo.

echo [2/7] Virtual environment
if not exist ".venv\Scripts\python.exe" (
	call %PY_EXE% -m venv .venv
	if !ERRORLEVEL! neq 0 (
		echo [ERROR] Failed to create virtual environment.
		goto :fail
	)
)
set "VENV_PY=.venv\Scripts\python.exe"
echo Using !VENV_PY!
echo.

echo [3/7] Dependencies
call "!VENV_PY!" -m pip install --upgrade pip >nul
call "!VENV_PY!" -m pip install -r requirements.txt
if !ERRORLEVEL! neq 0 (
	echo [WARN] Dependency installation failed - possibly offline. Checking existing install...
	call "!VENV_PY!" -c "import django, dj_database_url, dotenv, cloudinary, cloudinary_storage, pandas, sklearn, matplotlib, numpy, joblib" >nul 2>&1
	if !ERRORLEVEL! neq 0 (
		echo [ERROR] Required packages are missing and could not be installed.
		echo Connect to internet once, run RUN_APP.bat again, then offline startup will work.
		goto :fail
	)
	echo Existing packages detected. Continuing in offline mode.
)
echo.

echo [4/7] Environment file
if not exist ".env" (
	if exist ".env.example" (
		copy /Y ".env.example" ".env" >nul
		echo Created .env from .env.example
	) else (
		(
			echo SECRET_KEY=django-insecure-local-key
			echo DEBUG=True
			echo ALLOWED_HOSTS=127.0.0.1,localhost
			echo ALLOW_SQLITE_FALLBACK=True
			echo USE_LOCAL_DB=False
			echo USE_LOCAL_STORAGE=False
			echo FORCE_OFFLINE=False
			echo LOCAL_DB_PATH=local_offline.db
		) > .env
		echo Created minimal .env
	)
) else (
	echo .env already exists, using current configuration
)
echo.

echo [5/7] Database setup (auto-offline fallback enabled)
call "!VENV_PY!" manage.py migrate
if !ERRORLEVEL! neq 0 (
	echo [ERROR] Migration failed.
	echo Fix migration errors above and rerun RUN_APP.bat.
	goto :fail
)

echo Applying migrations to local offline SQLite as well...
set "_USE_LOCAL_DB_PREV=%USE_LOCAL_DB%"
set "USE_LOCAL_DB=True"
call "!VENV_PY!" manage.py migrate
if !ERRORLEVEL! neq 0 (
	echo [ERROR] Offline SQLite migration failed.
	echo Local fallback database is not ready; offline login may fail.
	goto :fail
)
set "USE_LOCAL_DB=%_USE_LOCAL_DB_PREV%"

call "!VENV_PY!" manage.py seed_shared_data
if !ERRORLEVEL! neq 0 (
	echo [WARN] seed_shared_data failed. Continuing without seed data.
)
echo.

echo [6/7] Starting server
echo URL: http://127.0.0.1:5000
echo Press Ctrl+C to stop
echo App will work seamlessly even if internet goes down!
echo.

start "" http://127.0.0.1:5000

call "!VENV_PY!" manage.py runserver 127.0.0.1:5000

if !ERRORLEVEL! neq 0 (
	echo.
	echo [ERROR] Django server stopped with an error.
	echo Check messages above. If this happened at first run, connect internet and rerun once.
	goto :fail
)

goto :end

:fail
echo.
echo Startup did not complete successfully.

:end
pause


