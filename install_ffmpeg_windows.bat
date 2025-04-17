@echo off
echo Genesis-Sphere FFmpeg Installer
echo ==============================
echo.
echo This script will download and install FFmpeg for Windows.
echo.

set INSTALL_DIR=%USERPROFILE%\ffmpeg
set PATH_TO_ADD=%INSTALL_DIR%\bin
set DOWNLOAD_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
set ZIP_FILE=%TEMP%\ffmpeg.zip

echo Step 1: Creating installation directory...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

echo Step 2: Downloading FFmpeg...
powershell -Command "Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%ZIP_FILE%'"

echo Step 3: Extracting FFmpeg...
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%TEMP%\ffmpeg_extract'"
powershell -Command "Get-ChildItem -Path '%TEMP%\ffmpeg_extract\*' -Directory | ForEach-Object { Copy-Item -Path \"$_\*\" -Destination '%INSTALL_DIR%' -Recurse -Force }"

echo Step 4: Adding FFmpeg to PATH...
setx PATH "%PATH%;%PATH_TO_ADD%"

echo Step 5: Cleaning up...
del "%ZIP_FILE%"
rmdir /s /q "%TEMP%\ffmpeg_extract"

echo.
echo FFmpeg installed successfully to %INSTALL_DIR%
echo Please restart your command prompt/terminal for the PATH changes to take effect.
echo.
pause