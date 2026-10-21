@echo off
rem Gera dist\WLSS3D\WLSS3D.exe (rode dentro do venv com as dependencias instaladas)
pip install pyinstaller
pyinstaller --noconfirm --name WLSS3D ^
  --collect-all moderngl --collect-all glcontext --collect-binaries onnxruntime ^
  --collect-submodules sounddevice --collect-data sounddevice ^
  run.py
echo.
echo Copie o ffmpeg.exe para dist\WLSS3D antes de distribuir.
