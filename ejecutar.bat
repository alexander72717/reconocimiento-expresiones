@echo off
title Reconocimiento de Expresiones
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No se encontro el entorno .venv. Ejecuta primero instalar_proyecto.bat
    pause
    exit /b 1
)

set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" prueba_emociones.py %*
if errorlevel 1 pause
