@echo off
cd /d "%~dp0"
title Instalador - Reconocimiento de Expresiones

echo ==========================================
echo   INSTALADOR DEL PROYECTO
echo   Reconocimiento de Expresiones
echo ==========================================
echo.

echo [1/4] Comprobando Python 3.11...

py -3.11 --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo ERROR: Python 3.11 no esta instalado.
    echo.
    echo Instala Python 3.11 de 64 bits y vuelve a ejecutar este archivo.
    echo.
    pause
    exit /b 1
)

echo Python 3.11 encontrado.
echo.

echo [2/4] Creando entorno virtual...

if exist ".venv" (
    echo El entorno .venv ya existe.
) else (
    py -3.11 -m venv .venv
)

echo.

echo [3/4] Actualizando pip...

call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip

echo.

echo [4/4] Instalando librerias...

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Fallo la instalacion de librerias. Revisa tu conexion a internet.
    pause
    exit /b 1
)

echo.
echo Descargando modelos de emociones (solo la primera vez)...
python -c "from hsemotion_onnx.facial_emotions import HSEmotionRecognizer; HSEmotionRecognizer('enet_b0_8_best_afew')"

echo.
echo ==========================================
echo   INSTALACION TERMINADA
echo ==========================================
echo.
echo Para ejecutar el programa haz doble clic en ejecutar.bat
echo o, desde la consola:
echo.
echo   .venv\Scripts\activate
echo   python prueba_emociones.py
echo.

pause