@echo off
setlocal

:: Определяем путь к папке со скриптом
set "APP=%~dp0"
if "%APP:~-1%"=="\" set "APP=%APP:~0,-1%"

:: Определяем путь к Python внутри виртуального окружения
set "PYTHON_EXE=%APP%\.env\Scripts\python.exe"

:: Устанавливаем переменные окружения для кэшей и весов
set "HF_HOME=%APP%\weights\hf"
set "CRAFT_TEXT_DETECTOR__WEIGHTS_DIR=%APP%\weights\craft"
set "ULTRALYTICS_SETTINGS_DIR=%APP%\weights\ultralytics"
set "YOLO_MODEL_PATH=%APP%\weights\bubbles_yolo.pt"
set "PANEL_MODEL_PATH=%APP%\weights\magiv3"

echo Starting API server...

:: Запускаем Uvicorn, используя Python из виртуального окружения
:: Ключевой момент: `uvicorn` запускается как модуль (-m)
"%PYTHON_EXE%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

endlocal