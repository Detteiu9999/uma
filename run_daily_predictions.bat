@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

echo ============================================================
echo  [1/3] Delete existing CSV files
echo ============================================================
for %%D in (CSV_predict odds predictions) do (
    if exist "%%D\*.csv" (
        echo Deleting %%D\*.csv
        del /f /q "%%D\*.csv"
        if exist "%%D\*.csv" (
            echo [ERROR] Could not delete all CSV files in %%D.
            goto ERROR
        )
    )
)

echo.
echo ============================================================
echo  [2/3] Fetch today's race data ^(JST^)
echo ============================================================
python -X utf8 uma.py
if errorlevel 1 (
    echo [ERROR] uma.py failed.
    goto ERROR
)
if not exist "CSV_predict\*.csv" (
    echo [ERROR] No CSV files were fetched. Check today's races and the fetch output.
    goto ERROR
)

echo.
echo ============================================================
echo  [3/3] Generate today's predictions
echo ============================================================
python -X utf8 predict_model.py --predict-only
if errorlevel 1 (
    echo [ERROR] predict_model.py failed.
    goto ERROR
)

echo.
echo Done: prediction CSV files saved to predictions.
popd
endlocal
exit /b 0

:ERROR
popd
endlocal
exit /b 1
