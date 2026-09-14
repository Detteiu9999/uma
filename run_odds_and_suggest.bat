@echo off
rem ============================================================
rem run_odds_and_suggest.bat
rem
rem Usage:
rem   run_odds_and_suggest.bat                     ... ?????E?S??E?S???[?X
rem   run_odds_and_suggest.bat --race 11           ... 11???[?X???
rem   run_odds_and_suggest.bat --place 1           ... ?D?y???
rem   run_odds_and_suggest.bat --date 20260906     ... ???t?w??
rem   ?? ?????g?????????\ (??: --place 1 --race 11)
rem ============================================================

setlocal
cd /d "%~dp0"

rem ???????????
set DATE_OPT=
set PLACE_OPT=
set PLACE_CODE_OPT=
set RACE_OPT=

rem ?????????[?v
:PARSE_ARGS
if "%~1"=="" goto RUN
if "%~1"=="--date" (
    set "DATE_OPT=--date %2"
    shift & shift & goto PARSE_ARGS
)
if "%~1"=="--place" (
    set "PLACE_OPT=--place %2"
    set "PLACE_CODE_OPT=--place-code %2"
    shift & shift & goto PARSE_ARGS
)
if "%~1"=="--race" (
    set "RACE_OPT=--race %2"
    shift & shift & goto PARSE_ARGS
)
rem ????`?????????????????X?L?b?v???????
shift & goto PARSE_ARGS

:RUN
echo ============================================================
echo  [1/2] fetch_odds.py
echo ============================================================
rem ????????  ????????????A?w?????I?v?V???????????W?J?????
python -X utf8 fetch_odds.py %DATE_OPT% %PLACE_OPT% %RACE_OPT% --from-predict
if errorlevel 1 (
    echo.
    echo [ABORT] fetch_odds.py failed. Skipping suggest_bets.py.
    exit /b 1
)

echo.
echo ============================================================
echo  [2/2] suggest_bets.py
echo ============================================================
python -X utf8 suggest_bets.py %PLACE_CODE_OPT% %RACE_OPT%
if errorlevel 1 (
    echo.
    echo [ERROR] suggest_bets.py failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Done: results saved to suggestions\suggested_bets.csv
echo ============================================================
endlocal