@echo off
rem Double-click to watch the Micro Duck walk (newest checkpoint of a run).
rem   play_walk.bat            -> run 4 "upright" (default)
rem   play_walk.bat bipedfix   -> run 3, stdfix -> run 2
set RUN=%1
if "%RUN%"=="" set RUN=upright
call C:\Users\punn_\miniconda3\Scripts\activate.bat env_isaaclab
cd /d C:\Users\punn_\microduck_walk
python run.py play --task MicroDuck-Flat-Play-v0 --num_envs 1 --load_run ".*%RUN%"
pause
