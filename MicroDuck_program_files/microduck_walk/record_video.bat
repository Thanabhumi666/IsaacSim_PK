@echo off
rem Double-click to record a 12 s walking video (.mp4) of the newest checkpoint.
rem   record_video.bat            -> run 4 "upright" (default)
rem   record_video.bat bipedfix   -> run 3
rem Video is saved in logs\rsl_rl\microduck_flat\<run>\videos\play\
set RUN=%1
if "%RUN%"=="" set RUN=upright
call C:\Users\punn_\miniconda3\Scripts\activate.bat env_isaaclab
cd /d C:\Users\punn_\microduck_walk
python run.py play --task MicroDuck-Flat-Play-v0 --num_envs 4 --headless --video --video_length 600 --load_run ".*%RUN%"
explorer logs\rsl_rl\microduck_flat
pause
