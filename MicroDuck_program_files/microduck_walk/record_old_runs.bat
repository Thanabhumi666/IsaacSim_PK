@echo off
rem Record 12 s videos of run 1 (stands still) and run 2 (hops on one leg) for the report.
rem MICRODUCK_LEGACY=1 plays them with the terminations they were trained with.
call C:\Users\punn_\miniconda3\Scripts\activate.bat env_isaaclab
cd /d C:\Users\punn_\microduck_walk
set MICRODUCK_LEGACY=1
python run.py play --task MicroDuck-Flat-Play-v0 --num_envs 4 --headless --video --video_length 600 --load_run ".*16-48-38"
python run.py play --task MicroDuck-Flat-Play-v0 --num_envs 4 --headless --video --video_length 600 --load_run ".*stdfix"
echo OLDRUNS DONE
