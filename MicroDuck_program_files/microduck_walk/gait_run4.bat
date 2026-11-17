@echo off
rem Measure the final run 4 (upright) policy at 4 speeds, plot each, compare with runs 1-3.
rem Output: gait_run4_0pXX.npz/.csv/.png, compare_gait_diagram.png, compare_summary.txt
call C:\Users\punn_\miniconda3\Scripts\activate.bat env_isaaclab
cd /d C:\Users\punn_\microduck_walk
set CKPT=logs\rsl_rl\microduck_flat\2026-11-06_17-34-35_upright\model_2999.pt
for %%v in (0.10 0.15 0.20 0.25) do python record_gait.py --headless --checkpoint %CKPT% --vx %%v --tag run4
for %%f in (gait_run4_*.npz) do python plot_gait.py %%f
python compare_gaits.py Run1_stand=gait_run1_final.npz Run2_hop=gait_0p20.npz Run3_crawl=gait_run3_final.npz Run4_0p10=gait_run4_0p10.npz Run4_0p15=gait_run4_0p15.npz Run4_0p20=gait_run4_0p20.npz Run4_0p25=gait_run4_0p25.npz
type compare_summary.txt
echo GAIT4 DONE
