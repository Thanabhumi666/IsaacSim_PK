# Redo the gait comparison and all report figures, zip report_figures\{training,kinematics,gait,info}
# and put the zip on the clipboard as a file (AnyDesk then lets the other PC paste / read it).
#   Run from the env_isaaclab prompt:  .\redo_figs.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
python compare_gaits.py Run1=gait_run1_final.npz Run2=gait_0p20.npz Run3=gait_run3_final.npz Run4_0p10=gait_run4_0p10.npz Run4_0p15=gait_run4_0p15.npz Run4_0p20=gait_run4_0p20.npz Run4_0p25=gait_run4_0p25.npz *> compare_log.txt
"compare_gaits exit $LASTEXITCODE"
python report_data.py *> report_data_log.txt
Select-String -Path report_data_log.txt -Pattern '^--- |REPORTDATA|Error' | ForEach-Object { $_.Line }
$zip = Join-Path $here "rf.zip"
Push-Location report_figures
Compress-Archive training, kinematics, gait, info $zip -Force
Pop-Location
Set-Clipboard -Path $zip
"zip $((Get-Item $zip).Length) bytes, md5 $((Get-FileHash $zip -Algorithm MD5).Hash) on clipboard"
