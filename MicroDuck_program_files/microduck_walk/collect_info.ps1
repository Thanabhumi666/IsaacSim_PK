# Collect facts about the machine, software and training setup for the report
# into report_figures\info\report_info.txt.   Run from the env_isaaclab prompt:  .\collect_info.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$dst = Join-Path $here "report_figures\info"
New-Item -ItemType Directory -Force $dst | Out-Null
$out = Join-Path $dst "report_info.txt"
$lab = "C:\Users\punn_\IsaacLab"
$L = New-Object System.Collections.Generic.List[string]

$L.Add("=== SYSTEM ===")
try {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $L.Add(("CPU: {0}, {1} cores / {2} threads" -f $cpu.Name.Trim(), $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors))
    $cs = Get-CimInstance Win32_ComputerSystem
    $L.Add(("RAM: {0:N0} GB" -f ($cs.TotalPhysicalMemory / 1GB)))
    $os = Get-CimInstance Win32_OperatingSystem
    $L.Add(("OS: {0} {1} (build {2})" -f $os.Caption, $os.Version, $os.BuildNumber))
} catch { $L.Add("system query failed: $_") }
try { $L.Add("GPU: " + ((nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) -join "; ")) } catch { $L.Add("GPU: nvidia-smi failed") }
$L.Add("")
$L.Add("=== SOFTWARE ===")
try { $L.Add("python: " + (python --version)) } catch { $L.Add("python: ?") }
try {
    python -m pip list 2>$null | Select-String -Pattern "^(isaacsim|isaaclab|isaaclab-rl|isaaclab-tasks|rsl-rl-lib|torch|gymnasium|numpy|trimesh|tensorboard|imageio|imageio-ffmpeg|opencv-python|matplotlib)\s" |
        ForEach-Object { $L.Add($_.Line) }
} catch { $L.Add("pip list failed") }
try { $L.Add("IsaacLab git: " + (git -C $lab describe --tags)) } catch { $L.Add("IsaacLab git: ?") }
$L.Add("")
foreach ($f in @("source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\g1\agents\rsl_rl_ppo_cfg.py",
                 "source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\velocity_env_cfg.py")) {
    $L.Add("=== FILE $f ===")
    $p = Join-Path $lab $f
    if (Test-Path $p) { Get-Content $p | ForEach-Object { $L.Add($_) } } else { $L.Add("(missing)") }
    $L.Add("")
}
foreach ($f in @("train_log.txt", "train2_log.txt", "train3_log.txt", "train4_log.txt")) {
    $L.Add("=== TAIL $f ===")
    $p = Join-Path $here $f
    if (Test-Path $p) { Get-Content $p -Tail 40 | ForEach-Object { $L.Add($_) } } else { $L.Add("(missing)") }
    $L.Add("")
}
$L | Out-File -Encoding utf8 $out
"wrote $out ($((Get-Item $out).Length) bytes)"
