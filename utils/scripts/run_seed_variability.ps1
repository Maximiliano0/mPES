# Lanza h1/general/scripts/seed_variability.py desacoplado de la terminal que lo invoca.
#
# Uso (desde la raíz del repositorio):
#   powershell -ExecutionPolicy Bypass -File utils\scripts\run_seed_variability.ps1
#
# El proceso se crea con Win32_Process (WMI), así que no depende de VS Code ni de la
# sesión que lo lanzó. Mientras corre le pide a Windows que no suspenda el equipo
# (SetThreadExecutionState). No evita la suspensión por cerrar la tapa ni un reinicio.
# Es reanudable: al relanzarlo, omite las semillas que ya tienen result.json.
# Registro: h1\general\work\seeds\seed_variability.log (y run_stdout.log).

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Inner = Join-Path $Root 'utils\scripts\_seed_variability_inner.cmd'
$Out = Join-Path $Root 'h1\general\work\seeds\run_stdout.log'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

@"
@echo off
cd /d "$Root"
call win_mpes_env\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
set CUDA_VISIBLE_DEVICES=-1
cd h1
python -m general.scripts.seed_variability run >> "$Out" 2>&1
"@ | Set-Content -Path $Inner -Encoding ascii

$Result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd.exe /c `"$Inner`""
    CurrentDirectory = $Root
}
if ($Result.ReturnValue -eq 0) {
    Write-Output "Lanzado. PID de cmd.exe: $($Result.ProcessId). Registro: $Out"
} else {
    Write-Output "Error al lanzar (ReturnValue = $($Result.ReturnValue))."
}
