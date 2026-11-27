# Lanza los estudios de semillas en CARRILES: cada carril es una consola oculta que corre sus trabajos uno detras de
# otro, desacoplada de VS Code y de la terminal que lo lanza.
#
# Recursos, para que la PC siga usable (VS Code, navegador) mientras corre:
# - Prioridad "inferior a la normal": cualquier programa del usuario le gana la CPU.
# - Afinidad: todos los procesadores logicos menos el 0 y el 1 (un nucleo fisico queda libre).
#   Los procesos hijos heredan prioridad y afinidad (tambien A2C y los subprocesos de los ensambles).
# - No mas carriles que procesadores disponibles. DQN, DQN recurrente y Transformer usan un hilo por diseno.
# - Cada trabajo se reintenta 2 veces; los arneses son reanudables (saltean lo que ya tiene resultado).
#
# Uso (desde cualquier carpeta; -Lane: un carril por cadena, trabajos separados por ';'):
#   powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\launch_lanes.ps1 -Module seed_variability `
#       -Lane '--models rdqn --seeds 1001;--models rdqn --seeds 1002', '--models a2c --seeds 1003 1004'
#   powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\launch_lanes.ps1 -Module ensemble_seeds `
#       -Lane '--ensembles pes_ens', '--ensembles pes_ens_sprb'
# Evaluar modelos ya entrenados en otros escenarios (sin reentrenar): -Command eval (solo seed_variability).
#   powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\launch_lanes.ps1 -Module seed_variability `
#       -Command eval -Lane '--models trf --scenarios all', '--models ql dql --scenarios all'
#   Ensambles en otros escenarios: -Module ensemble_seeds -Lane '--ensembles pes_ens --scenarios gen'
# Sumar carriles con otros ya corriendo: -First N (numera desde N; el .cmd de un carril vivo no se puede reescribir).
# Antes: power_on.ps1 (evita la suspension). Estado: status.ps1. Detener: stop.ps1. Despues: power_restore.ps1.

param(
    [ValidateSet('seed_variability', 'ensemble_seeds')][string]$Module = 'seed_variability',
    [ValidateSet('run', 'eval')][string]$Command = 'run',
    [Parameter(Mandatory = $true)][string[]]$Lane,
    [ValidateRange(1, 64)][int]$First = 1   # numero del primer carril: no reusar el de un carril que sigue corriendo
)
if ($Command -eq 'eval' -and $Module -ne 'seed_variability') { throw "-Command eval solo existe en seed_variability" }

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Logs = Join-Path $Repo ('h1\general\work\' + $(if ($Module -eq 'seed_variability') { 'seeds' } else { 'ens_seeds' }))
$CmdDir = Join-Path $env:TEMP 'mpes_lanes'
$BelowNormal = 16384
$Cpus = [Environment]::ProcessorCount
$Affinity = '{0:X}' -f ([int64][math]::Pow(2, $Cpus) - 1 - 3)   # todos menos los logicos 0 y 1

if (-not (Test-Path (Join-Path $Repo 'win_mpes_env\Scripts\activate.bat'))) { throw "No existe win_mpes_env en $Repo" }
if ($Lane.Count -gt ($Cpus - 2)) { throw "Demasiados carriles ($($Lane.Count)) para $Cpus procesadores logicos" }
New-Item -ItemType Directory -Force -Path $CmdDir, $Logs | Out-Null

$Index = $First - 1
foreach ($Jobs in $Lane) {
    $Index++
    $Tag = "$Module`_lane$Index"
    $Out = Join-Path $Logs "run_stdout_lane$Index.log"
    $Cmd = Join-Path $CmdDir "$Tag.cmd"
    $Calls = ($Jobs -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ } |
        ForEach-Object { "call :job $_" }) -join "`r`n"
    @"
@echo off
cd /d "$Repo"
call win_mpes_env\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
set CUDA_VISIBLE_DEVICES=-1
cd h1
$Calls
echo [carril] terminado >> "$Out"
exit /b

:job
for /L %%i in (1,1,2) do (
    echo [carril] %* - intento %%i >> "$Out"
    start "" /B /WAIT /BELOWNORMAL /AFFINITY $Affinity python -m general.scripts.$Module $Command %* >> "$Out" 2>&1
    if not errorlevel 1 goto :eof
)
goto :eof
"@ | Set-Content -Path $Cmd -Encoding ascii

    $Startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
        ShowWindow = [uint16]0; PriorityClass = [uint32]$BelowNormal }
    $R = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "cmd.exe /c `"$Cmd`""; CurrentDirectory = $Repo; ProcessStartupInformation = $Startup }
    if ($R.ReturnValue -eq 0) { Write-Output "carril $Index lanzado (cmd PID $($R.ProcessId)): $Jobs" }
    else { Write-Output "carril $Index ERROR al lanzar ($($R.ReturnValue))" }
}
Write-Output "Afinidad 0x$Affinity, prioridad inferior a la normal. Registros: $Logs\run_stdout_lane*.log"
