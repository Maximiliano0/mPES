<#
.SYNOPSIS
    Lanzar optimizacion Bayesiana en Windows.

.DESCRIPTION
    Equivalente Windows de run_bayesian_opt.sh.
    Lanza la optimizacion en segundo plano y configura la PC para no
    suspenderse.

    Los procesos se ejecutan con Start-Process -WindowStyle Hidden,
    por lo que NO es necesario mantener ninguna terminal abierta.

.PARAMETER Package
    Alias o nombre del paquete destino.
    Valores: bayesian|bay|1, dql|ql|2, dqn|3, ac|a2c|4, transformer|tr|5

.PARAMETER NTrials
    Numero de trials de optimizacion (por defecto 30).

.PARAMETER ResumeDate
    (Opcional) Fecha YYYY-MM-DD para reanudar una corrida previa.

.EXAMPLE
    .\utils\run_bayesian_opt.ps1 dqn 110
    .\utils\run_bayesian_opt.ps1 ac 100
    .\utils\run_bayesian_opt.ps1 bayesian 100 2026-02-12
#>
param(
    [Parameter(Mandatory, Position = 0)][string]$Package,
    [Parameter(Position = 1)][int]$NTrials = 30,
    [Parameter(Position = 2)][string]$ResumeDate = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Rutas base ───────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$Python     = Join-Path $ProjectDir 'win_mpes_env\Scripts\python.exe'


if (-not (Test-Path $Python)) {
    Write-Error "Python no encontrado: $Python"
    exit 1
}

# ── Resolver paquete ────────────────────────────────────────────
$pkgMap = @{
    'pes_ql'='pes_ql';   'bayesian'='pes_ql'; 'bay'='pes_ql'; '1'='pes_ql'
    'pes_dql'='pes_dql'; 'dql'='pes_dql';     'ql'='pes_dql'; '2'='pes_dql'
    'pes_dqn'='pes_dqn'; 'dqn'='pes_dqn';                     '3'='pes_dqn'
    'pes_rdqn'='pes_rdqn'; 'rdqn'='pes_rdqn';                 '7'='pes_rdqn'
    'pes_a2c'='pes_a2c';   'ac'='pes_a2c';  'a2c'='pes_a2c'; 'actor-critic'='pes_a2c'; '4'='pes_a2c'
    'pes_trf'='pes_trf'; 'transformer'='pes_trf'; 'tr'='pes_trf';   '5'='pes_trf'
}

$PkgName = $pkgMap[$Package]
if (-not $PkgName) {
    Write-Error "Paquete desconocido: '$Package'. Opciones: bayesian, dql, dqn, rdqn, ac, transformer"
    exit 1
}

# ── Resolver modulo de optimizacion ─────────────────────────────
$modMap = @{
    'pes_ql'  = 'pes_ql.ext.optimize_rl'
    'pes_dql' = 'pes_dql.ext.optimize_rl'
    'pes_dqn' = 'pes_dqn.ext.optimize_dqn'
    'pes_rdqn' = 'pes_rdqn.ext.optimize_rdqn'
    'pes_a2c'  = 'pes_a2c.ext.optimize_a2c'
    'pes_trf' = 'pes_trf.ext.optimize_tr'
}
$OptModule = $modMap[$PkgName]

# ── Preparar directorio de logs ─────────────────────────────────
$LogDir = Join-Path $ProjectDir "$PkgName\inputs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$LogSuffix = ''
if ($ResumeDate) { $LogSuffix = "_resume_$ResumeDate" }
$LogFile    = Join-Path $LogDir "bayesian_opt${LogSuffix}.log"
$ErrFile    = Join-Path $LogDir "bayesian_opt${LogSuffix}_err.log"

# ── Construir argumentos ────────────────────────────────────────
# Use utils/run_module.py wrapper so that stdout/stderr are redirected at the
# Python level (line-buffered) instead of via Start-Process -Redirect*, which
# holds an exclusive file lock that prevents monitoring the log in real-time.
$RunModule = Join-Path $ScriptDir 'run_module.py'
$pyArgs = @($RunModule, $OptModule, $LogFile, $ErrFile, "$NTrials")
if ($ResumeDate) { $pyArgs += @('--resume', $ResumeDate) }

# ── Evitar suspension / hibernacion / apagado de pantalla ───────
Write-Host "`n  Configurando energia: sin suspension ni hibernacion..."
powercfg /change standby-timeout-ac 0   2>$null
powercfg /change standby-timeout-dc 0   2>$null
powercfg /change hibernate-timeout-ac 0 2>$null
powercfg /change hibernate-timeout-dc 0 2>$null
powercfg /change monitor-timeout-ac 0   2>$null
powercfg /change monitor-timeout-dc 0   2>$null

# ── Configurar entorno para el proceso hijo ─────────────────────
$env:VIRTUAL_ENV       = Join-Path $ProjectDir 'win_mpes_env'
$env:PYTHONIOENCODING  = 'utf-8'
$env:TF_ENABLE_ONEDNN_OPTS = '0'

# ── Lanzar optimizacion en segundo plano ────────────────────────
# No -RedirectStandard* here — the run_module.py wrapper handles
# file-level stdout/stderr redirection (line-buffered, real-time readable).
$optProc = Start-Process -FilePath $Python `
    -ArgumentList $pyArgs `
    -WorkingDirectory $ProjectDir `
    -PassThru -WindowStyle Hidden

$OptPid = $optProc.Id

# Save PID to file for reliable monitoring
$PidFile = Join-Path $LogDir "opt${LogSuffix}.pid"
"$OptPid" | Set-Content -Path $PidFile -Encoding ASCII

# Verify the process actually started (wait a few seconds)
Start-Sleep -Seconds 3
$check = Get-Process -Id $OptPid -ErrorAction SilentlyContinue
if (-not $check -or $check.HasExited) {
    Write-Host "`n  ERROR: El proceso de optimizacion (PID=$OptPid) murio inmediatamente." -ForegroundColor Red
    Write-Host "    Revise el log de errores: $ErrFile" -ForegroundColor Red
    if (Test-Path $ErrFile) { Get-Content $ErrFile -Tail 10 }
    exit 1
}

Write-Host "  Optimizacion lanzada   PID=$OptPid  trials=$NTrials  (verificado: vivo)"
Write-Host "  Log: $PkgName\inputs\bayesian_opt${LogSuffix}.log"

# ── Resumen ─────────────────────────────────────────────────────
Write-Host "`n  =========================================="
Write-Host "  Optimizacion lanzada"
Write-Host "    Paquete:     $PkgName"
Write-Host "    Modulo:      $OptModule"
Write-Host "    Trials:      $NTrials"
Write-Host "    PID:         $OptPid"
Write-Host "    Log:         $PkgName\inputs\bayesian_opt${LogSuffix}.log"
Write-Host "  =========================================="

Write-Host "`n  Comandos utiles:"
Write-Host "    Progreso:    Select-String 'Trial' $LogFile | Select-Object -Last 10"
Write-Host "    Tiempo real: Get-Content $LogFile -Wait -Tail 20"
Write-Host "    Vivo?:       Get-Process -Id $OptPid -ErrorAction SilentlyContinue"
Write-Host "    Errores:     Get-Content $ErrFile -Tail 20"
Write-Host ""

Write-Host "  Los procesos corren en SEGUNDO PLANO (ShellExecute)."
Write-Host "  Sobreviven al cierre de VS Code, terminal y PowerShell."
Write-Host "  Puede cerrar todo con seguridad."
Write-Host ""
