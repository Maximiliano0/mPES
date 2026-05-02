<#
.SYNOPSIS
    Muestra el estado de las optimizaciones Bayesianas en curso.

.DESCRIPTION
    Busca procesos de optimizacion activos (optimize_rl, optimize_dqn,
    optimize_a2c, optimize_tr).  Para cada paquete detecta:
      - PID del proceso de optimizacion.
      - Ultimo trial completado, mejor valor y tiempo transcurrido.
      - Ultimas lineas del log de errores (si las hay).

    Tambien funciona sin argumentos: escanea todos los paquetes.

.PARAMETER Package
    (Opcional) Filtrar por un paquete especifico.
    Valores: pes_ql, pes_dql, pes_dqn, pes_a2c, pes_trf, all (default).

.EXAMPLE
    .\utils\optimization_status.ps1
    .\utils\optimization_status.ps1 pes_dqn
    .\utils\optimization_status.ps1 ac
#>
param(
    [Parameter(Position = 0)][string]$Package = 'all'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'SilentlyContinue'

# ── Rutas base ───────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# ── Resolver paquete ─────────────────────────────────────────────
$pkgMap = @{
    'pes_ql'='pes_ql';   'bayesian'='pes_ql'; 'bay'='pes_ql'; '1'='pes_ql'
    'pes_dql'='pes_dql'; 'dql'='pes_dql';     'ql'='pes_dql'; '2'='pes_dql'
    'pes_dqn'='pes_dqn'; 'dqn'='pes_dqn';                     '3'='pes_dqn'
    'pes_a2c'='pes_a2c';   'ac'='pes_a2c';  'a2c'='pes_a2c';      '4'='pes_a2c'
    'pes_trf'='pes_trf'; 'transformer'='pes_trf'; 'tr'='pes_trf'; '5'='pes_trf'
    'all'='all'
}

$FilterPkg = $pkgMap[$Package]
if (-not $FilterPkg) {
    Write-Error "Paquete desconocido: '$Package'. Opciones: pes_ql, pes_dql, pes_dqn, pes_a2c, pes_trf, all"
    exit 1
}

# ── Mapeo paquete → modulo de optimizacion ───────────────────────
$modMap = @{
    'pes_ql'  = 'optimize_rl'
    'pes_dql' = 'optimize_rl'
    'pes_dqn' = 'optimize_dqn'
    'pes_a2c'  = 'optimize_a2c'
    'pes_trf' = 'optimize_tr'
}

$allPackages = @('pes_ql', 'pes_dql', 'pes_dqn', 'pes_a2c', 'pes_trf')

if ($FilterPkg -ne 'all') {
    $allPackages = @($FilterPkg)
}

# ── Colores ──────────────────────────────────────────────────────
function Write-C { param([string]$Text, [string]$Color = 'White') Write-Host $Text -ForegroundColor $Color -NoNewline }
function Write-CL { param([string]$Text, [string]$Color = 'White') Write-Host $Text -ForegroundColor $Color }

# ── Cabecera ─────────────────────────────────────────────────────
Write-Host ""
Write-CL "  ╔══════════════════════════════════════════════════════════════╗" Cyan
Write-CL "  ║           ESTADO DE OPTIMIZACIONES BAYESIANAS               ║" Cyan
Write-CL "  ╚══════════════════════════════════════════════════════════════╝" Cyan
Write-Host ""
Write-CL "  Fecha: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')    Directorio: $ProjectDir" DarkGray
Write-Host ""

# ── Obtener todos los procesos relevantes una sola vez ───────────
# On Windows with Microsoft Store Python the venv python.exe delegates to
# python3.10.exe — we must scan both process names to find the real worker.
$allProcs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like '*optimize_*'
}

# ── Helper: determine if a process is alive vs zombie ────────────
function Test-ProcessHealth {
    <#
    .SYNOPSIS
        Returns a hashtable with health info for a given PID.
        Keys: Alive, Zombie, MB, CPU_s, Responding, Status, Color
    #>
    param([int]$ProcessId)
    $result = @{ Alive = $false; Zombie = $false; MB = 0; CPU_s = 0; Responding = $false; Status = 'NOT FOUND'; Color = 'DarkGray' }
    $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $p) { return $result }

    $result.Alive = $true
    $result.MB = [math]::Round($p.WorkingSet64 / 1MB, 1)
    try { $result.CPU_s = [math]::Round($p.CPU, 1) } catch { $result.CPU_s = 0 }
    # Responding may be $null for non-GUI / hidden-window processes
    $result.Responding = ($null -ne $p.Responding) -and $p.Responding

    # Zombie heuristic: process exists but uses < 20 MB RAM and < 1s CPU
    # after being alive for more than 60 seconds (gives TF time to load).
    $age_s = 0
    try { $age_s = (New-TimeSpan -Start $p.StartTime -End (Get-Date)).TotalSeconds } catch {}
    if ($result.MB -lt 20 -and $result.CPU_s -lt 1 -and $age_s -gt 60) {
        $result.Zombie = $true
        $result.Status = 'ZOMBIE'
        $result.Color  = 'Red'
    } else {
        $result.Status = 'ACTIVO'
        $result.Color  = 'Green'
    }
    return $result
}

$foundAny = $false

foreach ($pkg in $allPackages) {
    $optModule = $modMap[$pkg]
    $logDir    = Join-Path $ProjectDir "$pkg\inputs"

    # ── Detectar proceso de optimizacion ──────────────────────────
    # May match python.exe (venv wrapper) or python3.10.exe (real worker).
    # Prefer the heaviest (real) process; fall back to the wrapper.
    $optCandidates = $allProcs | Where-Object { $_.CommandLine -like "*$optModule*" }
    # Pick the python3.10 worker if present, otherwise the wrapper
    $optProc = $optCandidates | Where-Object { $_.Name -like 'python3*' } | Select-Object -First 1
    if (-not $optProc) {
        $optProc = $optCandidates | Select-Object -First 1
    }

    # ── Buscar log files (puede haber con o sin sufijo _resume) ──
    $logFiles = Get-ChildItem -Path $logDir -Filter 'bayesian_opt*.log' -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notmatch '_err' } |
                Sort-Object LastWriteTime -Descending
    $errFiles = Get-ChildItem -Path $logDir -Filter 'bayesian_opt*_err.log' -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending

    $logFile = if ($logFiles) { $logFiles[0].FullName } else { $null }
    $errFile = if ($errFiles) { $errFiles[0].FullName } else { $null }

    # ── Determinar si hay algo que mostrar ───────────────────────
    $hasProcess = $null -ne $optProc
    $hasLog     = $null -ne $logFile -and (Test-Path $logFile)

    if (-not $hasProcess -and -not $hasLog) { continue }
    $foundAny = $true

    # ── Health check del proceso ─────────────────────────────────
    $optHealth = $null
    if ($hasProcess) {
        $optHealth = Test-ProcessHealth -ProcessId $optProc.ProcessId
    }

    # ── Encabezado del paquete ───────────────────────────────────
    $statusIcon  = '○'
    $statusColor = 'DarkGray'
    $statusText  = 'FINALIZADO'
    if ($hasProcess -and $optHealth -and $optHealth.Zombie) {
        $statusIcon  = '!'
        $statusColor = 'Red'
        $statusText  = $optHealth.Status
    } elseif ($hasProcess) {
        $statusIcon  = '*'
        $statusColor = 'Green'
        $statusText  = 'EN CURSO'
    }

    Write-C "  $statusIcon " $statusColor
    Write-C "$pkg" Yellow
    Write-C "  [$statusText]" $statusColor
    Write-Host ""
    Write-CL "  $('-' * 60)" DarkGray

    # ── PID info ─────────────────────────────────────────────────
    if ($hasProcess) {
        $proc = Get-Process -Id $optProc.ProcessId -ErrorAction SilentlyContinue
        $cpuTime = '?'
        if ($proc) { try { $cpuTime = $proc.TotalProcessorTime.ToString('hh\:mm\:ss') } catch {} }
        $memMB = if ($optHealth) { $optHealth.MB } else { '?' }
        $hStatus = if ($optHealth) { $optHealth.Status } else { '?' }
        $hColor  = if ($optHealth) { $optHealth.Color } else { 'DarkGray' }
        Write-C "    PID optimizacion: " DarkGray
        Write-C "$($optProc.ProcessId)" White
        Write-C "  (CPU: $cpuTime, Mem: $memMB MB, Estado: " White
        Write-C $hStatus $hColor
        Write-CL ")" White

        if ($optHealth -and $optHealth.Zombie) {
            Write-CL "    *** PROCESO ZOMBIE DETECTADO -- debe relanzarse ***" Red
        }
    }

    # ── Parsear progreso del log ─────────────────────────────────
    if ($hasLog) {
        # Buscar target trials
        $targetLine = Select-String -Path $logFile -Pattern 'Target number of trials:\s*(\d+)' | Select-Object -Last 1
        $nTarget = if ($targetLine -and $targetLine.Matches[0].Groups[1].Value) { $targetLine.Matches[0].Groups[1].Value } else { '?' }

        # Buscar ultimo trial completado
        $trialLines = Select-String -Path $logFile -Pattern 'Trial\s+(\d+)/(\d+)\s+\|\s+value=([\d.]+)\s+\|\s+best=([\d.]+)\s+\|\s+elapsed=(\d+)s'
        $lastTrial = $trialLines | Select-Object -Last 1

        if ($lastTrial) {
            $m = $lastTrial.Matches[0].Groups
            $done     = $m[1].Value
            $total    = $m[2].Value
            $value    = $m[3].Value
            $best     = $m[4].Value
            $elapsed  = [int]$m[5].Value

            $pct      = [math]::Round(([int]$done / [int]$total) * 100, 1)
            $hrs      = [math]::Floor($elapsed / 3600)
            $mins     = [math]::Floor(($elapsed % 3600) / 60)
            $elStr    = if ($hrs -gt 0) { "${hrs}h ${mins}m" } else { "${mins}m" }

            # Barra de progreso
            $barLen   = 30
            $filled   = [math]::Floor($barLen * $pct / 100)
            $empty    = $barLen - $filled
            $bar      = ('█' * $filled) + ('░' * $empty)

            Write-Host ""
            Write-C "    Progreso:  " DarkGray
            Write-C "$bar" Green
            Write-CL "  $done/$total ($pct%)" White

            Write-C "    Mejor valor:      " DarkGray; Write-CL "$best" Cyan
            Write-C "    Ultimo valor:     " DarkGray; Write-CL "$value" White
            Write-C "    Tiempo:           " DarkGray; Write-CL "$elStr (${elapsed}s)" White

            # Estimar tiempo restante
            if ([int]$done -gt 0) {
                $avgPerTrial = $elapsed / [int]$done
                $remaining   = ($total - $done) * $avgPerTrial
                $remHrs      = [math]::Floor($remaining / 3600)
                $remMins     = [math]::Floor(($remaining % 3600) / 60)
                $etaStr      = if ($remHrs -gt 0) { "~${remHrs}h ${remMins}m" } else { "~${remMins}m" }
                Write-C "    ETA restante:     " DarkGray; Write-CL "$etaStr (~$([math]::Round($avgPerTrial, 0))s/trial)" DarkYellow
            }
        } else {
            # No hay trials aun, pero hay log
            $resumePattern = 'Starting fresh|Resuming'
            $resumeLine = Select-String -Path $logFile -Pattern $resumePattern | Select-Object -Last 1
            Write-Host ""
            Write-C "    Progreso:  " DarkGray
            if ($resumeLine) {
                Write-CL "$($resumeLine.Line.Trim())" White
            } else {
                Write-CL "Inicializando... (trial #1 en curso)" DarkYellow
            }
            Write-C "    Target trials:    " DarkGray; Write-CL "$nTarget" White
        }

        # ── Log path ────────────────────────────────────────────
        $relLog = $logFile.Replace($ProjectDir + '\', '')
        Write-C "    Log:              " DarkGray; Write-CL "$relLog" DarkGray
    }

    # ── Errores ──────────────────────────────────────────────────
    if ($errFile -and (Test-Path $errFile)) {
        $errContent = Get-Content $errFile -ErrorAction SilentlyContinue
        $errLines = $errContent | Where-Object { $_ -match '\S' }
        if ($errLines -and $errLines.Count -gt 0) {
            $errCount = $errLines.Count
            Write-C "    Errores/Warnings: " DarkGray
            Write-CL "$errCount linea(s) en stderr" DarkYellow
            # Mostrar las ultimas 3 lineas no vacias
            $tail = $errLines | Select-Object -Last 3
            foreach ($line in $tail) {
                $trimmed = $line.Trim()
                if ($trimmed.Length -gt 80) { $trimmed = $trimmed.Substring(0, 77) + '...' }  # truncate
                Write-CL "      $trimmed" DarkYellow
            }
        }
    }

    Write-Host ""
}

if (-not $foundAny) {
    Write-CL "  No se detectaron optimizaciones activas ni logs recientes." DarkYellow
    Write-Host ""
}

# ── Comandos utiles ──────────────────────────────────────────────
Write-CL "  ────────────────────────────────────────────────────────────" DarkGray
Write-CL "  Comandos utiles:" DarkGray
Write-CL "    Tiempo real:  Get-Content <pkg>\inputs\bayesian_opt.log -Wait -Tail 20" DarkGray
Write-CL "    Vivo?:        Get-Process -Id <PID> -ErrorAction SilentlyContinue" DarkGray
Write-CL "    Errores:      Get-Content <pkg>\inputs\bayesian_opt_err.log -Tail 20" DarkGray
Write-Host ""
