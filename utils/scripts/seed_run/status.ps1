# Estado de los estudios de semillas: avance, pendientes, tiempo restante y carriles.
# Uso (desde cualquier carpeta):
#   powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\status.ps1            # resumen
#   powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\status.ps1 -Detalle   # + carriles viejos y estudios previos
param([switch]$Detalle)

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$General = Join-Path $Repo 'h1\general'
$Seeds = Join-Path $General 'work\seeds'
$Ens = Join-Path $General 'work\ens_seeds'
$Models = 'pes_ql', 'pes_dql', 'pes_dqn', 'pes_rdqn', 'pes_a2c', 'pes_trf'
$Ensembles = 'pes_ens', 'pes_ens_sprb', 'pes_ens_accq', 'pes_ens_trf_guard', 'pes_ens_consensus', 'pes_ens_consensus_prior'
$Runs = @('entregado') + (1001..1010 | ForEach-Object { "seed_$_" })
$PlanFile = Join-Path $General 'results\ensemble\ensemble_seeds_plan.json'
$Combos = @('entregado') + @(if (Test-Path $PlanFile) { (Get-Content $PlanFile -Raw | ConvertFrom-Json).combinaciones.id })
$Reference = @('sev_base') + (1..5 | ForEach-Object { "heldout_s$_" })
$AllScenarios = @(Get-ChildItem (Join-Path $General 'work\pes_ens\scenarios') -Directory | ForEach-Object Name)
$GenScenarios = @($AllScenarios | Where-Object { $Reference -notcontains $_ })
# Minutos por escenario medidos en el benchmark (se usan hasta que haya tiempos propios).
$Fallback = @{ 'pes_ql' = 0.2; 'pes_dql' = 0.2; 'pes_dqn' = 0.35; 'pes_rdqn' = 0.75; 'pes_a2c' = 0.35; 'pes_trf' = 1.1;
    'pes_ens' = 0.55; 'pes_ens_sprb' = 0.8; 'pes_ens_accq' = 1.25; 'pes_ens_trf_guard' = 0.6;
    'pes_ens_consensus' = 0.7; 'pes_ens_consensus_prior' = 0.7 }

function Read-Run([string]$Path, [string[]]$Wanted) {
    # Escenarios hechos (de $Wanted) y segundos medidos en un generalizacion.json / <combinacion>.json.
    $Result = [pscustomobject]@{ Done = 0; Seconds = 0.0; Timed = 0 }
    if (-not (Test-Path $Path)) { return $Result }
    try { $Json = Get-Content $Path -Raw | ConvertFrom-Json } catch { return $Result }
    foreach ($Name in @($Json.por_secuencia.PSObject.Properties.Name)) {
        if ($Wanted -contains $Name) {
            $Result.Done++
            $Seconds = $Json.segundos.$Name
            if ($Seconds) { $Result.Seconds += [double]$Seconds; $Result.Timed++ }
        }
    }
    return $Result
}

function Bar([int]$Done, [int]$Total) {
    $Pct = if ($Total) { [math]::Floor(100 * $Done / $Total) } else { 0 }
    $Fill = [math]::Round(20 * $Pct / 100)
    '[{0}{1}] {2,3}%' -f ('#' * $Fill), ('.' * (20 - $Fill)), $Pct
}

function Label([string]$Run) { if ($Run -eq 'entregado') { 'E' } else { $Run -replace '^seed_', '' -replace '_.*', '' } }

# ---------------------------------------------------------------- encabezado
$Now = Get-Date
Write-Output ("ESTADO DEL ESTUDIO DE SEMILLAS - {0:yyyy-MM-dd HH:mm}" -f $Now)
$Battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
if ($Battery) {
    $OnAc = $Battery.BatteryStatus -eq 2
    $Power = if ($OnAc) { 'con cargador' } else { 'A BATERIA: enchufar (sin cargador la PC puede suspenderse)' }
    Write-Output ("Alimentacion: {0} ({1}%)" -f $Power, $Battery.EstimatedChargeRemaining)
}
$Procs = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'seed_variability|ensemble_seeds|ens\.pes_ens' })
$Ram = ($Procs | ForEach-Object { (Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue).WorkingSet64 } |
    Measure-Object -Sum).Sum / 1GB
$Free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
Write-Output ("Procesos python del estudio: {0} ({1:N1} GB); RAM libre {2:N1} GB" -f $Procs.Count, $Ram, $Free)

# ---------------------------------------------------------------- avance
$RemainingMin = 0.0
$Pending = @()

Write-Output "`n== GENERALIZACION - MODELOS INDIVIDUALES (11 corridas x 27 escenarios)"
foreach ($M in $Models) {
    $Done = 0; $Complete = 0; $Sec = 0.0; $Timed = 0; $Missing = @()
    foreach ($Run in $Runs) {
        $R = Read-Run (Join-Path $Seeds "$M\$Run\generalizacion.json") $AllScenarios
        $Done += $R.Done; $Sec += $R.Seconds; $Timed += $R.Timed
        if ($R.Done -ge $AllScenarios.Count) { $Complete++ }
        elseif ($R.Done -gt 0) { $Missing += '{0}({1}/{2})' -f (Label $Run), $R.Done, $AllScenarios.Count }
        else { $Missing += Label $Run }
    }
    $Total = $Runs.Count * $AllScenarios.Count
    $PerScenario = if ($Timed) { $Sec / 60 / $Timed } else { $Fallback[$M] }
    $RemainingMin += ($Total - $Done) * $PerScenario
    $State = if ($Complete -eq $Runs.Count) { 'LISTO' } else { 'faltan: ' + ($Missing -join ' ') }
    Write-Output ('  {0,-9} {1}  {2,2}/{3} corridas  {4}' -f $M, (Bar $Done $Total), $Complete, $Runs.Count, $State)
    if ($Complete -lt $Runs.Count) { $Pending += $M }
}

Write-Output ("`n== GENERALIZACION - ENSAMBLES ({0} combinaciones x {1} escenarios)" -f $Combos.Count, $GenScenarios.Count)
foreach ($E in $Ensembles) {
    $Done = 0; $Complete = 0; $Sec = 0.0; $Timed = 0; $Missing = @()
    foreach ($C in $Combos) {
        $R = Read-Run (Join-Path $Ens "generalizacion\$E\$C.json") $GenScenarios
        $Done += $R.Done; $Sec += $R.Seconds; $Timed += $R.Timed
        if ($R.Done -ge $GenScenarios.Count) { $Complete++ }
        elseif ($R.Done -gt 0) { $Missing += '{0}({1}/{2})' -f (Label $C), $R.Done, $GenScenarios.Count }
        else { $Missing += Label $C }
    }
    $Total = $Combos.Count * $GenScenarios.Count
    $PerScenario = if ($Timed) { $Sec / 60 / $Timed } else { $Fallback[$E] }
    $RemainingMin += ($Total - $Done) * $PerScenario
    $State = if ($Complete -eq $Combos.Count) { 'LISTO' } else { 'faltan: ' + ($Missing -join ' ') }
    Write-Output ('  {0,-24} {1}  {2,2}/{3}  {4}' -f $E, (Bar $Done $Total), $Complete, $Combos.Count, $State)
    if ($Complete -lt $Combos.Count) { $Pending += $E }
}
Write-Output '  (E = entregado; c01..c10 = combinaciones; n(a/b) = a medias, b escenarios)'

# ---------------------------------------------------------------- carriles
Write-Output "`n== CARRILES"
$Active = 0
$Logs = @(Get-ChildItem $Seeds, $Ens -Filter 'run_stdout_lane*.log' -ErrorAction SilentlyContinue | Sort-Object FullName)
foreach ($Log in $Logs) {
    # Marcas de carril en todo el archivo (los paquetes imprimen mucho y el inicio puede quedar lejos del final).
    $Marks = @(Select-String -Path $Log.FullName -Pattern '^\[carril\] ' -ErrorAction SilentlyContinue)
    $LastEnd = 0; $LastJob = $null; $Start = 0
    foreach ($Mark in $Marks) {
        if ($Mark.Line -match '^\[carril\] terminado') { $LastEnd = $Mark.LineNumber }
        elseif ($Mark.Line -match ' - intento') {
            $LastJob = $Mark
            if ($Start -le $LastEnd) { $Start = $Mark.LineNumber }
        }
    }
    $Running = $LastJob -and $LastJob.LineNumber -gt $LastEnd
    $Age = ($Now - $Log.LastWriteTime).TotalHours
    if (-not $Running -and $Age -gt 12 -and -not $Detalle) { continue }
    $Name = '{0}/{1}' -f (Split-Path (Split-Path $Log.FullName) -Leaf), ($Log.BaseName -replace 'run_stdout_', '')
    $Retries = @($Marks | Where-Object { $_.LineNumber -ge $Start -and $_.Line -match ' - intento [2-9]' }).Count
    $Errors = @(Select-String -Path $Log.FullName -Pattern '^Traceback' -ErrorAction SilentlyContinue |
        Where-Object { $_.LineNumber -ge $Start }).Count
    $Last = Get-Content $Log.FullName -Tail 3000 -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] ' } | Select-Object -Last 1
    if ($Running) {
        $Active++
        $Job = ($LastJob.Line -replace '^\[carril\] ', '' -replace ' - intento.*', '').Trim()
        Write-Output ('  {0,-17} EN CURSO  trabajo: {1}' -f $Name, $Job)
    } else {
        Write-Output ('  {0,-17} TERMINADO ({1:dd/MM HH:mm})' -f $Name, $Log.LastWriteTime)
    }
    if ($Last) { Write-Output ('  {0,-17} ultimo:   {1}' -f '', $Last.Trim()) }
    if ($Errors -or $Retries) { Write-Output ('  {0,-17} ATENCION: {1} errores (Traceback), {2} reintentos' -f '', $Errors, $Retries) }
}
if (-not $Logs) { Write-Output '  (sin registros de carriles)' }

# ---------------------------------------------------------------- resumen
Write-Output "`n== RESUMEN"
if ($Pending.Count -eq 0) {
    Write-Output '  Todo evaluado. Siguiente paso (desde h1\): python -m general.scripts.seed_generalisation'
} else {
    Write-Output ('  Pendientes: {0}' -f ($Pending -join ', '))
    $Hours = $RemainingMin / 60
    $Wall = if ($Active) { ' ~{0:N1} h con {1} carriles activos' -f ($Hours / $Active), $Active } else { ' (no hay carriles activos)' }
    Write-Output ('  Trabajo restante estimado: {0:N1} h de CPU;{1}' -f $Hours, $Wall)
    if (-not $Active) { Write-Output '  ATENCION: hay pendientes pero ningun carril corriendo (ver launch_lanes.ps1).' }
}

# ---------------------------------------------------------------- estudios previos
$Trained = 0
foreach ($M in $Models) { $Trained += @(Get-ChildItem (Join-Path $Seeds $M) -Directory -Filter 'seed_*' -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'result.json') }).Count }
$RefCombos = @(Get-ChildItem (Join-Path $Ens 'results') -Recurse -Filter '*.json' -ErrorAction SilentlyContinue).Count
Write-Output ('  Estudios previos (referencia y replicas): semillas entrenadas {0}/60, ensambles {1}/66' -f $Trained, $RefCombos)
if ($Detalle) {
    foreach ($M in $Models) {
        $Done = @(Get-ChildItem (Join-Path $Seeds $M) -Directory -Filter 'seed_*' -ErrorAction SilentlyContinue |
            Where-Object { Test-Path (Join-Path $_.FullName 'result.json') } | ForEach-Object { $_.Name -replace 'seed_', '' })
        Write-Output ('    {0,-9} {1,2}/10  {2}' -f $M, $Done.Count, ($Done -join ' '))
    }
}
