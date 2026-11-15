# Estado de los estudios de semillas: procesos vivos, semillas y combinaciones terminadas, avance de cada carril.
# Uso: powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\status.ps1
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Seeds = Join-Path $Repo 'h1\general\work\seeds'
$Ens = Join-Path $Repo 'h1\general\work\ens_seeds'

Write-Output '== Procesos python de los estudios'
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'seed_variability|ensemble_seeds|ens\.pes_ens' } | ForEach-Object {
        $P = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
        $Args = ($_.CommandLine -replace '.*(seed_variability|ensemble_seeds) run', '' -replace '.*-m ', '').Trim()
        '{0,6}  {1,-12} afinidad 0x{2:X}  CPU {3,8:N0} s  RAM {4,6:N0} MB  {5}' -f $_.ProcessId, $P.PriorityClass,
            [int64]$P.ProcessorAffinity, $P.CPU, ($P.WorkingSet64 / 1MB), $Args
    }

Write-Output "`n== Modelos individuales: semillas terminadas (result.json)"
foreach ($M in 'pes_ql', 'pes_dql', 'pes_dqn', 'pes_rdqn', 'pes_a2c', 'pes_trf') {
    $Dir = Join-Path $Seeds $M
    $Done = if (Test-Path $Dir) {
        Get-ChildItem $Dir -Directory -Filter 'seed_*' |
            Where-Object { Test-Path (Join-Path $_.FullName 'result.json') } | ForEach-Object { $_.Name -replace 'seed_', '' }
    } else { @() }
    '{0,-9} {1,2}  {2}' -f $M, @($Done).Count, ($Done -join ' ')
}

if (Test-Path (Join-Path $Ens 'results')) {
    Write-Output "`n== Ensambles: combinaciones terminadas (entregado + 10)"
    foreach ($Dir in Get-ChildItem (Join-Path $Ens 'results') -Directory) {
        '{0,-25} {1,2}/11' -f $Dir.Name, @(Get-ChildItem $Dir.FullName -Filter '*.json').Count
    }
}

Write-Output "`n== Carriles (semilla en curso y ultimo 'Episode N', impreso cada 10 000 episodios)"
$Total = @{ 'rdqn' = 30000; 'a2c' = 125000; 'dql' = 360000; 'trf' = 30000; 'dqn' = 40000 }
foreach ($Log in @(Get-ChildItem $Seeds, $Ens -Filter 'run_stdout_lane*.log' -ErrorAction SilentlyContinue)) {
    $Lines = @(Get-Content $Log.FullName)
    $Ends = @($Lines | Select-String -Pattern 'semilla \d+: fin en|: fin; referencia').Count
    $Start = $Lines | Select-String -Pattern '(\w+) semilla (\d+): inicio' | Select-Object -Last 1
    $State = if ($Lines | Select-String -Pattern '\[carril\] terminado') { 'CARRIL TERMINADO' }
    elseif ($Start) {
        $Kind = $Start.Matches[0].Groups[1].Value; $Seed = $Start.Matches[0].Groups[2].Value
        $After = $Lines[($Start.LineNumber)..($Lines.Count - 1)] | Select-String -Pattern 'Episode (\d+) Average' |
            Select-Object -Last 1
        $Ep = if ($After) { [int]$After.Matches[0].Groups[1].Value } else { 0 }
        'en curso: {0} {1}, episodio {2}/{3}' -f $Kind, $Seed, $Ep, $(if ($Total.ContainsKey($Kind)) { $Total[$Kind] } else { '?' })
    } else { 'en curso' }
    '{0,-45} terminados {1,2}  {2}  (log {3:HH:mm})' -f $Log.FullName.Replace("$Repo\", ''), $Ends, $State, $Log.LastWriteTime
}
