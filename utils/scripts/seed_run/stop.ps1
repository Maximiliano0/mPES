# Detiene los estudios de semillas: primero las consolas de los carriles (para que no reintenten) y despues los
# python.exe. Lo que ya tiene resultado queda guardado; al relanzar se retoma desde ahi.
# Uso: powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\stop.ps1
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Lanes = Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" | Where-Object { $_.CommandLine -match 'mpes_lanes' }
foreach ($P in $Lanes) { Stop-Process -Id $P.ProcessId -Force -ErrorAction SilentlyContinue }
$Pattern = 'seed_variability|ensemble_seeds|ens\.pes_ens'
$Py = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match $Pattern }
foreach ($P in $Py) { Stop-Process -Id $P.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
$Left = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "$Pattern|mpes_lanes" }).Count
Write-Output "Detenidos: $(@($Lanes).Count) carriles y $(@($Py).Count) procesos python. Quedan: $Left"
# Un ensamble cortado a mitad de escenario deja sus CSV cambiados: se restauran desde git.
$Stash = Get-ChildItem (Join-Path $Repo 'h1\ens') -Recurse -Filter '*.bench_stash' -ErrorAction SilentlyContinue
if ($Stash) {
    foreach ($S in $Stash) { Remove-Item $S.FullName -Force }
    git -C $Repo checkout -- 'h1/ens/*/inputs/initial_severity.csv' 'h1/ens/*/inputs/sequence_lengths.csv'
    Write-Output "CSV de ensambles restaurados desde git ($(@($Stash).Count) respaldos .bench_stash eliminados)."
}
