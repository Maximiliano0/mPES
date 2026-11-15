# Repone la configuracion de energia guardada por power_on.ps1 y borra el respaldo.
# Uso: powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\power_restore.ps1
$Backup = Join-Path $env:LOCALAPPDATA 'mpes_power_backup.json'
if (-not (Test-Path $Backup)) { Write-Output "No hay respaldo en ${Backup}: nada que restaurar."; exit 0 }
$Saved = Get-Content $Backup -Raw | ConvertFrom-Json
foreach ($P in $Saved.PSObject.Properties) {
    $Group, $Name = $P.Name -split ' '
    powercfg /setacvalueindex SCHEME_CURRENT $Group $Name $P.Value
    Write-Output "$($P.Name) -> $($P.Value)"
}
powercfg /setactive SCHEME_CURRENT
Remove-Item $Backup -Force
Write-Output 'Configuracion de energia restaurada.'
