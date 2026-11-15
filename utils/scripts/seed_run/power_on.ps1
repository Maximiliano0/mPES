# Evita que la notebook se suspenda mientras corren los estudios (solo con el cargador conectado).
# Guarda antes la configuracion actual en %LOCALAPPDATA%\mpes_power_backup.json; power_restore.ps1 la repone.
# En equipos con Modern Standby, apagar la pantalla equivale a suspender: por eso tambien se desactiva.
# Uso: powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\power_on.ps1
$Backup = Join-Path $env:LOCALAPPDATA 'mpes_power_backup.json'
$Settings = @(('SUB_SLEEP', 'STANDBYIDLE'), ('SUB_SLEEP', 'HIBERNATEIDLE'), ('SUB_BUTTONS', 'LIDACTION'),
              ('SUB_VIDEO', 'VIDEOIDLE'), ('SUB_VIDEO', 'VIDEOCONLOCK'))
if (-not (Test-Path $Backup)) {
    $Saved = @{}
    foreach ($S in $Settings) {
        $Line = (powercfg /qh SCHEME_CURRENT $S[0] $S[1] | Select-String 'Current AC Power Setting Index').Line
        if ($Line) { $Saved["$($S[0]) $($S[1])"] = [Convert]::ToInt32(($Line -split ':')[-1].Trim(), 16) }
    }
    $Saved | ConvertTo-Json | Set-Content -Path $Backup -Encoding utf8
    Write-Output "Configuracion original guardada en $Backup"
} else { Write-Output "Ya habia un respaldo en $Backup (no se sobrescribe)" }
foreach ($S in $Settings) { powercfg /setacvalueindex SCHEME_CURRENT $S[0] $S[1] 0 }
powercfg /setactive SCHEME_CURRENT
Write-Output 'Con cargador: sin suspension, sin hibernacion, tapa = no hacer nada y pantalla siempre encendida.'
if ((Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue).BatteryStatus -ne 2) {
    Write-Output 'ATENCION: la notebook no esta enchufada; esta configuracion solo vale con cargador.'
}
if ((Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired') -or
    (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending')) {
    Write-Output 'ATENCION: Windows tiene un reinicio pendiente; reinicia o pausa las actualizaciones antes de lanzar.'
}
