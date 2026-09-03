<#
.SYNOPSIS
    Lanza Optuna Dashboard para cualquier paquete mPES (Windows).

.DESCRIPTION
    Equivalente PowerShell de utils/linux/optuna_dashboard.sh.
    Detecta el .db más reciente bajo <pkg>/inputs/*_BAYESIAN_OPT/ y arranca
    optuna-dashboard contra él.

.PARAMETER Project
    Alias o nombre del paquete: bayesian|pes_ql, dql|pes_dql, dqn|pes_dqn,
    ac|a2c|pes_a2c, transformer|tr|pes_trf. Si se omite, muestra menú.

.PARAMETER Port
    Puerto HTTP (default: 8080).

.EXAMPLE
    .\utils\win\optuna_dashboard.ps1
    .\utils\win\optuna_dashboard.ps1 dqn
    .\utils\win\optuna_dashboard.ps1 bayesian 9090

.NOTES
    Requisitos: entorno virtual activado y `pip install optuna-dashboard`.
#>
param(
    [string]$Project = '',
    [int]$Port = 8080
)

$ErrorActionPreference = 'Stop'

# Rutas base
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Resolve-Path (Join-Path $ScriptDir '..\..')

# Mapa paquete -> directorio de inputs
$PkgInputs = @{
    'pes_ql'  = 'tabular\pes_ql\inputs'
    'pes_dql' = 'tabular\pes_dql\inputs'
    'pes_dqn' = 'ml\pes_dqn\inputs'
    'pes_a2c' = 'ml\pes_a2c\inputs'
    'pes_trf' = 'ml\pes_trf\inputs'
}

function Resolve-Package([string]$alias) {
    switch -Regex ($alias) {
        '^(bayesian|bay|pes_ql|1)$'           { return 'pes_ql' }
        '^(dql|ql|pes_dql|2)$'                { return 'pes_dql' }
        '^(dqn|pes_dqn|3)$'                   { return 'pes_dqn' }
        '^(ac|a2c|actor-critic|pes_a2c|4)$'   { return 'pes_a2c' }
        '^(transformer|tr|pes_trf|5)$'        { return 'pes_trf' }
        default                               { return $null }
    }
}

function Find-LatestDb([string]$inputsDir) {
    if (-not (Test-Path $inputsDir)) { return $null }
    $candidates = Get-ChildItem -Path $inputsDir -Recurse -Filter 'optuna_study_*.db' `
                  -ErrorAction SilentlyContinue |
                  Where-Object { $_.DirectoryName -match '_BAYESIAN_OPT$' }
    if (-not $candidates) { return $null }
    return ($candidates | Sort-Object Name | Select-Object -Last 1).FullName
}

function Test-Dependency {
    if (-not (Get-Command optuna-dashboard -ErrorAction SilentlyContinue)) {
        Write-Host "❌ Error: optuna-dashboard no encontrado." -ForegroundColor Red
        Write-Host "   Instálalo con: pip install optuna-dashboard"
        exit 1
    }
}

function Start-Dashboard([string]$dbPath, [int]$portNum) {
    if (-not (Test-Path $dbPath)) {
        Write-Host "❌ No se encontró la base de datos: $dbPath" -ForegroundColor Red
        exit 1
    }
    $absPath   = (Resolve-Path $dbPath).Path
    $sqliteUri = "sqlite:///$($absPath -replace '\\','/')"

    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════"
    Write-Host "  Optuna Dashboard" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════"
    Write-Host "  Base de datos: $dbPath" -ForegroundColor Blue
    Write-Host "  Puerto:        $portNum" -ForegroundColor Blue
    Write-Host "  URL:           http://localhost:$portNum" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════"
    Write-Host "  Presiona Ctrl+C para detener el servidor." -ForegroundColor Yellow
    Write-Host ""

    & optuna-dashboard $sqliteUri --port $portNum
}

function Show-Menu {
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════"
    Write-Host "  Optuna Dashboard Launcher" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════"
    Write-Host ""

    $pkgs = @('pes_ql', 'pes_dql', 'pes_dqn', 'pes_a2c', 'pes_trf')
    $menuDb = @{}
    for ($i = 0; $i -lt $pkgs.Length; $i++) {
        $pkg = $pkgs[$i]
        $db  = Find-LatestDb (Join-Path $ProjectDir $PkgInputs[$pkg])
        $menuDb[$i + 1] = @{ Pkg = $pkg; Db = $db }
        if ($db) {
            Write-Host ("  {0}) {1}   → {2}" -f ($i + 1), $pkg, (Split-Path $db -Leaf)) -ForegroundColor Green
        } else {
            Write-Host ("  {0}) {1}   (sin estudios)" -f ($i + 1), $pkg) -ForegroundColor Red
        }
    }
    Write-Host "  q) Salir" -ForegroundColor Yellow
    Write-Host ""

    $choice = Read-Host "  Selección [1-5/q]"
    if ($choice -match '^[qQ]$') { exit 0 }
    if ($choice -notmatch '^[1-5]$') {
        Write-Host "❌ Opción inválida: $choice" -ForegroundColor Red
        exit 1
    }
    $entry = $menuDb[[int]$choice]
    if (-not $entry.Db) {
        Write-Host "❌ No se encontró ningún estudio en $($PkgInputs[$entry.Pkg])" -ForegroundColor Red
        exit 1
    }
    Start-Dashboard $entry.Db $Port
}

# ── Punto de entrada ─────────────────────────────────────────────────────────
Test-Dependency

if ([string]::IsNullOrWhiteSpace($Project)) {
    Show-Menu
} else {
    $pkgName = Resolve-Package $Project
    if (-not $pkgName) {
        Write-Host "❌ Proyecto desconocido: '$Project'" -ForegroundColor Red
        Write-Host "   Uso: .\optuna_dashboard.ps1 [bayesian|dql|dqn|ac|transformer] [puerto]"
        exit 1
    }
    $db = Find-LatestDb (Join-Path $ProjectDir $PkgInputs[$pkgName])
    if (-not $db) {
        Write-Host "❌ No se encontró ningún estudio en $($PkgInputs[$pkgName])" -ForegroundColor Red
        exit 1
    }
    Start-Dashboard $db $Port
}
