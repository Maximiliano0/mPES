<#
.SYNOPSIS
    Lanzador unificado de auditoría de la tesis LaTeX.

.DESCRIPTION
    Ejecuta una de las dos modalidades de auditoría:

      code    Auditoría programática (script Python con 6 criterios y
              compilación pdflatex). Genera writings/audit/AUDIT.md.

      prompt  Auditoría asistida por LLM: empaqueta los criterios + el
              texto completo de la tesis en un único Markdown listo
              para pegar en ChatGPT, Claude o Copilot. Genera
              writings/audit/PROMPT.md.

.PARAMETER Mode
    'code' (por defecto) o 'prompt'.

.PARAMETER NoTex
    En modo 'code', omite la compilación pdflatex.

.PARAMETER Clean
    Elimina artefactos previos en writings/out/ (aux, log, bbl, blg,
    toc, out, pdf) antes de ejecutar. Útil para forzar una compilación
    completamente limpia.

.PARAMETER Open
    Tras ejecutar, abre automáticamente el artefacto generado:
      - modo 'code'   -> writings/audit/AUDIT.md y el PDF (si existe).
      - modo 'prompt' -> writings/audit/PROMPT.md.

.EXAMPLE
    ./audit.ps1                       # default: code
    ./audit.ps1 code                  # auditoría programática (con pdflatex)
    ./audit.ps1 code -NoTex           # auditoría programática sin pdflatex
    ./audit.ps1 code -Clean -Open     # limpia, audita y abre AUDIT.md + PDF
    ./audit.ps1 prompt -Open          # genera el prompt y lo abre
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('code', 'prompt')]
    [string]$Mode = 'code',

    [switch]$NoTex,
    [switch]$Clean,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Clean) {
        $outDir = Join-Path $PSScriptRoot 'out'
        if (Test-Path -LiteralPath $outDir) {
            Write-Host "[clean] Eliminando artefactos en $outDir ..." -ForegroundColor Yellow
            $patterns = '*.aux','*.log','*.bbl','*.blg','*.toc','*.out','*.pdf','*.synctex.gz'
            foreach ($pat in $patterns) {
                Get-ChildItem -Path (Join-Path $outDir $pat) -File -Recurse -ErrorAction SilentlyContinue |
                    Remove-Item -Force -ErrorAction SilentlyContinue
            }
        }
    }

    switch ($Mode) {
        'code'   {
            $pyArgs = @('audit/audit.py')
            if ($NoTex) { $pyArgs += '--no-tex' }
            python @pyArgs
        }
        'prompt' {
            python 'audit/audit_prompt.py'
        }
    }

    if ($Open) {
        $codeCmd = Get-Command code -ErrorAction SilentlyContinue
        function Open-File([string]$Path) {
            if (-not (Test-Path -LiteralPath $Path)) {
                Write-Host "[open] No existe: $Path" -ForegroundColor Red
                return
            }
            Write-Host "[open] $Path" -ForegroundColor Green
            $ext = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
            if ($ext -eq '.md' -and $codeCmd) {
                & $codeCmd.Source --reuse-window $Path | Out-Null
            }
            elseif ($ext -eq '.pdf') {
                Start-Process -FilePath $Path | Out-Null
            }
            else {
                Invoke-Item -LiteralPath $Path
            }
        }

        switch ($Mode) {
            'code' {
                Open-File (Join-Path $PSScriptRoot 'audit\AUDIT.md')
                $pdf = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'out') `
                    -Filter '*.pdf' -ErrorAction SilentlyContinue |
                    Select-Object -First 1
                if ($pdf) { Open-File $pdf.FullName }
                else { Write-Host "[open] No se encontro PDF en out/" -ForegroundColor Red }
            }
            'prompt' {
                Open-File (Join-Path $PSScriptRoot 'audit\PROMPT.md')
            }
        }
    }
}
finally {
    Pop-Location
}
