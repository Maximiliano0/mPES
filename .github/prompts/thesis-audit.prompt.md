---
description: "Audita y compila la tesis LaTeX de writings/: ejecuta writings/audit/audit.py, lee AUDIT.md y corrige iterativamente los problemas hasta que la auditoría y pdflatex queden limpios. Solo modifica archivos dentro de writings/."
---

# Thesis Audit (Fix Loop)

> Last updated: 2026-09-08

Ejecuta la auditoría programática de la tesis y **corrige iterativamente
cada problema** hasta que el informe quede limpio y el documento compile
sin errores. Este prompt es la fuente canónica del flujo de auditoría de
`writings/`; los agentes que lo referencien deben leer este archivo con
`read_file` y seguir sus pasos (los prompts no se ejecutan solos).

## Inputs

- `$MODE` (opcional) — `full` (por defecto: audita + compila con
  pdflatex/bibtex) o `quick` (`--no-tex`, solo criterios programáticos,
  sin compilación).

## Restricciones

- **Escritura solo dentro de `writings/`**: correcciones en `.tex`, `.bib`
  e imágenes de `writings/02_Images/`. Nunca modificar `h1/`, `h2/`, `h3/`,
  `utils/` ni `.github/` desde este flujo.
- No cambiar el preámbulo de `writings/00_Main/Main.tex` (paquetes,
  geometría, estilo) salvo que un error de compilación lo exija y no haya
  alternativa; en ese caso, explicarlo al usuario.
- No editar `writings/audit/audit.py` ni relajar sus criterios para
  "aprobar" la auditoría. Los problemas se corrigen en los `.tex`/`.bib`.
- Los artefactos de compilación viven en `writings/out/`; no versionarlos
  ni moverlos.

## Workflow

Repite el bucle hasta que la auditoría reporte **cero problemas** y la
compilación termine sin errores en la misma iteración.

```
while issues remain:
    1. Ejecutar audit.py → leer AUDIT.md y stdout
    2. Corregir cada hallazgo en writings/ (tex/bib/imágenes)
    3. Re-ejecutar
```

### Paso 0 — Entorno y directorio de trabajo

```powershell
win_mpes_env\Scripts\Activate.ps1
cd writings
$env:PYTHONIOENCODING = "utf-8"
```

El script resuelve sus rutas con `Path(__file__)`, pero ejecútalo con
`writings/` como cwd, tal como documenta su docstring.

### Paso 1 — Ejecutar la auditoría

```powershell
# Modo full (audita + pdflatex/bibtex)
python audit\audit.py

# Modo quick (solo criterios, sin compilar)
python audit\audit.py --no-tex
```

### Paso 2 — Leer el informe y corregir

1. Lee el informe en [writings/audit/AUDIT.md](../../writings/audit/AUDIT.md)
   y la salida de consola completa.
2. Para **cada** criterio fallido o warning:
   - Abre el archivo y línea reportados.
   - Corrige la causa raíz: referencias `\ref`/`\cite` rotas, claves `.bib`
     faltantes o duplicadas, figuras inexistentes o sin `\label`, capítulos
     no incluidos por `Main.tex`, encabezado `% !TEX root` ausente, etc.
3. Si se compiló, revisa el log en `writings/out/` ante cualquier error de
   pdflatex/bibtex: undefined references, missing figures, errores de
   sintaxis LaTeX. Los warnings de underfull/overfull hbox no bloquean,
   pero repórtalos si son numerosos.
4. Vuelve al Paso 1.

## Criterio de éxito

- `audit.py` termina con todos los criterios en verde en `AUDIT.md`.
- En modo `full`, pdflatex/bibtex generan el PDF en `writings/out/` sin
  errores ni referencias indefinidas.

## Reporte final

- Resultado de cada criterio de la auditoría (no solo "se ejecutó").
- Archivos de `writings/` modificados y por qué.
- Ubicación del PDF generado (modo `full`).
- Cualquier hallazgo que no pueda resolverse editando `writings/`
  (p. ej. una figura que debe regenerarse desde `h1/`), indicando el
  agente adecuado para hacerlo.
