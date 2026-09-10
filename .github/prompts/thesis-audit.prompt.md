---
description: "Audita y compila la tesis LaTeX de writings/: ejecuta writings/audit/audit.py, lee AUDIT.md y corrige iterativamente los problemas hasta que la auditoría y pdflatex queden limpios. Solo modifica archivos dentro de writings/ (incluido audit.py cuando el defecto está en el propio script)."
---

# Thesis Audit (Fix Loop)

> Last updated: 2026-09-10

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
- `writings/audit/audit.py` **puede modificarse** únicamente para corregir
  defectos del propio script: falsos positivos (p. ej. palabras tomadas de
  comentarios `%` o bloques `verbatim`), rutas incorrectas hacia `h1/`,
  errores de tipado o de ejecución. **Nunca** relajar, eliminar o
  desactivar criterios para "aprobar" la auditoría: los hallazgos
  legítimos se corrigen en los `.tex`/`.bib`. Tras editar `audit.py`,
  ejecutar desde la raíz del workspace
  `pyright --project utils\config\pyrightconfig.json writings\audit\audit.py`
  y `pylint --rcfile=utils\config\.pylintrc writings\audit\audit.py`
  hasta obtener `0 errors` y `10.00/10`, y reportar qué criterio se
  ajustó y por qué.
- Los artefactos de compilación viven en `writings/out/`; no versionarlos
  ni moverlos.

## Revisión como juez evaluador

Antes de corregir los hallazgos de la auditoría, asume el rol de **juez
evaluador de una tesis de maestría** y revisa críticamente el documento,
contrastándolo con `h1/`, que es la fuente de verdad experimental. La
revisión debe ser técnica, independiente y verificable; no debes aprobar una
afirmación solo porque esté redactada con seguridad.

Comprueba explícitamente:

1. **Imágenes y tablas actuales**: verifica que cada figura, tabla, cifra y
  pie de figura empleado en la tesis corresponda a los resultados vigentes
  de `h1/`, sus scripts, configuraciones y archivos de resultados. Comprueba
  modelo, escenario, muestra, métrica y periodo. Señala artefactos antiguos,
  referencias sin trazabilidad o resultados que no puedan verificarse. No
  inventes reemplazos ni des por actual una figura solo por su nombre.
2. **Hipótesis y conclusiones**: localiza dónde se formula, analiza y
  responde cada hipótesis. Determina si los datos permiten sostenerla,
  matizarla o rechazarla, y si las conclusiones son específicas,
  proporcionales a la evidencia y aportan valor. Señala hipótesis no
  abordadas, argumentos circulares, saltos inferenciales y conclusiones que
  no respondan a las preguntas de investigación.
3. **Tono académico**: revisa que el tono técnico sea adecuado para un
  maestrando en Ciencia de Datos y se mantenga de forma consistente en todo
  el escrito. Marca lenguaje promocional, coloquialismos, grandilocuencia,
  vaguedad o cambios injustificados de registro.
4. **Rigor y contenido espurio**: busca señales de *AI slop* (texto genérico,
  repetitivo o inflado), alucinaciones, citas que no respaldan lo afirmado,
  temas o métodos no realizados en el trabajo, métricas no estudiadas o
  inferidas sin cálculo, y conclusiones sin sentido, sin evidencia o sin
  valor agregado. Describe la señal concreta y la evidencia que la confirma
  o contradice; no declares que un texto fue generado por IA únicamente por
  su estilo.

Para cada hallazgo registra archivo, sección, figura o tabla afectada,
evidencia contrastada y severidad (`crítico`, `mayor`, `menor` u
`observación`). Separa hechos comprobados, afirmaciones no verificables,
contradicciones con `h1/` y recomendaciones. Si falta información para
decidir, indícalo como verificación pendiente en lugar de convertir la
ausencia de evidencia en evidencia de ausencia.

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
- Archivos de `writings/` modificados y por qué; si se tocó `audit.py`,
  indicar el criterio ajustado, la causa del falso positivo o error y el
  resultado de pyright/pylint.
- Ubicación del PDF generado (modo `full`).
- Cualquier hallazgo que no pueda resolverse editando `writings/`
  (p. ej. una figura que debe regenerarse desde `h1/`), indicando el
  agente adecuado para hacerlo.
