---
name: lint-and-typecheck
description: Ejecuta el bucle obligatorio pyright + pylint sobre un paquete de mPES y corrige en el código fuente hasta pyright 0 errors/0 warnings/0 informations y pylint 10.00/10. Úsala después de crear o modificar cualquier .py.
argument-hint: "<paquete> (p. ej. pes_ens, ml/pes_dqn, general, utils, writings/audit/audit.py)"
---

# Lint & Type-Check

Objetivo: `$ARGUMENTS` (si está vacío, infiere el paquete del último `.py` editado).

1. Lee `.github/prompts/lint-and-typecheck.prompt.md`: es la fuente canónica (resolución de `$PACKAGE_DIR` →
   `$TARGET_PATH`, comandos, tabla de correcciones comunes y reglas). Síguelo al pie de la letra.
2. Comandos desde la raíz del repo, nunca desde `h1/` ni `h2/` (PowerShell):

   ```powershell
   win_mpes_env\Scripts\Activate.ps1
   pyright --project utils\config\pyrightconfig.json <TARGET_PATH>
   pylint --rcfile=utils\config\.pylintrc <TARGET_PATH>
   ```

3. Repite hasta que ambas metas se cumplan en la misma iteración. Prohibido `# type: ignore`, `# pylint: disable=`
   para reglas exigidas y editar `.pylintrc` / `pyrightconfig.json`. Los cambios deben ser neutros en comportamiento.
4. Reporta la salida final de ambas herramientas (conteos y puntaje).
