---
name: update-pkg-docs
description: Lee el código de un paquete de mPES y actualiza su documentación (doc/<pkg>_explained.md, doc/<pkg>_theory.md) y sus docstrings NumPy para que reflejen fielmente la implementación; refresca el README.md raíz si cambian hechos globales.
argument-hint: "<paquete>|all"
disable-model-invocation: true
---

# Update Package Documentation

Objetivo: `$ARGUMENTS`.

1. Lee `.github/prompts/update-pkg-docs.prompt.md` completo y sigue sus pasos 1–5 y el checklist final.
2. Los `.md` de `doc/` van en español con KaTeX; el `README.md` raíz, en inglés y bajo ~200 líneas. No generes HTML.
3. Los docstrings se editan dentro de los `.py` (en inglés, estilo NumPy); después de tocarlos corre la skill
   `lint-and-typecheck` sobre el paquete.
4. Valida los `.md` editados con `win_mpes_env\Scripts\pymarkdown.exe scan <ruta>`.
5. Reporta qué discrepancias se corrigieron y cuáles quedaron sin resolver.
