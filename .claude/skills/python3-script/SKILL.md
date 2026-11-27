---
name: python3-script
description: Genera un script Python 3 nuevo dentro de un paquete de mPES siguiendo la estructura y convenciones del proyecto (docstring, secciones de imports, main(), terminal_utils, rutas fechadas, numpy sin alias np) y documenta su fundamento en doc/.
argument-hint: "<paquete destino> <propósito del script>"
---

# Script Python 3 con las convenciones de mPES

Argumentos: `$ARGUMENTS` (si falta el propósito, pregunta).

1. Lee `.github/prompts/python3-script.prompt.md` completo: estructura obligatoria, convenciones, esqueleto de
   ejemplo y flujo (usar `h1/tabular/pes_ql` como referencia; `h1/ml/pes_dqn` para deep learning).
2. Respeta los permisos de escritura del agente `python-h1` (no escribir en `h1/ml/` ni `h1/tabular/` sin
   autorización del usuario).
3. Cierra con la skill `lint-and-typecheck` sobre el paquete y reporta el resultado.
