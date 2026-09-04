# pes_ens_sprb — Guía de uso

> Paquete: `ens.pes_ens_sprb`  
> Algoritmo: **ensemble por votación suave ponderada**

---

## 1. ¿Qué es este paquete?

`pes_ens_sprb` combina los modelos ya entrenados de `pes_dqn`, `pes_rdqn`, `pes_a2c` y `pes_trf` mediante una votación ponderada por confianza. La estrategia principal es la agregación de distribuciones de probabilidad, no la elección dura de una sola acción.

Esta variante prioriza la estabilidad del voto y la preservación de incertidumbre: el modelo final no colapsa instantáneamente a una única acción si varios miembros tienen distribuciones plausibles.

---

## 2. Comandos principales

Desde `h1/`:

```powershell
python -m ens.pes_ens_sprb
```

Para optimización con Optuna:

```powershell
python -m ens.pes_ens_sprb.ext.optimize_ens 50
```

---

## 3. Miembros del ensemble

Los miembros base son los modelos de la línea activa:

- `pes_dqn`
- `pes_rdqn`
- `pes_a2c`
- `pes_trf`

La configuración del paquete define pesos y temperaturas de agregación para combinar sus salidas sin acoplar el paquete a un código fuente cruzado entre líneas.

---

## 4. Modo operativo

El paquete usa:

- distribución de acciones por miembro,
- enmascarado por factibilidad,
- ponderación por confianza,
- agregación suave final.

Esto produce una decisión final que incorpora la incertidumbre de cada modelo y reduce los empates artificiales típicos de la votación dura.

---

## 5. Salidas esperadas

El paquete escribe artefactos en sus carpetas `inputs/` y `outputs/`, junto con los resultados del ensemble y los informes de evaluación asociados.

---

## 6. Referencia

- Teoría: `pes_ens_sprb_theory.md`
- Benchmark general: `h1/general/README.md`
