# pes_ens_trf_guard — Guía de uso

> Paquete: `ens.pes_ens_trf_guard`  
> Algoritmo: **ensemble con prioridad de Transformer y guardia por confianza**

---

## 1. ¿Qué es este paquete?

`pes_ens_trf_guard` introduce una regla de seguridad para priorizar el modelo Transformer cuando su confianza es suficientemente alta. Si el Transformer no supera el umbral de confianza, el ensemble recurre a los otros miembros para producir la decisión final.

Esta variante intenta combinar la mejor capacidad de generalización del Transformer con la robustez de un respaldo analítico.

---

## 2. Comandos principales

Desde `h1/`:

```powershell
python -m ens.pes_ens_trf_guard
```

Optimización:

```powershell
python -m ens.pes_ens_trf_guard.ext.optimize_ens 50
```

---

## 3. Lógica de decisión

El flujo principal es:

1. inferencia del Transformer,
2. cálculo de confianza,
3. validación del umbral de seguridad,
4. fallback a otros miembros si la confianza no alcanza el valor mínimo.

Esto reduce los errores catastróficos cuando el Transformer se vuelve poco confiable en escenarios adversos.

---

## 4. Miembros integrados

Los miembros activos usados por el ensemble son los modelos principales del proyecto:

- `pes_dqn`
- `pes_rdqn`
- `pes_a2c`
- `pes_trf`

---

## 5. Salidas esperadas

El paquete escribe resultados de evaluación y métricas de desempeño en sus respectivos `inputs/` y `outputs/`.

---

## 6. Referencia

- Teoría: `pes_ens_trf_guard_theory.md`
- Benchmark general: `h1/general/README.md`
