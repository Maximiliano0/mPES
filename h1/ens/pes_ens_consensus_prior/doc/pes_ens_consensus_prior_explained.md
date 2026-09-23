# pes_ens_consensus_prior — Guía de uso

> Paquete: `ens.pes_ens_consensus_prior`  
> Algoritmo: **consenso con prior basado en severidad y confianza ponderada**

---

## 1. ¿Qué es este paquete?

`pes_ens_consensus_prior` combina las decisiones de varios modelos activos del proyecto con un criterio de consenso que incorpora:

- votación por acción ponderada por confianza,
- penalización por desacuerdo entre miembros,
- bono por acuerdo entre modelos,
- prior informativo basado en la severidad del estado actual.

Este enfoque está diseñado para decisiones más estables cuando la situación de riesgo o severidad es alta, sin permitir que una sola predicción domine la elección final.

---

## 2. Comandos principales

Desde `h1/`:

```powershell
python -m ens.pes_ens_consensus_prior
```

Optimización:

```powershell
python -m ens.pes_ens_consensus_prior.ext.optimize_ens 50
```

---

## 3. Lógica de agregación

El ensemble toma la salida de los modelos activos:

- `pes_dqn`
- `pes_a2c`
- `pes_rdqn`
- `pes_trf`

Cada miembro produce una distribución de acción o un vector de valores Q, que se transforma en una probabilidad de acción. Luego se calculan:

- una confianza por miembro basada en la entropía normalizada inversa,
- un peso por modelo usando la confianza y un exponente configurable,
- un término de acuerdo y desacuerdo sobre la acción elegida,
- un prior severidad-adaptado que modula la decisión final según el nivel de riesgo del estado.

La salida final favorece decisiones consistentes con la mayoría y con los modelos más confiables, pero incorpora un prior que puede orientar la acción hacia niveles de riesgo más adecuados cuando la severidad del caso es alta.

---

## 4. Prior de severidad

El paquete incluye un prior informativo cuyo centro depende de la severidad del estado actual. En la implementación, la severidad se estima a partir del estado y se usa para construir una distribución gaussiana sobre las acciones posibles.

Este prior sirve para:

- suavizar decisiones muy ruidosas,
- reforzar acciones compatibles con el nivel de severidad observado,
- evitar que un único modelo con alta confianza imponga una acción sin tener en cuenta el contexto del estado.

Los parámetros clave de esta componente se definen en la configuración del paquete:

- `DEFAULT_PRIOR_WEIGHT`
- `DEFAULT_PRIOR_SIGMA`
- `MAX_SEVERITY`

---

## 5. Parámetros relevantes

La configuración del paquete usa los siguientes valores por defecto:

- `ACTION_COUNT = 11`
- `DEFAULT_WEIGHTS = {'dqn': 0.15, 'a2c': 0.10, 'rdqn': 0.25, 'trf': 0.50}`
- `DEFAULT_CONFIDENCE_POWER = 1.0`
- `DEFAULT_PRIOR_WEIGHT = 0.10`
- `DEFAULT_PRIOR_SIGMA = 1.5`
- `MAX_SEVERITY = 9`

Estos valores controlan la fuerza relativa de cada modelo, la sensibilidad de la confianza y la influencia del prior severidad.

---

## 6. Salidas esperadas

El paquete genera resultados de evaluación y métricas de desempeño en los directorios `inputs/` y `outputs/` del paquete, además de participar en la comparación del benchmark general.

La evaluación (`ext/evaluate_ens.py`) escribe resultados en `outputs/` y produce artefactos como:

- JSON/PNG de resultados,
- vector de rendimiento por secuencia,
- registro de acciones por trial,
- archivos de diagnóstico del ensemble.

---

## 7. Referencia

- Teoría: `pes_ens_consensus_prior_theory.md`
- Benchmark general: `h1/general/README.md`
- Paquete análogo: `ens.pes_ens_consensus`
