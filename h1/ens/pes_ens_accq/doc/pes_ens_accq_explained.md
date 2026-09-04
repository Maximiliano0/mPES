# pes_ens_accq — Guía de uso

> Paquete: `ens.pes_ens_accq`  
> Algoritmo: **ensemble basado en votación de acciones con desempate por Q-value**

---

## 1. ¿Qué es este paquete?

`pes_ens_accq` combina los miembros activos del proyecto para producir una decisión final sobre la acción a ejecutar. A diferencia de una votación por probabilidad simple, esta variante prioriza la acción más plausible y resuelve empates con información de valor Q cuando hay conflicto entre miembros.

Este enfoque es útil cuando se desea que la decisión final siga siendo muy interpretable pero con un criterio de desempate más estructurado que el mero promedio de distribuciones.

---

## 2. Comandos principales

Desde `h1/`:

```powershell
python -m ens.pes_ens_accq
```

Optimización Bayesiana:

```powershell
python -m ens.pes_ens_accq.ext.optimize_ens 50
```

---

## 3. Miembros integrados

La variante usa los modelos base del benchmark activo:

- `pes_dqn`
- `pes_rdqn`
- `pes_a2c`
- `pes_trf`

La salida de cada miembro se transforma en una preferencia de acción y luego se agrega con una regla de confianza y desempate por valor estimado.

---

## 4. Regla de decisión

La decisión de ensemble se basa en:

1. factibilidad por recursos disponibles,
2. ponderación por confianza del miembro,
3. consenso de acciones,
4. desempate por valor Q normalizado si hay conflicto.

Esto hace que la salida sea más robusta que la simple regla de mayoría, pero sin perder la semántica de decisión directa del agente.

---

## 5. Salidas esperadas

El paquete genera resultados en `inputs/` y `outputs/` y participa en las evaluaciones del benchmark general del proyecto.

---

## 6. Referencia

- Teoría: `pes_ens_accq_theory.md`
- Benchmark general: `h1/general/README.md`
