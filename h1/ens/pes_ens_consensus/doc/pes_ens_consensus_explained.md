# pes_ens_consensus — Guía de uso

> Paquete: `ens.pes_ens_consensus`  
> Algoritmo: **consenso con recompensa por acuerdo y penalización por desacuerdo**

---

## 1. ¿Qué es este paquete?

`pes_ens_consensus` agrega las decisiones de varios modelos activos del proyecto con un criterio de consenso que favorece la agreement entre miembros y penaliza las decisiones que divergen demasiado entre sí.

Este enfoque resulta útil cuando se busca una acción final más estable y menos impulsada por un único miembro con alta confianza pero poca concordancia con el resto del ensemble.

---

## 2. Comandos principales

Desde `h1/`:

```powershell
python -m ens.pes_ens_consensus
```

Optimización:

```powershell
python -m ens.pes_ens_consensus.ext.optimize_ens 50
```

---

## 3. Lógica de agregación

La política del ensemble combina:

- apoyos de cada miembro,
- acuerdo global sobre acciones candidatas,
- penalización por desacuerdo,
- ponderación de confianza.

La salida final favorece decisiones en las que la mayoría o los miembros más confiables coinciden, pero evita que una sola voz domine sin respaldo colectivo.

---

## 4. Miembros integrados

El paquete trabaja con los modelos activos del benchmark:

- `pes_dqn`
- `pes_rdqn`
- `pes_a2c`
- `pes_trf`

---

## 5. Salidas esperadas

Genera resultados de evaluación y métricas de desempeño en los directorios `inputs/` y `outputs/` del paquete, además de participar en la comparación del benchmark general.

---

## 6. Referencia

- Teoría: `pes_ens_consensus_theory.md`
- Benchmark general: `h1/general/README.md`
