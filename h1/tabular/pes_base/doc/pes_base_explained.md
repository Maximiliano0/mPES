# pes_base — Guía de uso e implementación

> Paquete: `tabular.pes_base`  
> Algoritmo: **Q-Learning tabular**  
> Estado: **activo**, base del benchmark mPES

---

## 1. ¿Qué es este paquete?

`pes_base` es la línea base del proyecto. Implementa un agente de Q-Learning tabular sobre el entorno `Pandemic` para aprender la política más económica de asignación de recursos frente a la severidad emergente.

Es el punto de referencia principal para comparar con variantes más avanzadas:

- `pes_ql` añade optimización bayesiana con Optuna,
- `pes_dql` añade doble Q-Learning y decay de epsilon,
- `pes_dqn`, `pes_rdqn`, `pes_a2c` y `pes_trf` reemplazan la tabla por redes neuronales.

---

## 2. Comandos de ejecución

Desde la raíz del repositorio:

```powershell
win_mpes_env\Scripts\Activate.ps1
cd h1
python -m tabular.pes_base
```

También puede entrenarse directamente con el módulo de entrenamiento de la línea base:

```powershell
cd h1
python -m tabular.pes_base.ext.train_rl 20000
```

---

## 3. Estructura principal

```text
h1/tabular/pes_base/
├── __init__.py
├── __main__.py
├── config/CONFIG.py
├── doc/
├── ext/
├── inputs/
├── outputs/
├── src/
└── README.md
```

Los archivos más relevantes son:

- `ext/pandemic.py`: entorno de simulación y lógica de transición.
- `ext/train_rl.py`: entrenamiento del agente tabular.
- `config/CONFIG.py`: hiperparámetros del problema.
- `src/exp_utils.py`: cálculo de severidades y secuencias.

---

## 4. Pipeline de trabajo

1. Se prepara la simulación del escenario Pandémico.
2. El agente explora el entorno y actualiza una tabla `Q[s, a]`.
3. La política se ejecuta sobre secuencias de bloques / trials.
4. Se registran métricas de rendimiento, recompensas y respuestas.

La política resultante se consume luego por el experimento general bajo `h1/general`.

---

## 5. Salidas esperadas

El paquete genera artefactos en `inputs/` y `outputs/` con:

- `q.npy` y `rewards.npy` para el agente entrenado,
- logs de sesión,
- resultados del experimento y métricas de rendimiento.

---

## 6. Referencia rápida

- Teoría: `pes_base_theory.md`
- Benchmark general: `h1/general/README.md`

> Este documento es el archivo canónico del paquete. Los documentos antiguos con nombres alternativos quedan como referencias legacy y no deben usarse como documentación activa.
