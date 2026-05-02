# Guía de Optimización Bayesiana — Q-Learning

## 1. Lanzar la optimización

```bash
cd /home/mecatronica/Documentos/maximiliano/mPES
source linux_mpes_env/bin/activate

# Lanzar con 100 trials (valor por defecto)
python3 -m pes_ql.ext.optimize_rl

# Lanzar con N trials en segundo plano (ejemplo: 200)
nohup python3 -m pes_ql.ext.optimize_rl 200 \
  > pes_ql/inputs/bayesian_opt.log 2>&1 &
```

Los resultados se guardan en `pes_ql/inputs/<FECHA>_BAYESIAN_OPT/`.

---

## 2. Evitar que la PC se suspenda

Inmediatamente después de lanzar la optimización:

```bash
# Obtener el PID del proceso
pgrep -f "pes_ql.ext.optimize_rl"

# Bloquear suspensión (reemplazar <PID> con el número obtenido)
nohup systemd-inhibit \
  --what=idle:sleep:handle-lid-switch \
  --who="mPES Bayesian Optimization" \
  --why="Running Bayesian optimization" \
  --mode=block \
  tail --pid=<PID> -f /dev/null > /dev/null 2>&1 &
```

El inhibidor se desactiva automáticamente cuando la optimización termina.

Si GNOME sigue suspendiendo al cerrar la tapa, desactivar eso explícitamente:

```bash
gsettings set org.gnome.settings-daemon.plugins.power lid-close-ac-action 'nothing'
```

---

## 3. Reanudar tras interrupción (--resume)

Cada trial completado se guarda en una base de datos SQLite (`optuna_study_<FECHA>.db`).
Si el proceso se interrumpe (suspensión, apagado, crash), los trials completados
se conservan.

### 3.1 Opciones de línea de comando

| Flag | Default | Descripción |
|------|---------|-------------|
| `n_trials` (posicional) | `100` | Número objetivo de trials Optuna. |
| `--resume YYYY-MM-DD` | (sin) | Reanuda un estudio previo de esa fecha. Sin esta opción se crea uno nuevo con la fecha actual. |
| `--out-dir PATH` | `inputs/<fecha>_BAYESIAN_OPT` | Directorio destino para artefactos. Útil para escribir en Google Drive desde Colab. |
| `--storage URL` | `sqlite:///<out-dir>/optuna_study_<fecha>.db` | URL alternativo de Optuna storage (p.ej. SQLite en Drive). |

**Mismo día** — re-ejecutar el mismo comando:

```bash
nohup python3 -m pes_ql.ext.optimize_rl 100 \
  > pes_ql/inputs/bayesian_opt.log 2>&1 &
```

**Día diferente** — usar `--resume` con la fecha original:

```bash
nohup python3 -m pes_ql.ext.optimize_rl 100 --resume 2026-02-12 \
  > pes_ql/inputs/bayesian_opt_resume.log 2>&1 &
```

El script detecta los trials previos y ejecuta solo los restantes:

```
ℹ Resuming: 45 trials already completed, 55 remaining
```

---

## 4. Interpretar los logs

### Ver progreso

```bash
grep "Trial" pes_ql/inputs/bayesian_opt.log | tail -10
```

### Formato de salida

```
  Trial   1/100  |  value=0.7563  |  best=0.7563  |  elapsed=19s
  Trial   2/100  |  value=0.6941  |  best=0.7563  |  elapsed=76s
  Trial   8/100  |  value=0.8254  |  best=0.8254  |  elapsed=274s
```

| Campo | Significado |
|-------|-------------|
| `Trial N/100` | Número de trial actual / total solicitado |
| `value` | Rendimiento medio normalizado de este trial (0–1, mayor = mejor) |
| `best` | Mejor rendimiento encontrado hasta ahora entre todos los trials |
| `elapsed` | Tiempo transcurrido desde el inicio de esta corrida |

### Ver log en tiempo real

```bash
tail -f pes_ql/inputs/bayesian_opt.log
```

### Verificar que el proceso sigue vivo

```bash
pgrep -f "pes_ql.ext.optimize_rl" -a
```

Si no devuelve nada, el proceso terminó (completó todos los trials o fue interrumpido).
Revisar el final del log para determinar qué ocurrió:

```bash
tail -5 pes_ql/inputs/bayesian_opt.log
```

### Tiempo estimado

Desde abril 2026 la optimización llama a `QLearning(...,
track_confidence=False)`, lo que omite el cálculo de
`rl_agent_meta_cognitive` por paso (no afecta a `mean_perf`) y reduce el
coste de cada trial entre 5 y 10 ×. Combinado con `MedianPruner`, los
tiempos típicos son:

- Decenas de segundos a pocos minutos por trial completado, dependiendo
  del `num_episodes` muestreado entre 500k y 1.2M
- 100 trials suele tomar entre 30 y 90 minutos
- 200 trials puede tomar entre 1 y 3 horas

Usar el script `utils/run_bayesian_opt.sh` para lanzar con inhibición de suspensión automática.

---

## 5. Flujo de trabajo completo

La optimización es el primer paso de un flujo de tres etapas:

### Paso 1 — Optimizar hiperparámetros

```bash
python3 -m pes_ql.ext.optimize_rl 100
```

Genera la Q-table óptima y un reporte en `inputs/<fecha>_BAYESIAN_OPT/`,
y además **espeja** automáticamente `q.npy`, `rewards.npy` y
`best_params.json` a `pes_ql/inputs/` para que el experimento (paso 3)
pueda ejecutarse sin pasos manuales adicionales tras la optimización.
Anotar los mejores hiperparámetros del reporte
(`optimization_results_<fecha>.txt`).

### Paso 2 — Entrenar con hiperparámetros óptimos

`optimize_rl.py` escribe `best_params_<fecha>.json` y
`repro_fingerprint_<fecha>.json` dentro de
`inputs/<fecha>_BAYESIAN_OPT/`. `train_rl.py` los descubre automáticamente
(busca el directorio con la fecha más reciente bajo `inputs/`), por lo que
**no es necesario copiar manualmente los hiperparámetros al código**:

```bash
python3 -m pes_ql.ext.train_rl
```

El script:

- Carga los hiperparámetros y la semilla por‑trial (`SEED + best_trial_number + 1`)
  desde el JSON.
- Verifica el `repro_fingerprint` contra el entorno actual; muestra una
  advertencia si difiere (la `mean_perf` resultante podría no coincidir).
- Reentrena la Q‑table con esos hiperparámetros y guarda los artefactos en
  `inputs/<fecha>_RL_TRAIN/`.

Si no existe ningún `_BAYESIAN_OPT/`, `train_rl.py` cae en
`_FALLBACK_PARAMS` (los valores históricos del trial #40) e informa el
fallback antes de continuar.

### Paso 3 — Ejecutar el experimento

`optimize_rl.py` ya espejó `q.npy`, `rewards.npy` y `best_params.json` a
`pes_ql/inputs/` al finalizar la corrida (paso 1) y `train_rl.py`
regeneró los mismos archivos tras el reentrenamiento local (paso 2). No
hace falta copiar nada a mano:

```bash
python3 -m pes_ql
```

`__main__.py` valida la existencia de `q.npy` y `rewards.npy` antes de iniciar;
si no los encuentra, indica que se debe ejecutar `train_rl.py` primero.
