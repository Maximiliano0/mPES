# pes_ens — Cómo Probar el Agente Ensamblado

> Última actualización: 2026-05-02

`pes_ens` no se entrena: combina varios modelos ya entrenados.
Antes de usarlo, asegúrate de que existen los archivos `.keras` de
todos los miembros marcados como `enabled = True` en
`config/CONFIG.py`:

| Miembro | Ruta esperada                       | `enabled` (default) |
|---------|-------------------------------------|---------------------|
| DQN     | `pes_dqn/inputs/dqn_model.keras`    | `True`              |
| A2C     | `pes_a2c/inputs/ac_actor.keras`     | `False`             |
| RDQN    | `pes_rdqn/inputs/rdqn_model.keras`  | `True`              |
| TRF     | `pes_trf/inputs/trf_model.keras`    | `True`              |

Si alguno falta, ejecuta el entrenamiento del paquete correspondiente
(`python -m pes_<pkg>`) o usa los scripts de `utils/scripts/` para
finalizar una optimización Bayesiana interrumpida. El archivo de
`pes_a2c` puede no estar presente si nunca se entrenó: por defecto
está `enabled = False` y el ensamble funciona sin él.

## 1. Configuración

Edita `pes_ens/config/CONFIG.py`. Los hiperparámetros relevantes son:

- `ENS_MEMBER_MODELS` — una entrada por miembro
  (`name`, `role`, `path`, `history_len`, `weight`, `enabled`).
  Cambiar `enabled: False` desactiva al miembro sin borrar su
  configuración (útil para comparar variantes 2 vs 3 vs 4 miembros).
- `ENS_SOFTMAX_TEMPERATURE` (default `15.0`) — temperatura del
  softmax aplicado a los miembros tipo Q-network antes del
  promediado. Valores altos suavizan, valores bajos hacen el voto
  más "duro".
- `ENS_SEVERITY_PRIOR_WEIGHT` (default `0.17`) — peso de la mezcla
  con el prior gaussiano centrado en la severidad cruda. `0.0`
  desactiva la mezcla.
- `ENS_SEVERITY_PRIOR_SIGMA` (default `3.0`) — desviación estándar
  del prior, en unidades crudas de severidad.
- `PLAYER_TYPE` — debe ser `'ENS_AGENT'` (único valor admitido).

Los pesos crudos `weight` no necesitan sumar 1; `EnsembleAgent` los
renormaliza internamente. La regla de seguridad
"piso por severidad ≥ 6" no es configurable (siempre activa).

## 2. Ejecución

### Linux

```bash
source linux_mpes_env/bin/activate
export PYTHONIOENCODING=utf-8
export TF_ENABLE_ONEDNN_OPTS=0
python -m pes_ens
```

### Windows (PowerShell)

```powershell
win_mpes_env\Scripts\Activate.ps1
$env:PYTHONIOENCODING = 'utf-8'
$env:TF_ENABLE_ONEDNN_OPTS = '0'
python -m pes_ens
```

`__main__` ejecuta primero una **fase de validación** que carga cada
miembro habilitado con `tf.keras.models.load_model(..., safe_mode=False)`
e imprime los parámetros y peso de cada uno. Si algún archivo falta
o la deserialización falla, el experimento se detiene antes de
empezar.

A continuación corre `NUM_BLOCKS × NUM_SEQUENCES = 64` secuencias,
cada una con 3-10 trials, y produce:

- `outputs/<fecha>_ENS_AGENT/PES_ENS_responses_<id>.txt` — log
  trial-a-trial (severity, response, confidence, RTs).
- `outputs/<fecha>_ENS_AGENT/<id>_results.json` — resumen
  agregado por bloque + media global.
- `outputs/<fecha>_ENS_AGENT/<id>_results.png` — visualización.

## 3. Métrica `raw_mean_perf`

Al final del experimento, el log imprime explícitamente:

```
raw_mean_perf = X.XXXXXX  (std=Y.YYYYYY, n=64)
```

Esta métrica es directamente comparable contra el `mean_perf` de
cualquier otro paquete (`pes_dqn`, `pes_a2c`, `pes_rdqn`, `pes_trf`)
para evaluar si el ensamble mejora sobre sus miembros individuales.

## 4. Diagnóstico

Si `VERBOSE = True` (default), por cada trial verás algo como:

```
State indices - Resources: 18, City: 2, Severity: 7
  member dqn    top= 3 p_top=0.412
  member rdqn   top= 3 p_top=0.475
  member trf    top= 4 p_top=0.501
  ENSEMBLE      top= 4 p_top=0.437
ENS Agent Response: 4, Confidence: 0.382
```

Útil para detectar miembros consistentemente desviados del consenso
(p. ej. si A2C, cuando se reactiva, vota sistemáticamente acciones
distintas del resto, puede convenir reducir su `weight` o
reentrenarlo).

Recuerda que el `top` impreso para cada miembro corresponde a la
distribución **antes** del enmascarado por factibilidad y antes del
voto ponderado por confianza; el `top` del `ENSEMBLE` ya incluye:

1. Máscara de factibilidad por miembro + renormalización.
2. Voto ponderado por confianza (`dyn_weight = w_norm * (0.1 + (1 −
   H_norm))`).
3. Penalización de la acción `0` (`* 0.3`) cuando hay presupuesto.
4. Mezcla con el prior gaussiano de severidad
   (`ENS_SEVERITY_PRIOR_WEIGHT`, `ENS_SEVERITY_PRIOR_SIGMA`).
5. Regla de piso por severidad si `severidad ≥ 6` y
   `argmax < ⌊severidad/2⌋`.

Consulta `doc/explained_ens.md` para los detalles matemáticos de
cada paso.

## 5. Cambios respecto a paquetes hermanos

| Aspecto             | pes_ens                                    |
|---------------------|--------------------------------------------|
| Entrenamiento       | Ninguno — inferencia pura                  |
| Optimización        | Ninguna (voto manual + prior fijo)         |
| GPU                 | Innecesaria; los modelos corren en CPU     |
| Imports cruzados    | Ninguno; los `.keras` se cargan por ruta   |
| Tiempo de ejecución | ≈ N × un agente individual (N forwards)    |
| Restart safety      | `EnsembleAgent.reset_episode` por secuencia|
