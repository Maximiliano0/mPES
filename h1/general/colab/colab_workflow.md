# Flujo actual de mPES en Google Colab

> Guia para los notebooks de `h1/general/colab`.

## Preparacion de una sesion

Sube estas dos carpetas a `/content/drive/MyDrive/mPES/`:

```text
mPES/
├── h1/
└── utils/
   └── config/requirements.txt
```

`launch.ipynb` copia ambas carpetas a `/content/mPES/` y ejecuta los modulos
desde `/content/mPES/h1`. No es necesario clonar el repositorio ni crear una
carpeta `models/` separada.

1. Abre el notebook en Colab y monta Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. `launch.ipynb` copia `h1` y `utils` al almacenamiento local y prepara el entorno:

   ```bash
   cd /content/mPES
   bash h1/general/colab/setup_colab.sh
   ```

   El script no instala el archivo completo
   `utils/config/requirements.txt`. Comprueba las dependencias runtime exactas
   necesarias para optimizacion y entrenamiento: `numpy==2.4.3`,
   `tensorflow==2.21.0`, `keras==3.13.2`, `gymnasium==1.2.3`,
   `optuna==4.7.0`, `matplotlib==3.10.8` y `pygame==2.5.2`. Si una dependencia
   falta o su version no coincide, instala la version fijada; si ya coincide,
   no realiza ninguna instalacion. El script no usa `MPES_FAST_SETUP`.
   Tambien comprueba Drive y genera `/content/mpes_env.sh`.

3. Los modulos Python se ejecutan desde `/content/mPES/h1`:

   ```bash
   cd /content/mPES/h1
   source /content/mpes_env.sh
   python -m <modulo>
   ```

## Optimizacion: `launch.ipynb`

Usa [`launch.ipynb`](launch.ipynb) para optimizar modelos individuales o los
parametros de un ensemble. En la primera celda configura `PKG`, `N_TRIALS`,
`RESUME_DATE` (opcional) y `USE_GPU`. Los valores admitidos son `ql`, `dql`, `dqn`, `rdqn`, `ac`, `tr`,
`ens_sprb` y `ens_accq`.

Ejecuta todas las celdas. La celda de lanzamiento ejecuta exactamente:

```bash
cd /content/mPES
source /content/mpes_env.sh
bash h1/general/colab/run_colab.sh "$PKG" "$N_TRIALS" "$RESUME_DATE"
```

Los aliases individuales corresponden a estos modulos:

| `PKG` | Modulo Python |
| --- | --- |
| `ql` | `tabular.pes_ql.ext.optimize_rl` |
| `dql` | `tabular.pes_dql.ext.optimize_rl` |
| `dqn` | `ml.pes_dqn.ext.optimize_dqn` |
| `rdqn` | `ml.pes_rdqn.ext.optimize_rdqn` |
| `ac` | `ml.pes_a2c.ext.optimize_a2c` |
| `tr` | `ml.pes_trf.ext.optimize_tr` |

Para reanudar, pon en `RESUME_DATE` la fecha de la carpeta existente, por
ejemplo `2026-09-03`.

### Ensembles

`ens_sprb` y `ens_accq` no entrenan redes. Solo optimizan sus parametros de
votacion despues de disponer de estos tres modelos compatibles:

```text
dqn_model.keras
rdqn_model.keras
trf_model.keras
```

Los modelos se buscan automáticamente dentro de la copia local de `h1`, en
`ml/pes_dqn/inputs`, `ml/pes_rdqn/inputs` y `ml/pes_trf/inputs`. Si falta
cualquiera de los tres archivos, el ensemble no puede iniciarse.

Los comandos directos son:

```bash
cd /content/mPES
source /content/mpes_env.sh
bash h1/general/colab/run_colab.sh ens_sprb 50
bash h1/general/colab/run_colab.sh ens_accq 50
```

## Reentrenamiento GPU: `retrain_gpu.ipynb`

Usa [`retrain_gpu.ipynb`](retrain_gpu.ipynb) unicamente para `dqn`, `rdqn` y
`tr`. Selecciona un runtime GPU y ejecuta todas las celdas. El notebook ejecuta
desde `/content/mPES/h1` los modulos de entrenamiento y copia los
artefactos `.keras` y `.npy` a:

```text
/content/drive/MyDrive/mPES/<pes_dqn|pes_rdqn|pes_trf>/
```

Los ensembles se optimizan despues desde `launch.ipynb` usando los modelos de `h1`.

## Outputs y monitorizacion

Cada ejecucion de `run_colab.sh` guarda los outputs de optimizacion en:

```text
/content/drive/MyDrive/mPES/<paquete>/<YYYY-MM-DD>_BAYESIAN_OPT/
```

La carpeta contiene la base SQLite de Optuna, `trials.csv`, los mejores
parametros, `study_plots/`, `run_meta.json`, `optimize.pid`,
`bayesian_opt.log` y `bayesian_opt_err.log`.

Usa [`monitor.ipynb`](monitor.ipynb) para consultar los runs en Drive sin
interferir con ellos. Para los seis modelos individuales tambien puedes usar
[`check_progress.py`](check_progress.py) desde `/content/mPES/h1`:

```bash
cd /content/mPES/h1
python general/colab/check_progress.py --pkg pes_dqn --date 2026-09-03
```

`check_progress.py` admite `pes_ql`, `pes_dql`, `pes_dqn`, `pes_rdqn`,
`pes_a2c` y `pes_trf`; no admite los ensembles. Para esos runs usa
`monitor.ipynb` y los archivos de la carpeta de Drive.
