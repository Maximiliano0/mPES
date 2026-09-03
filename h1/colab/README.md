# mPES en Google Colab

Esta guía documenta el uso de [mPES](../../README.md) desde Google Colab mediante el notebook [`mPES_Colab_Runbook.ipynb`](mPES_Colab_Runbook.ipynb) y el runner [`runner.py`](runner.py).

El diseño separa claramente los dos tipos de almacenamiento:

- `/content/mPES` es el repositorio local y el directorio principal de ejecución en Colab.
- `DRIVE_ROOT` es un bucket persistente en Google Drive. Se usa para recuperar y guardar artefactos, no como directorio de trabajo principal.

El runner está pensado para trabajos largos de optimización o entrenamiento. No ejecuta benchmarks.

## 1. Preparar Google Colab

1. Abre `h1/colab/mPES_Colab_Runbook.ipynb` en Google Colab.
2. En **Entorno de ejecución > Cambiar tipo de entorno de ejecución**, selecciona **GPU** para `pes_dqn`, `pes_rdqn`, `pes_a2c` y `pes_trf`.
3. Los paquetes tabulares pueden ejecutarse en CPU. TPU no es necesaria para este flujo.
4. Ejecuta la primera celda para comprobar Python, TensorFlow y las GPU disponibles.

La GPU debe estar seleccionada antes de importar o ejecutar los modelos de TensorFlow. La celda de validación muestra el resultado de `nvidia-smi` y las GPU detectadas por TensorFlow.

## 2. Montar Drive y disponer del repositorio

Ejecuta en Colab:

```python
from google.colab import drive

drive.mount('/content/drive')
```

El notebook clona o actualiza automáticamente el repositorio antes de instalar
los `requirements`. Por defecto usa:

```text
REPO_URL = https://github.com/Maximiliano0/mPES.git
REPO_REF = new_uq
```

La copia usada para ejecutar es siempre:

```text
/content/mPES
```

Si necesitas otra rama o ref, cambia `REPO_URL` y `REPO_REF` en la celda de
configuración del notebook antes de ejecutarlo. `REPO_REF` puede ser una rama,
tag o commit que Git pueda resolver.

El notebook crea `/content/mPES` si no existe y, si ya existe, obtiene y
actualiza esa copia desde el remoto configurado. No es necesario clonarlo ni
subirlo manualmente. Por ejemplo, el flujo equivalente para otra referencia
sería:

```bash
%cd /content
!git clone --branch <REPO_REF> <REPO_URL> mPES
%cd /content/mPES
```

Comprueba que existen `/content/mPES/h1`, `/content/mPES/h2`, `/content/mPES/h3` y `utils/config/requirements.txt`. No uses una carpeta de Drive como `REPO_ROOT`: Colab debe ejecutar sobre el disco local.

Drive solo almacena artefactos persistentes, como `inputs/`, `outputs/`, logs,
estados y manifiestos; no es la copia del repositorio usada para ejecutar.

## 3. Instalar y validar dependencias

Instala las dependencias una vez por sesión:

```python
%pip install -q -r /content/mPES/utils/config/requirements.txt
```

Después, ejecuta la celda de validación del notebook. Debe encontrar al menos `numpy`, `optuna`, `tensorflow` y `gymnasium`.

Si Colab solicita reiniciar el entorno después de instalar, reinícialo y vuelve a ejecutar las celdas desde el principio, incluido el montaje de Drive y la definición de rutas.

## 4. Configurar el bucket persistente

En la celda de configuración del notebook define las variables:

```python
REPO_ROOT = '/content/mPES'
DRIVE_ROOT = '/content/drive/MyDrive/mPES-bucket'
LINE = 'h1'
PACKAGES = 'pes_dqn,pes_rdqn,pes_trf'
OPERATION = 'optimize'
TRIALS = 30
RESUME_DATE = None
RUN_ID = f'{LINE}_{OPERATION}_colab_001'
SYNC_INTERVAL = 300
```

`DRIVE_ROOT` se crea si no existe. El runner copia desde el bucket, antes de cada paquete, sus directorios `inputs/` y `outputs/` al repositorio local. Durante y al finalizar el trabajo copia esos directorios de vuelta a Drive.

No coloques el repositorio completo en `DRIVE_ROOT` ni ejecutes el proceso con `cwd` dentro de Drive. Drive proporciona persistencia frente a la desconexión de Colab; `/content/mPES` proporciona el entorno de ejecución.

`SYNC_INTERVAL` se expresa en segundos y debe ser como mínimo `10`. El valor recomendado para trabajos largos es `300`.

## 5. Paquetes válidos por línea

El runner valida los nombres contra un registro cerrado. Usa únicamente estos paquetes:

| Línea | Paquetes válidos | `optimize` | `train` |
|---|---|---:|---:|
| `h1` | `pes_ql`, `pes_dql`, `pes_dqn`, `pes_rdqn`, `pes_a2c`, `pes_trf`, `pes_ens_sprb`, `pes_ens_accq` | Sí | Sí, excepto los ensembles |
| `h2` | `ql_conf` | Sí | Sí |
| `h3` | `ql_uq` | Sí | Sí |

Los nombres se pasan sin el grupo (`tabular`, `ml`, `ens`, `tabular_conf` o `tabular_uq`). El runner resuelve internamente la ruta del paquete.

Ejemplos de combinaciones válidas:

```text
h1 + pes_dqn,pes_rdqn + optimize
h1 + pes_ens_sprb + optimize
h2 + ql_conf + train
h3 + ql_uq + optimize
```

`h1/pes_base` no está registrado por este runner. Los ensembles `pes_ens_sprb` y `pes_ens_accq` solo se pueden optimizar porque no tienen módulo de entrenamiento registrado.

## 6. Operaciones permitidas

Solo se aceptan dos valores para `OPERATION`:

- `optimize`: ejecuta la optimización bayesiana del paquete con el número de pruebas indicado en `TRIALS`.
- `train`: ejecuta el módulo de entrenamiento del paquete.

El runner no admite una operación de benchmark. En particular, no se deben ejecutar desde este notebook `general/orchestrate.py`, agregaciones, comparaciones ni gráficos de estrés. El campo `benchmarks_enabled` del manifiesto se registra siempre como `false`.

## 7. Ejecutar desde el notebook

Después de pasar las pruebas rápidas de rutas, línea, paquetes y operación, ejecuta la celda de lanzamiento. Conceptualmente construye este comando:

```bash
python /content/mPES/h1/colab/runner.py \
  --line h1 \
  --packages pes_dqn,pes_rdqn,pes_trf \
  --operation optimize \
  --trials 30 \
  --repo-root /content/mPES \
  --drive-root /content/drive/MyDrive/mPES-bucket \
  --run-id h1_optimize_colab_001 \
  --sync-interval 300
```

En el notebook no es necesario copiar este comando manualmente: la celda usa `subprocess.run`, `REPO_ROOT`, `DRIVE_ROOT` y el resto de variables configuradas.

Ejemplos para cada línea:

```bash
# h1: optimizar modelos de aprendizaje profundo
python /content/mPES/h1/colab/runner.py --line h1 --packages pes_dqn,pes_rdqn --operation optimize --trials 30 --repo-root /content/mPES --drive-root /content/drive/MyDrive/mPES-bucket --run-id h1_opt_001

# h1: optimizar un ensemble (no admite train)
python /content/mPES/h1/colab/runner.py --line h1 --packages pes_ens_accq --operation optimize --trials 30 --repo-root /content/mPES --drive-root /content/drive/MyDrive/mPES-bucket --run-id h1_ens_opt_001

# h2: entrenar el paquete experimental registrado
python /content/mPES/h1/colab/runner.py --line h2 --packages ql_conf --operation train --trials 30 --repo-root /content/mPES --drive-root /content/drive/MyDrive/mPES-bucket --run-id h2_train_001

# h3: optimizar el prototipo UQ registrado
python /content/mPES/h1/colab/runner.py --line h3 --packages ql_uq --operation optimize --trials 30 --repo-root /content/mPES --drive-root /content/drive/MyDrive/mPES-bucket --run-id h3_opt_001
```

Aunque el runner está ubicado bajo `h1/colab`, `--line` selecciona explícitamente la línea y el proceso se inicia con la línea correspondiente como directorio de trabajo.

## 8. Sincronización y archivos de estado

El runner sincroniza periódicamente, cada `SYNC_INTERVAL` segundos:

- `inputs/` y `outputs/` del paquete actual.
- Los logs de la ejecución.
- El directorio `.colab_runs/<RUN_ID>/` con sus metadatos.

También hace una sincronización final cuando el proceso termina, tanto si tiene éxito como si falla.

La copia persistente de los metadatos queda en:

```text
<DRIVE_ROOT>/runs/<RUN_ID>/
├── manifest.json
└── <line>/<package>/
    ├── optimize.log o train.log
    └── status.json
```

`manifest.json` resume la ejecución: `run_id`, línea, paquetes, operación, pruebas solicitadas, fecha de reanudación, timestamps, `drive_root`, `benchmarks_enabled: false`, `git_revision` y `git_remote`.

Cada `status.json` informa del paquete y la operación, estado (`running`, `completed` o `failed`), comando, código de retorno y, cuando el log lo permite, pruebas completadas y mejor valor. `last_sync` permite comprobar que la persistencia sigue avanzando.

## 9. Reanudar un trabajo

Define `RESUME_DATE` con una fecha de resultados existente, por ejemplo:

```python
RESUME_DATE = '2026-04-29'
```

La variable se traduce a `--resume-date` del runner y este la adapta así:

- En `optimize`, todos los paquetes registrados reciben `--resume <fecha>`.
- En `train`, solo los modelos profundos de `h1` (`pes_dqn`, `pes_rdqn`, `pes_a2c` y `pes_trf`) reciben `--from-best <fecha>`.
- Para entrenamiento tabular de `h1`, `h2` o `h3`, la fecha no se transforma en una opción de entrenamiento; no debe asumirse que esos trabajos reanudan mediante `RESUME_DATE`.

Para reanudar después de una desconexión, monta Drive de nuevo, usa el mismo `DRIVE_ROOT`, conserva la misma línea y paquete, y selecciona un `RUN_ID` identificable. Recupera los artefactos del bucket antes de lanzar el trabajo. Comprueba `manifest.json` y `status.json` para elegir la fecha y confirmar el último estado conocido.

## 10. Descargar artefactos

El notebook crea un archivo ZIP desde:

```text
<DRIVE_ROOT>/runs/<RUN_ID>/
```

y lo deja en `/content/<RUN_ID>.zip` para descargarlo con `google.colab.files.download`. El ZIP contiene el manifiesto, los estados y logs de la ejecución. Los modelos, CSV, parámetros y otros artefactos de cada paquete permanecen en sus rutas `inputs/` y `outputs/` sincronizadas bajo el bucket.

Para conservar un resultado fuera de Colab:

1. Espera a que `status.json` indique `completed` y que `manifest.json` tenga estado `completed`.
2. Ejecuta la celda de exportación del notebook.
3. Descarga el ZIP generado.
4. Conserva también la estructura del paquete en `DRIVE_ROOT` si necesitarás reanudar u operar posteriormente.

## Resumen operativo

1. Selecciona GPU cuando corresponda.
2. Monta Drive.
3. Coloca el repositorio en `/content/mPES`.
4. Instala y valida dependencias.
5. Configura `DRIVE_ROOT` como bucket persistente y `REPO_ROOT` como disco local.
6. Elige una línea y un paquete registrado.
7. Ejecuta únicamente `optimize` o `train`.
8. Revisa la sincronización, `status.json` y `manifest.json`.
9. Descarga los artefactos cuando el trabajo haya terminado.
10. No ejecutes benchmarks desde este flujo.
