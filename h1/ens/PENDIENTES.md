# Pendientes de los ensembles

## Alcance

Los ensembles deben combinar únicamente los tres modelos entrenados:

- `pes_dqn`
- `pes_rdqn`
- `pes_trf`

Ambos deben utilizar los mismos archivos de entrada que esos modelos:

- `initial_severity.csv`
- `sequence_lengths.csv`

Las copias de ambos CSV son idénticas en DQN, RDQN y TRF.

## Estado resuelto

- `pes_ens_sprb` implementa votación blanda ponderada por confianza.
- `pes_ens_accq` implementa votación de acciones con desempate mediante valores Q normalizados.
- Ambos ensembles cargan los modelos DQN, RDQN y TRF.
- Los optimizadores ya no utilizan `target_actions`.
- El criterio de ambos optimizadores es el mismo que en DQN, RDQN y TRF:

  ```python
  mean_perf = float(numpy.mean(perfs))
  ```

- Pyright y Pylint pasan en los dos paquetes de ensembles.

## Pendientes prioritarios

### 1. Ejecutar la optimización de los ensembles

Ejecutar Optuna usando los CSV compartidos:

```powershell
cd h1
python -m ens.pes_ens_sprb.ext.optimize_ens 50
python -m ens.pes_ens_accq.ext.optimize_ens 50
```

Los optimizadores aceptan también `--severity`, `--lengths` y `--output` para indicar rutas alternativas.

Resultados esperados:

- `h1/ens/pes_ens_sprb/inputs/best_params.json`
- `h1/ens/pes_ens_accq/inputs/best_params.json`

Parámetros optimizados:

- SPRB: pesos de DQN, RDQN y TRF, temperatura y potencia de confianza.
- accQ: pesos de DQN, RDQN y TRF y potencia de confianza.

### 2. Verificar los resultados de Optuna

Comprobar que:

- cada optimización termina sin errores;
- `best_params.json` contiene `hyperparameters`, `value`, `mean_perf` y `std_perf`;
- los pesos son no negativos;
- `mean_perf` está calculado sobre las mismas secuencias fijas;
- se conserva la reproducibilidad con la semilla `42`.

### 3. Integrar los ensembles en el benchmark

Añadir `pes_ens_sprb` y `pes_ens_accq` al registro de modelos de:

- `h1/general/scripts/runner.py`
- `h1/general/scripts/orchestrate.py`

También se debe adaptar el runner al flujo de evaluación de los ensembles, ya que sus entry points reciben estados y no ejecutan directamente el flujo estándar de escenarios.

### 4. Generar los ensayos en `general/work`

Ejecutar los dos ensembles sobre los 22 escenarios existentes.

Resultados esperados:

- 22 escenarios para `pes_ens_sprb`;
- 22 escenarios para `pes_ens_accq`;
- 44 JSON adicionales en `h1/general/results/raw/`;
- directorios de trabajo y logs en `h1/general/work/`.

El benchmark completo quedaría en:

```text
8 modelos x 22 escenarios = 176 celdas
```

### 5. Validar la integridad del benchmark

Verificar para los 176 resultados:

- `returncode == 0`;
- 22 escenarios por modelo;
- 64 secuencias por escenario;
- acciones dentro de `[0, 10]`;
- ausencia de resultados incompletos;
- métricas calculadas con la misma referencia baseline.

### 6. Ejecutar el análisis agregado

Después de generar los resultados de los ensembles:

```powershell
cd h1
python -m general.scripts.aggregate
python -m general.scripts.plot_matrix
python -m general.scripts.report
```

Se deben generar:

- matrices CSV de rendimiento y métricas estadísticas;
- `matrix_summary.json`;
- heatmaps PNG/PDF;
- histogramas por escenario;
- `benchmark_report.md`.

### 7. Actualizar la documentación comparativa

Actualizar `h1/general/doc/comparacion_modelos.md` para incluir:

- los dos ensembles;
- sus pesos optimizados;
- sus resultados por escenario;
- la comparación frente a DQN, RDQN y TRF;
- las limitaciones metodológicas y el criterio `mean_perf`.

También falta crear documentación específica para cada ensemble si se desea mantener la misma estructura que los paquetes ML.

### 8. Añadir pruebas automatizadas

Crear pruebas para:

- carga y validación de los tres modelos;
- máscaras de acciones factibles;
- votación blanda de SPRB;
- desempates y normalización de accQ;
- mantenimiento y reinicio del historial recurrente;
- generación de escenarios;
- integridad de los JSON agregados.

## Orden de ejecución recomendado

1. Ejecutar y verificar Optuna para SPRB y accQ.
2. Integrar ambos ensembles en el runner del benchmark.
3. Generar los 44 ensayos faltantes.
4. Validar los 176 resultados.
5. Ejecutar agregación, gráficos y reporte.
6. Actualizar documentación y añadir pruebas.

## Nota metodológica

Los ensembles no necesitan `target_actions`. Su optimización debe seguir el mismo criterio que las optimizaciones de DQN, RDQN y TRF: ejecutar el entorno sobre las secuencias derivadas de `initial_severity.csv` y `sequence_lengths.csv` y maximizar el rendimiento medio normalizado por secuencia.
