# Tareas pendientes de la tesis (v7, 24/09/2026)

**Criterio:** la verdad es el código Python de `h1/` y los resultados en JSON/CSV/TXT; la tesis, los `.md` y los comentarios se ajustaron a ellos.

## Resueltas en v6.1 (se mantienen)

1. **Referencia de Parisi.** Parisi et al. (2021) se cita en la Sección 1.2, junto a Kuhl (2021).
2. **Figuras `PES_*` en inglés:** se dejan como están.
3. **Leyenda de `ind_05`:** se deja como está.
6. **Tablas viejas de los .md:** en v7 se actualizaron a los valores del benchmark.
7. **`pes_ens`:** no está archivado; se quitaron los avisos de archivo de sus `doc/`.

## Corregidas en v7 porque el código las contradice

4. **Transformer (Tabla 4.3)** y 5. **DQN recurrente:** las cifras de la tesis coinciden con los modelos `.keras` desplegados (27 019 y 34 027 parámetros).
8. **Escenarios en que el Transformer tiene la mayor media:** con precisión completa son 16 de 22. Con dos decimales, dos celdas aparecen empatadas en la figura.
9. **DQN:** el modelo desplegado se entrenó con 40 000 episodios (`training_config_2026-04-30.txt`). `CONFIG.py` tenía 175 000 por defecto; ahora usa 40 000 (y 30 000 en DQN recurrente y Transformer).
10. **Arquitecturas:** la búsqueda incluía la ventana y la arquitectura, pero `train_transformer.py` y `train_rdqn.py` siempre toman la arquitectura de `CONFIG.py`. Del mejor ensayo sólo se toman los hiperparámetros de entrenamiento y la cabeza densa. Se verificó contra la base Optuna y los `.keras`: los ensayos #14 (DQN recurrente, LSTM 32) y #2 (Transformer, W3, d16, 8 cabezas, 4 bloques) no son la arquitectura desplegada. Los `CONFIG.py` de DQN, DQN recurrente y Transformer se alinearon con `best_params.json` (cabeza, episodios, hiperparámetros de entrenamiento). Construir la red desde `CONFIG.py` da exactamente las formas de pesos de los `.keras` (5 131, 34 027 y 27 019 parámetros), así que no hace falta reentrenar. Reentrenar sólo haría falta para usar la arquitectura del mejor ensayo.

## Resueltas en v7

11. **Ensayos de Double Q-Learning.** El estudio del 2026-04-21 tuvo 100 ensayos (14 completos y 86 podados). El mejor es el #3 (semilla 46). La Tabla 4.2 tenía los valores heredados de `_DEFAULT_HYPERPARAMS` y se reemplazaron por los de `best_params.json`: α 0,1132; γ 0,9777; ε 0,5998/0,0327; w 0,0260; q 0,6430; κ 0,000392; 360 000 episodios.
12. **Parámetros de los ensambles (Tabla 4.6).** Coinciden con los `best_params.json` de cada paquete. Cuatro de esos archivos no incluyen el peso de A2C, que vale 0,10 por defecto, y su `mean_perf` no coincide con la media del benchmark. Sólo el consenso con prior reproduce su referencia (0,918). Se documentó en Materiales 4.5 y en Limitaciones.
13. **Frecuencias.** Se midieron con `auxiliar/scripts/ensemble_decisions.py`, que reproduce las medias del benchmark con una diferencia máxima de 0,002. Se agregaron a Resultados 5.2 y a Discusión 6.2.
14. **`ds004477`.** Papastylianou, Ramele, Citi, Cinel y Poli (2023), *PES – Pandemic Emergency Scenario*, OpenNeuro, doi:10.18112/openneuro.ds004477.v1.0.2.
15. **Bibliografía verificada en la web:**
    - Shannon (1948): se agregó el doi.
    - Schulman et al. (2016): ICLR.
    - Wiering y van Hasselt (2008): 38(4):930–936; se agregó el doi.
    - Parisi et al. (2021): *PLOS Comput. Biol.* 17(7):e1009090; se completaron los autores y el doi.
    - `BCINE2022`: pasó a 2023, el año de creación del repositorio.

## Otras correcciones de fondo

- **Métrica:** $S_{\mathrm{mejor}}$ es el óptimo exacto por programación dinámica (`_best_feasible_sequence_severity`), no una regla voraz. $\bar r = 1$ significa asignación óptima. Se eliminó la limitación sobre la "referencia voraz".
- **Datos de entrenamiento:** las severidades del entrenamiento van de 2 a 8. Las filas $S = 0, 1, 9$ de las tablas Q quedaron con su inicialización aleatoria, lo que explica la caída de los métodos tabulares con severidad 10–12.
- **Severidad recortada:** A2C y el ensamble ponderado recortan la severidad a 9 antes de escalarla.
- **Escenarios fuera de rango:** se definen por las cotas del entorno ($S \le 9$, $T \le 10$). Otros 11 escenarios también incluyen severidades 0, 1 o 9, que no aparecen en el entrenamiento.
- **Apéndice A:** los comandos se ejecutan desde `h1/`. Se agregaron las columnas de entrenamiento y de evaluación. Los comandos de entrenamiento ya no necesitan el número de episodios.

## Siguen pendientes

- Nada bloqueante para la presentación.
- Opcional: reentrenar el DQN recurrente y el Transformer con la arquitectura del mejor ensayo, o fijar esas arquitecturas en `CONFIG.py`, si se quiere que el modelo final coincida con la búsqueda.
