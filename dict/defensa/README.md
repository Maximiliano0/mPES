# Material para la defensa

Temas que Bernardez sugirió preparar («Recomendaciones adicionales»): la variabilidad entre semillas, el criterio
de elección de los modelos y la política aprendida frente a la regla $S + k$. Todo sale del Apéndice E de la tesis
y de `h1/general/results/ensemble/policy_vs_rule.json`; para regenerarlo, correr
`python writings/auxiliar/scripts/policy_vs_rule.py`.

| Archivo | Uso |
|---|---|
| `politica_vs_regla_slide.png` | Lámina 16:9 |
| `politica_vs_regla.png` | Misma figura al ancho de página (Figura E.1 de la tesis) |

## 1. Criterio de elección de los modelos (Tabla E.1, §4.4–§4.5)

Cada modelo incorpora una capacidad que otro del conjunto no tiene. El punto de partida fue DQN y los demás
salieron de la bibliografía. Ninguno de los siete modelos se excluyó después de evaluarlo.

| Modelo | Capacidad que incorpora | Referencia | Ensambles |
|---|---|---|---|
| Q-Learning base | Referencia sin optimizar | Watkins y Dayan (1992) | no |
| Q-Learning | Optimización de hiperparámetros | Bergstra et al. (2011) | no |
| Double Q-Learning | Menor sobreestimación | van Hasselt (2010) | no |
| DQN | Red que generaliza entre estados | Mnih et al. (2015) | los seis |
| DQN recurrente | Memoria por recurrencia | Hausknecht y Stone (2015) | los seis |
| Transformer | Memoria por atención causal | Parisotto et al. (2020) | los seis |
| A2C | Gradiente de política | Mnih et al. (2016) | cinco (no el ponderado) |

- Los tabulares quedan fuera de los ensambles porque son los de menor capacidad de generalización: no comparten
  información entre estados y obtienen las menores medias en generalización.
- A2C queda fuera del ponderado porque reducía su desempeño durante el ajuste *ad hoc*. Además, Optuna le asignó el
  menor peso base en 4 de los otros 5 ensambles.

## 2. Política aprendida frente a la regla $S + k$ (Figura E.1, Tabla E.2)

Asignación media en las decisiones con recursos de las 384 secuencias (referencia más 5 réplicas *held-out*).

| Política | $a-S$ ref. | $a-S$ *held-out* | $\bar r$ ref. | $\bar r$ *held-out* |
|---|---|---|---|---|
| Óptimo | +1,6 | +1,6 | 1,000 | 1,000 |
| Regla $a=\min(S+2,R)$ | +1,5 | +1,5 | 0,940 | 0,943 |
| Ensamble ponderado | +1,4 | +1,6 | 0,937 | 0,939 |
| Compuerta del Transformer | +2,6 | +2,2 | 0,928 | 0,928 |
| Transformer | +2,5 | +2,2 | 0,927 | 0,929 |
| Consenso con *prior* | +1,6 | +1,6 | 0,918 | 0,915 |
| Voto por acción | +1,1 | +1,1 | 0,914 | 0,916 |
| Voto suave | +1,0 | +1,2 | 0,914 | 0,917 |
| DQN recurrente | +0,5 | +0,7 | 0,899 | 0,903 |
| Double Q-Learning | +2,7 | +2,6 | 0,896 | 0,900 |
| DQN | +3,0 | +3,0 | 0,894 | 0,891 |
| Consenso | +3,0 | +2,8 | 0,889 | 0,893 |
| A2C | +3,0 | +3,1 | 0,887 | 0,887 |
| Q-Learning | +2,2 | +1,8 | 0,887 | 0,881 |
| Q-Learning base | +2,0 | +2,0 | 0,871 | 0,860 |

Mensajes para la lámina:

- La dinámica fija el equilibrio en $a = S$, pero no dice cuánto conviene superarlo; eso depende de la holgura del
  presupuesto. El óptimo lo supera en 1,6 unidades.
- La regla «$S+2$» asigna en la práctica $S+1{,}5$, porque la recortan el presupuesto y el máximo de 10 unidades.
- El ensamble ponderado reproduce la forma del óptimo; el Transformer asigna de más con severidades bajas, y DQN
  asigna unas 7,7 unidades casi sin importar $S$.
- El exceso medio no ordena por sí solo a los modelos: el consenso con *prior* asigna lo mismo que el óptimo y queda
  por debajo del Transformer. También importa cómo cambia la asignación con $S$ y con el presupuesto restante.
- Los modelos llegaron a esas políticas sólo a partir de la recompensa, sin conocer la regla ni el óptimo (§6.3).
- Ojo: aquí el Transformer da $S+2{,}2$ en sus propias trayectorias. El $S+2{,}4$ de la §6.2 se midió en los estados
  que visita el ensamble ponderado.

## 3. Variabilidad entre semillas (Tabla E.3, §7.3)

Los modelos entregados salen del mejor ensayo de Optuna, que fija también la semilla, así que no son un sorteo
insesgado. Se reentrenaron con la configuración entregada: Q-Learning y DQN con 5 semillas nuevas (1001–1005) y el
Transformer con 4 (1001, 1002, 1004, 1005). La evaluación es la misma, determinista, y reproduce los valores de los
entregados. Fuente: `h1/general/results/individual/seed_comparison.json`.

| Modelo | $n$ | Entregado ref. | Entregado *held-out* | Nuevas ref. | Rango | Nuevas *held-out* | Rango |
|---|---|---|---|---|---|---|---|
| Q-Learning | 5 | 0,887 | 0,881 | 0,859 ± 0,022 | 0,836–0,895 | 0,866 ± 0,016 | 0,842–0,886 |
| DQN | 5 | 0,894 | 0,891 | 0,874 ± 0,005 | 0,866–0,880 | 0,876 ± 0,005 | 0,868–0,881 |
| Transformer | 4 | 0,927 | 0,929 | 0,927 ± 0,009 | 0,914–0,933 | 0,927 ± 0,010 | 0,912–0,934 |

Mensajes para la lámina:

- La ventaja del Transformer no depende de la semilla: su peor semilla supera a la mejor de DQN en 0,035 (ref.) y
  0,032 (*held-out*); las medias difieren en 0,053 y 0,052.
- DQN y Q-Learning no se separan: sus rangos se superponen, la diferencia de medias (0,015 y 0,010) es menor que la
  desviación estándar de Q-Learning y la mejor semilla de Q-Learning (0,895 / 0,886) supera a las 5 de DQN.
- El sesgo de selección de Optuna se ve en DQN, cuyo entregado (0,894) supera a las 5 semillas nuevas, pero no en
  el Transformer: su entregado queda dentro de una desviación estándar de la media.
- El ensamble ponderado (0,937 y 0,939) supera a las 4 semillas del Transformer, pero su ventaja sobre la media
  (0,010 y 0,012) es del orden de la desviación estándar entre semillas y el ensamble no se reentrenó. Es coherente
  con el «efecto despreciable» de la tesis; no se puede generalizar.
- Alcance: sólo la referencia y las 5 réplicas (misma distribución); no los 21 escenarios de generalización. La
  semilla 1003 del Transformer se interrumpió por costo, no por su resultado (iba al ~15 % del entrenamiento).

Para preguntas (no está en la tesis):

- *¿Por qué media ± DE y no intervalos de confianza, si §7.3 cita a Agarwal et al.?* Con $n = 4$–$5$ un intervalo
  $t$ al 95 % es ancho; §7.3 lo deja como trabajo futuro. Calcularlo desde `seed_variability.json` si se pide.
- *¿Por qué el Transformer entregado no muestra sesgo de selección?* Su ensayo de Optuna dio $\bar r = 0{,}9245$
  (`h1/ml/pes_trf/inputs/best_params.json`), pero el modelo final, reentrenado con la semilla 42+i+1, dio 0,927: la
  selección eligió la configuración, y el reentrenamiento volvió a sortear la semilla. En Q-Learning y DQN el
  modelo final reproduce el puntaje del ensayo (0,8866 y 0,8937).
- *Indicio adicional del sesgo:* en Q-Learning, la distancia entre el entregado y la media de sus semillas es mayor
  en la referencia (0,027), donde seleccionó Optuna, que en *held-out* (0,016).

Costo en la notebook (CPU): 3 min por semilla de Q-Learning, 18 min de DQN y 3,3–7 h del Transformer, según la
limitación térmica. Falta la semilla 1003 del Transformer: desde `h1/`,
`python -m general.scripts.seed_variability run --models trf --seeds 1003`, luego `summary` y
`python writings/auxiliar/scripts/seed_comparison.py` desde la raíz. Los modelos por semilla están en
`h1/general/work/seeds/` (Git LFS).
