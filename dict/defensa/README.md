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

## 3. Variabilidad entre semillas

Pendiente (fase 2: reentrenamiento con varias semillas; ver `dict/correcciones.md`).
