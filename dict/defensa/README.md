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
insesgado. Se reentrenaron con la configuración entregada todos los modelos individuales salvo Q-Learning base, con
10 semillas nuevas cada uno (1001–1010). La evaluación es la misma, determinista, y reproduce los valores de los
entregados. Fuente: `h1/general/results/individual/seed_comparison.json` (`writings/auxiliar/scripts/seed_comparison.py`).

| Modelo | $n$ | Entregado ref. | Entregado *held-out* | Nuevas ref. | Rango | Nuevas *held-out* | Rango |
|---|---|---|---|---|---|---|---|
| Q-Learning | 10 | 0,887 | 0,881 | 0,865 ± 0,017 | 0,836–0,895 | 0,869 ± 0,016 | 0,842–0,889 |
| Double Q-Learning | 10 | 0,896 | 0,900 | 0,874 ± 0,008 | 0,855–0,881 | 0,876 ± 0,008 | 0,860–0,883 |
| DQN | 10 | 0,894 | 0,891 | 0,869 ± 0,009 | 0,858–0,880 | 0,871 ± 0,009 | 0,860–0,881 |
| DQN recurrente | 10 | 0,899 | 0,903 | 0,902 ± 0,020 | 0,864–0,921 | 0,898 ± 0,023 | 0,862–0,924 |
| A2C | 10 | 0,887 | 0,887 | 0,842 ± 0,035 | 0,786–0,887 | 0,842 ± 0,037 | 0,780–0,887 |
| Transformer | 10 | 0,927 | 0,929 | 0,919 ± 0,012 | 0,895–0,933 | 0,920 ± 0,011 | 0,901–0,934 |

Mensajes para la lámina:

- La ventaja del Transformer sobre DQN no depende de la semilla: su peor semilla supera a la mejor de DQN en 0,016
  (ref.) y 0,020 (*held-out*); las medias difieren en 0,050 y 0,049.
- **Frente al DQN recurrente, sí depende:** las medias difieren en 0,016 y 0,022 y los rangos se superponen (la mejor
  semilla del DQN recurrente, 0,921 / 0,924, supera a la peor del Transformer). El Transformer conserva la mayor media.
- DQN, Double Q-Learning y Q-Learning no se separan: las medias difieren en menos de 0,01 y la mejor semilla de
  Q-Learning (0,895 / 0,889) supera a las 10 de DQN.
- A2C es el más sensible a la semilla (DE 0,035 / 0,037) y su entregado iguala a la mejor de sus semillas.
- Tres semillas de DQN (1007, 1009 y 1010) convergieron a la política trivial «asignar el máximo disponible»,
  $\min(R, 10)$, en el 100 % de las decisiones: redes distintas, trayectorias idénticas, 0,858 y 0,860. Fijan el
  mínimo de DQN.
- El sesgo de selección de Optuna se ve en Double Q-Learning y DQN, cuyos entregados superan a sus 10 semillas
  nuevas, y en A2C; no en el Transformer (entregado dentro de una DE de la media) ni en el DQN recurrente (entregado
  por debajo de la media en la referencia).
- Alcance: sólo la referencia y las 5 réplicas (misma distribución); no los 21 escenarios de generalización.

### Ensambles con miembros de otras semillas (Tabla E.4)

Regla y parámetros de cada ensamble fijos; sus miembros (DQN, DQN recurrente, Transformer y A2C) se reemplazan por
modelos reentrenados. Diez combinaciones comunes a los seis ensambles: la mejor semilla de cada miembro, la peor, la
más cercana a su media y 7 sorteadas (criterio: $\bar r$ medio en la referencia y las réplicas). El ponderado no usa
A2C. Con los miembros entregados se reproducen exactamente las celdas del benchmark. Fuente:
`h1/general/results/ensemble/ensemble_seeds.json` (`h1/general/scripts/ensemble_seeds.py`).

| Ensamble | Entregado ref. | Entregado *held-out* | Combinaciones ref. | Rango | Combinaciones *held-out* | Rango | Superan a su Transformer |
|---|---|---|---|---|---|---|---|
| Ensamble ponderado | 0,937 | 0,939 | 0,928 ± 0,006 | 0,916–0,937 | 0,932 ± 0,004 | 0,927–0,940 | 9 / 9 |
| Compuerta del Transformer | 0,928 | 0,928 | 0,919 ± 0,015 | 0,894–0,934 | 0,921 ± 0,013 | 0,900–0,935 | 5 / 4 |
| Consenso con *prior* | 0,918 | 0,915 | 0,895 ± 0,022 | 0,854–0,924 | 0,892 ± 0,022 | 0,854–0,916 | 0 / 0 |
| Voto por acción | 0,914 | 0,916 | 0,909 ± 0,018 | 0,862–0,925 | 0,905 ± 0,017 | 0,868–0,928 | 2 / 0 |
| Voto suave | 0,914 | 0,917 | 0,908 ± 0,020 | 0,856–0,925 | 0,903 ± 0,019 | 0,866–0,927 | 1 / 0 |
| Consenso | 0,889 | 0,893 | 0,872 ± 0,015 | 0,853–0,896 | 0,870 ± 0,013 | 0,857–0,890 | 0 / 0 |

Mensajes para la lámina:

- El ensamble ponderado es el único que mejora al Transformer de forma estable: lo supera en 9 de 10 combinaciones
  (+0,009 / +0,010 en promedio) y su DE (0,006 / 0,004) es menos de la mitad de la del Transformer. Su ventaja es
  pequeña, pero no depende de la semilla de los miembros.
- La compuerta reproduce a su Transformer (diferencia media 0,000 / −0,001).
- Los votos y los consensos quedan por debajo de su Transformer en casi todas las combinaciones y varían más que él
  (DE 0,013–0,022).
- Los seis entregados quedan por encima de la media de sus combinaciones: sus parámetros se ajustaron con los
  miembros entregados.

Para preguntas (no está en la tesis):

- *¿Por qué media ± DE y no intervalos de confianza, si §7.3 cita a Agarwal et al.?* La tabla describe la
  dispersión entre semillas. Con $n = 10$, el intervalo $t$ al 95 % de la media (referencia / *held-out*) es
  Q-Learning 0,853–0,877 / 0,857–0,881, Double Q-Learning 0,868–0,880 / 0,870–0,881, DQN 0,862–0,875 / 0,864–0,877,
  DQN recurrente 0,888–0,916 / 0,881–0,914, A2C 0,817–0,867 / 0,815–0,868 y Transformer 0,910–0,927 / 0,912–0,928.
  Los del Transformer y DQN no se superponen; los del Transformer y el DQN recurrente, sí (por poco).
- *¿Por qué tres semillas de DQN dan exactamente el mismo valor?* Porque las tres redes, aunque tienen pesos
  distintos, eligen la acción máxima en todos los estados que visitan; recorren las mismas trayectorias. Es una
  política degenerada a la que DQN converge con algunas semillas (asignar todo hasta agotar el presupuesto), no un
  error de la evaluación.
- *¿Por qué el Transformer entregado no muestra sesgo de selección?* Su ensayo de Optuna dio $\bar r = 0{,}9245$
  (`h1/ml/pes_trf/inputs/best_params.json`), pero el modelo final, reentrenado con la semilla 42+i+1, dio 0,927: la
  selección eligió la configuración, y el reentrenamiento volvió a sortear la semilla. En Q-Learning y DQN el
  modelo final reproduce el puntaje del ensayo (0,8866 y 0,8937).
- *Indicio adicional del sesgo:* en Q-Learning, la distancia entre el entregado y la media de sus semillas es mayor
  en la referencia (0,022), donde seleccionó Optuna, que en *held-out* (0,012).

Costo en CPU por semilla: 1–3 min (tabulares), 18–25 min (DQN), ~30 min (A2C) y 3–7 h (DQN recurrente,
Transformer); los ensambles sólo se evalúan (6 × 11 en ~30 min). DQN, el DQN recurrente y el Transformer usan un solo
hilo (determinismo), así que las semillas se pueden correr en paralelo en procesos separados. Para regenerar los
resúmenes: desde `h1/`, `python -m general.scripts.seed_variability summary` y
`python -m general.scripts.ensemble_seeds summary`, y `python writings/auxiliar/scripts/seed_comparison.py` desde
la raíz. Los modelos por semilla están en `h1/general/work/seeds/` (Git LFS).
