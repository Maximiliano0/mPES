# pes_ens — ensamble ponderado (voto suave con *prior* de severidad)

`pes_ens` es un ensamble de sólo inferencia sobre los modelos ya entrenados
de `ml/pes_dqn`, `ml/pes_rdqn` y `ml/pes_trf`. Forma parte del benchmark
vigente (`general/`) y es el ensamble con mejor desempeño: $\bar r = 0{,}937$
en la referencia y $0{,}939$ de media en los 21 escenarios de
generalización.

## Uso

Desde `h1/`, con `win_mpes_env` activado desde la raíz del repositorio:

```powershell
python -m ens.pes_ens
```

No tiene optimización bayesiana: todos sus parámetros son constantes de
[`config/CONFIG.py`](../config/CONFIG.py).

## Miembros y parámetros

| Miembro | Rol | Archivo | Peso base | Peso normalizado |
|---------|-----|---------|-----------|------------------|
| DQN | `q_dense` | `ml/pes_dqn/inputs/dqn_model.keras` | 0,18 | 0,030 |
| A2C | `actor` | `ml/pes_a2c/inputs/ac_actor.keras` | 1,0 | deshabilitado |
| DQN recurrente | `q_recurrent` (ventana 6) | `ml/pes_rdqn/inputs/rdqn_model.keras` | 0,9 | 0,148 |
| Transformer | `q_recurrent` (ventana 6) | `ml/pes_trf/inputs/trf_model.keras` | 5,0 | 0,822 |

| Constante | Valor |
|-----------|-------|
| `ENS_SOFTMAX_TEMPERATURE` | 15,0 |
| `ENS_SEVERITY_PRIOR_WEIGHT` | 0,17 |
| `ENS_SEVERITY_PRIOR_SIGMA` | 3,0 |

## Regla de decisión

Implementada en `EnsembleAgent.predict()`
([`ext/ensemble_model.py`](../ext/ensemble_model.py)):

1. Cada miembro Q produce $\mathrm{softmax}(Q/15)$ sobre las 11 acciones.
2. Se anulan las acciones no factibles ($a > R$) y se renormaliza.
3. Cada miembro pesa $w_k^{\mathrm{norm}}\,(0{,}1 + c_k)$, con
   $c_k = 1 - H(p_k)/\log_2 11$.
4. Si quedan recursos, la probabilidad de $a = 0$ se multiplica por 0,3.
5. La distribución se mezcla con un *prior* gaussiano centrado en la
   severidad actual: $(1 - 0{,}17)\,p + 0{,}17\,\pi_{\mathrm{prior}}$,
   $\sigma = 3$.
6. Cota de seguridad: si $S \ge 6$ y la acción elegida es menor que
   $\lfloor S/2 \rfloor$ (y esa cota es factible), se asigna
   $\lfloor S/2 \rfloor$.

[`src/pygameMediator.py`](../src/pygameMediator.py) construye el estado
recortando la severidad a `MAX_SEVERITY` (9) antes de normalizarla, de modo
que en severidades mayores que 9 tanto los miembros como el *prior* ven
$S = 9$.

## Métrica

El desempeño por secuencia es
$\bar r = (S_{\mathrm{peor}} - S_{\mathrm{agente}}) / (S_{\mathrm{peor}} - S_{\mathrm{mejor}})$,
donde $S_{\mathrm{mejor}}$ es la severidad mínima alcanzable con el
presupuesto de la secuencia, calculada de forma exacta por programación
dinámica en [`src/exp_utils.py`](../src/exp_utils.py).

## Origen de los parámetros y justificación a posteriori

Los parámetros no salen de una búsqueda sistemática: entraron ya fijos, a
mano, en el commit `d2f0c49` (2026-05-02). La documentación de ese commit
justifica la cota por los mínimos de ~0,75 de los primeros bloques de la
referencia y la atenuación de $a = 0$ por el registro de respuestas; para
$w_\pi = 0{,}17$, $\sigma = 3$, los pesos $0{,}18/0{,}9/5{,}0$ y el término
$0{,}1$ no hay justificación. Se analizaron después, sin modificarlos, con
tres scripts de `writings/auxiliar/scripts/` (resultados en
`h1/general/results/ensemble/`):

| Script | Qué hace | Resultado |
|--------|----------|-----------|
| `weighted_ens_sensitivity.py` | Varía un parámetro por vez y evalúa en `sev_base` y `heldout_s1..s5` | `weighted_ens_sensitivity.json` |
| `weighted_ens_oracle.py` | Reconstruye la asignación óptima (DP con reconstrucción y empates) y mide el efecto de cada regla en los estados que visita el ensamble | `weighted_ens_oracle.json` |
| `fixed_rule.py` | Evalúa la regla sin modelo $a = \min(S + k, R)$ en los 27 escenarios | `fixed_rule.json` |

Resultados principales (réplicas = media de las 5 réplicas fuera de muestra):

- **Sensibilidad.** $\tau$, $w_\pi$, $\sigma$ y el peso del Transformer tienen
  su máximo en el valor elegido, en `sev_base` y en las réplicas, pero son
  máximos angostos (un paso de la grilla cuesta 0,002–0,008). $\eta$, el
  término $0{,}1$ y la cota están en meseta. Sin *prior* el ensamble supera
  al Transformer en las réplicas por 0,004; con el *prior* y $\tau = 1$, por
  0,001; con ambos, por 0,010.
- **Óptimo.** Con presupuesto disponible, el óptimo nunca asigna 0 y asigna
  entre $S + 0{,}7$ y $S + 1{,}9$. Una gaussiana centrada en $S$ ajusta con
  $\sigma = 2{,}7$; con el centro libre, en $S + 2{,}2$ y $\sigma = 1{,}4$. La
  cota cambia 2 de 1 726 decisiones; el *prior* cambia el 37 % y mejora
  $\bar r$ en 0,012 por secuencia.
- **Regla fija.** $a = \min(S + 2, R)$ obtiene 0,940 en la referencia, 0,937
  en generalización y 0,943 en las réplicas: iguala al ensamble (0,937 /
  0,939 / 0,939) y supera al Transformer (0,927 / 0,930 / 0,929). Con
  $k = 0$ obtiene 0,782 / 0,805 / 0,787. Con $k = 2$ queda por encima de 12
  de los 13 modelos del benchmark (tabla completa en
  `comparacion_modelos.md`); los 13 la superan sólo con secuencias cortas.
  Con $a = S + k$ constante, $S_n = \max(0, S + k(1 - 1{,}4^n))$: la
  dinámica fija el equilibrio $a = S$, y el desplazamiento 2 se eligió a la
  vista del óptimo con información completa, por lo que la regla no es un
  competidor en igualdad de condiciones; indica que la política óptima del
  entorno es simple y explica por qué el *prior* ayuda.

Ver la base teórica en [pes_ens_theory.md](pes_ens_theory.md) y la
comparación completa en
[`../../../general/doc/comparacion_modelos.md`](../../../general/doc/comparacion_modelos.md).
