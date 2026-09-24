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

Ver la base teórica en [pes_ens_theory.md](pes_ens_theory.md) y la
comparación completa en
[`../../../general/doc/comparacion_modelos.md`](../../../general/doc/comparacion_modelos.md).
