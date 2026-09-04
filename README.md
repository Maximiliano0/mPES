<div align="center">

# 🦠 mPES — *Multiple Pandemic Experiment Suite*

**A reinforcement-learning benchmark for resource allocation under uncertainty.**

[![Python](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-ff6f00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-4.7-1a73e8.svg)](https://optuna.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2-7c4dff.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/license-private-lightgrey.svg)](#-license)

</div>

> This project documentation reflects the active branch: `h1/`. Legacy and suspended material is intentionally excluded from the current workflow.

---

## ✨ Current scope

- 🧠 **Active benchmark line:** `h1/` with tabular, neural and ensemble training packages.
- 📊 **Under Stress Experiments:** [`h1/general/`](h1/general/) over six active benchmark models.
- 🔬 **Bayesian optimisation:** Optuna for the trainable variants.
- 🌍 **Windows-first workflow:** Python 3.12 and the `win_mpes_env` environment.
- 📚 **Package-level docs:** focused on the active models and their executable workflows.

---

## 📦 Active packages

### `h1/tabular/`

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_base` | Tabular Q-Learning baseline | [`ext/pandemic.py`](h1/tabular/pes_base/ext/pandemic.py), [`ext/train_rl.py`](h1/tabular/pes_base/ext/train_rl.py) |
| `pes_ql` | Q-Learning + Bayesian optimisation | [`ext/optimize_rl.py`](h1/tabular/pes_ql/ext/optimize_rl.py) |
| `pes_dql` | Double Q-Learning + ε-decay warm-up + PBRS | [`ext/pandemic.py`](h1/tabular/pes_dql/ext/pandemic.py), [`ext/optimize_rl.py`](h1/tabular/pes_dql/ext/optimize_rl.py) |

### `h1/ml/`

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_dqn` | Deep Q-Network | [`ext/dqn_model.py`](h1/ml/pes_dqn/ext/dqn_model.py), [`ext/train_dqn.py`](h1/ml/pes_dqn/ext/train_dqn.py), [`ext/optimize_dqn.py`](h1/ml/pes_dqn/ext/optimize_dqn.py) |
| `pes_rdqn` | Recurrent DQN (LSTM) | [`ext/rdqn_model.py`](h1/ml/pes_rdqn/ext/rdqn_model.py), [`ext/train_rdqn.py`](h1/ml/pes_rdqn/ext/train_rdqn.py), [`ext/optimize_rdqn.py`](h1/ml/pes_rdqn/ext/optimize_rdqn.py) |
| `pes_a2c` | Advantage Actor-Critic | [`ext/ac_model.py`](h1/ml/pes_a2c/ext/ac_model.py), [`ext/train_a2c.py`](h1/ml/pes_a2c/ext/train_a2c.py), [`ext/optimize_a2c.py`](h1/ml/pes_a2c/ext/optimize_a2c.py) |
| `pes_trf` | Causal Transformer DQN | [`ext/transformer_model.py`](h1/ml/pes_trf/ext/transformer_model.py), [`ext/train_transformer.py`](h1/ml/pes_trf/ext/train_transformer.py), [`ext/optimize_tr.py`](h1/ml/pes_trf/ext/optimize_tr.py) |

### `h1/ens/`

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_ens_sprb` | Confidence-weighted soft voting ensemble | [`__main__.py`](h1/ens/pes_ens_sprb/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_sprb/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_sprb/ext/optimize_ens.py) |
| `pes_ens_accq` | Confidence-weighted action/Q-value ensemble | [`__main__.py`](h1/ens/pes_ens_accq/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_accq/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_accq/ext/optimize_ens.py) |
| `pes_ens_trf_guard` | Transformer-first confidence-gated ensemble | [`__main__.py`](h1/ens/pes_ens_trf_guard/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_trf_guard/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_trf_guard/ext/optimize_ens.py) |
| `pes_ens_consensus` | Agreement/disagreement confidence consensus | [`__main__.py`](h1/ens/pes_ens_consensus/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_consensus/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_consensus/ext/optimize_ens.py) |

### Support directories

| Path | Purpose |
|------|---------|
| [`h1/general/`](h1/general/) | Under Stress Experiments harness |
| [`utils/`](utils/) | Windows scripts, requirements, lint config |

---

## ▶️ Usage

> Run all commands from inside `h1/`.

```bash
cd h1

python -m tabular.pes_base
python -m tabular.pes_ql
python -m tabular.pes_dql

python -m ml.pes_dqn
python -m ml.pes_rdqn
python -m ml.pes_a2c
python -m ml.pes_trf

python -m ens.pes_ens_sprb
python -m ens.pes_ens_accq
python -m ens.pes_ens_trf_guard
python -m ens.pes_ens_consensus
```

### Bayesian optimisation

```bash
cd h1
python -m tabular.pes_ql.ext.optimize_rl 50
python -m tabular.pes_dql.ext.optimize_rl 50
python -m ml.pes_dqn.ext.optimize_dqn 50
python -m ml.pes_rdqn.ext.optimize_rdqn 50
python -m ml.pes_a2c.ext.optimize_a2c 50
python -m ml.pes_trf.ext.optimize_tr 50
python -m ens.pes_ens_sprb.ext.optimize_ens 50
python -m ens.pes_ens_accq.ext.optimize_ens 50
python -m ens.pes_ens_trf_guard.ext.optimize_ens 50
python -m ens.pes_ens_consensus.ext.optimize_ens 50
```

---

## 📚 Documentation

The active repository documentation is centred on the executable packages in `h1/`. The general benchmark harness is the authoritative comparison point for the current six-model stress catalogue; historical thesis-style write-ups are retained only as archive material and are not part of the active workflow.

| Package | Guide |
|---------|-------|
| `pes_base` | [`explained_pes.md`](h1/tabular/pes_base/doc/explained_pes.md) · [`how_to_train_and_test.md`](h1/tabular/pes_base/doc/how_to_train_and_test.md) |
| `pes_ql` | [`pes_ql_explained.md`](h1/tabular/pes_ql/doc/pes_ql_explained.md) |
| `pes_dql` | [`pes_dql_explained.md`](h1/tabular/pes_dql/doc/pes_dql_explained.md) |
| `pes_dqn` | [`pes_dqn_explained.md`](h1/ml/pes_dqn/doc/pes_dqn_explained.md) |
| `pes_rdqn` | [`pes_rdqn_explained.md`](h1/ml/pes_rdqn/doc/pes_rdqn_explained.md) |
| `pes_a2c` | [`pes_a2c_explained.md`](h1/ml/pes_a2c/doc/pes_a2c_explained.md) |
| `pes_trf` | [`pes_trf_explained.md`](h1/ml/pes_trf/doc/pes_trf_explained.md) |
| ensemble packages | [`h1/general/README.md`](h1/general/README.md) |

---

## 📈 Current stress benchmark

The active benchmark in [`h1/general/`](h1/general/) evaluates six models:
`pes_ql`, `pes_dql`, `pes_dqn`, `pes_rdqn`, `pes_a2c` and `pes_trf`.

```bash
cd h1
python -m general.scripts.orchestrate
python -m general.scripts.aggregate
python -m general.scripts.plot_matrix
python -m general.scripts.report
```

---

## 📄 License

Private repository — all rights reserved.
