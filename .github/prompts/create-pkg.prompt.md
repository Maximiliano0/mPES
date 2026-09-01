---
description: "Scaffold a new mPES package (pes_*) for testing a new model or architecture, mirroring the standard layout, conventions, and end-to-end pipeline used by existing packages."
argument-hint: "Provide the new package name (e.g. pes_ppo) and the algorithm/architecture to implement (e.g. PPO, SAC, RNN-DQN, Transformer-AC...)"
agent: "agent"
---

# Create New mPES Package — `<NEW_PKG>`

> Last updated: 2026-05-04

Generate a brand-new package inside the mPES workspace that follows the
established `pes_*` structure, conventions, and end-to-end pipeline. The
purpose is to enable rapid experimentation with **different RL models and
architectures** on the shared Pandemic Scenario task, while keeping every
package self-contained and comparable.

## Workspace layout

Packages are grouped by algorithm family under two top-level directories:

- `tabular/` — value-based tabular RL (`pes_base`, `pes_ql`, `pes_dql`).
- `ml/`      — deep / neural RL (`pes_dqn`, `pes_rdqn`, `pes_a2c`,
               `pes_trf`, `pes_ens`).

New packages **must** be placed inside the appropriate group based on the
algorithm family. Tabular methods go in `tabular/`; anything that uses
Keras/TensorFlow weights, deep nets, or ensembles thereof goes in `ml/`.

## Usage

When invoking this prompt, supply **two** arguments:

```text
@create-pkg <NEW_PKG> <ALGORITHM_OR_ARCHITECTURE>
```

Examples:

```text
@create-pkg pes_ppo Proximal Policy Optimization (clipped surrogate, GAE)  # ml/pes_ppo
@create-pkg pes_sac Soft Actor-Critic with discrete action space            # ml/pes_sac
@create-pkg pes_ddqn Dueling Double DQN                                     # ml/pes_ddqn
@create-pkg pes_tac Transformer-Actor-Critic                                # ml/pes_tac
@create-pkg pes_munch Munchausen DQN                                        # ml/pes_munch
@create-pkg pes_sarsa Tabular SARSA(λ)                                       # tabular/pes_sarsa
```

Throughout this prompt:
- `<NEW_PKG>` is the new package folder name (must start with `pes_`,
  lowercase, snake_case).
- `<GROUP>` is `tabular` or `ml`, inferred from the algorithm family.
- `<ALG>` is the algorithm / architecture short tag used in filenames
  (e.g. `ppo`, `sac`, `rdqn`, `tac`). Infer it from `<NEW_PKG>` (strip the
  `pes_` prefix) unless the user provides a different one.
- `<ALG_UPPER>` is the uppercase form used in CONFIG constants and log
  prefixes (e.g. `PPO`, `SAC`, `RDQN`, `TAC`).

If any argument (or the group) is ambiguous, ask the user before
scaffolding.

## Directive

Create a fully working **skeleton** of `<NEW_PKG>` that:

1. Mirrors the canonical `pes_*` package layout exactly.
2. Implements the requested algorithm/architecture in `ext/`.
3. Exposes the standard three-stage pipeline: **optimize → train → run**.
4. Reuses the shared support modules from the closest existing package
   verbatim (`src/exp_utils.py`, `src/log_utils.py`,
   `src/pygameMediator.py`, `src/result_formatter.py`,
   `src/terminal_utils.py`) — adapted only as needed for the new agent.
5. Passes the project quality gates (pyright 0 errors, pylint 10.00/10).
6. Does **not** modify any other package.

## Step 0 — Pick the closest reference package

Choose the existing package that most closely matches the requested
architecture; copy its scaffold as the starting point:

| Requested style | Closest reference | Group |
|----------------|-------------------|-------|
| Tabular / value-table methods | `tabular/pes_base` | `tabular` |
| Tabular + Bayesian optimisation | `tabular/pes_ql` | `tabular` |
| Tabular variants (Double Q, PBRS, ε-decay warm-up…) | `tabular/pes_dql` | `tabular` |
| Deep value-based (DQN family, replay buffer, target net) | `ml/pes_dqn` | `ml` |
| Policy-gradient / actor-critic (A2C, PPO, SAC, …) | `ml/pes_a2c` | `ml` |
| Sequence / attention models (Transformer, RNN over history) | `ml/pes_rdqn` (LSTM over history) or `ml/pes_trf` (causal Transformer) | `ml` |
| Ensemble / soft voting | `ml/pes_ens` | `ml` |

State your choice (and the resulting `<GROUP>`) explicitly before
generating files.

## Step 1 — Discover the reference

Before creating anything:

1. List the reference package: root, `config/`, `ext/`, `src/`, `doc/`
   (e.g. `ml/pes_a2c/...`).
2. Read `<GROUP>/<REF>/__init__.py`, `<GROUP>/<REF>/__main__.py`,
   `<GROUP>/<REF>/config/CONFIG.py`, and every file in
   `<GROUP>/<REF>/ext/` and `<GROUP>/<REF>/src/`.
3. Identify which constants in `CONFIG.py` are algorithm-specific
   (prefixed with the algorithm tag, e.g. `AC_*`, `DQN_*`) — these become
   the `<ALG_UPPER>_*` constants in the new package.

## Step 2 — Create the package skeleton

Create the directory tree below under `<GROUP>/`. Empty directories must
contain a `.gitkeep` file.

```
<GROUP>/<NEW_PKG>/
├── __init__.py            # Mirrors reference: config re-export, ANSI, numpy/TF setup
├── __main__.py            # Experiment entry point (blocks / sequences / trials)
├── config/
│   ├── __init__.py
│   └── CONFIG.py          # All tuneable constants, including <ALG_UPPER>_* hyperparams
├── doc/
│   ├── <NEW_PKG>_explained.md   # Implementation guide
│   └── <NEW_PKG>_theory.md      # Theoretical foundations
├── ext/
│   ├── __init__.py
│   ├── pandemic.py        # Gym environment (copy from reference; adapt only if state/reward changes)
│   ├── tools.py           # Shared helpers (copy from reference)
│   ├── <alg>_model.py     # NEW — model definition (Keras / numpy table)
│   ├── train_<alg>.py     # NEW — training loop, writes to inputs/<DATE>_<ALG_UPPER>_TRAIN/
│   └── optimize_<alg>.py  # NEW — Optuna Bayesian optimisation, writes to inputs/<DATE>_BAYESIAN_OPT/
├── inputs/
│   └── .gitkeep           # Generated artefacts land here at runtime
├── outputs/
│   └── .gitkeep           # Experiment logs/results land here at runtime
└── src/                   # Verbatim copy from the reference package
    ├── __init__.py
    ├── exp_utils.py
    ├── log_utils.py
    ├── pygameMediator.py
    ├── result_formatter.py
    └── terminal_utils.py
```

### Naming rules

- Filenames in `ext/` use the lowercase `<alg>` tag.
- CONFIG constants use the `<ALG_UPPER>_` prefix (e.g. `PPO_ACTOR_LR`,
  `SAC_TAU`, `RDQN_HIDDEN_UNITS`).
- Log file name pattern: `PES_<ALG_UPPER>_log_<DATE>_<ALG_UPPER>_AGENT.txt`.
- `PLAYER_TYPE` value used by `__main__.py` must be
  `'<ALG_UPPER>_AGENT'`.
- Date-stamped artefact folders:
  `inputs/<DATE>_<ALG_UPPER>_TRAIN/` and `inputs/<DATE>_BAYESIAN_OPT/`.
- Canonical model artefact: `inputs/<alg>_<role>.keras` (e.g.
  `inputs/ppo_actor.keras`) plus `inputs/rewards.npy`.
- Run command: `python -m <GROUP>.<NEW_PKG>` (e.g.
  `python -m ml.pes_ppo`).
- Module paths in imports use the group prefix
  (`from <GROUP>.<NEW_PKG>.ext.<alg>_model import ...`).

## Step 3 — Adapt the code

For every copied file, perform a **find-and-replace pass** so the new
package is fully self-referential:

1. Replace the reference package name with `<NEW_PKG>` everywhere
   (imports, log prefixes, docstrings, paths).
2. Rename algorithm-specific symbols and constants to the new
   `<ALG>` / `<ALG_UPPER>` tags.
3. Replace the model implementation (`<alg>_model.py`) with the new
   architecture. Keep the same public surface that
   `train_<alg>.py`, `optimize_<alg>.py`, and
   `src/pygameMediator.py` expect (a callable that returns an action
   given a state).
4. Update `train_<alg>.py` and `optimize_<alg>.py` to:
   - Read hyperparameters from `config/CONFIG.py` (single source of truth
     — never hard-code).
   - Write artefacts to the date-stamped folders described above.
   - Also overwrite the canonical `inputs/<alg>_<role>.keras` and
     `inputs/rewards.npy` after a successful training run.
5. Update `src/pygameMediator.py` to add a `provide_<alg>_agent_response()`
   function (or rename the equivalent one from the reference) that loads
   the model from the canonical path on every call.
6. Update `__main__.py` to dispatch to the new agent when
   `PLAYER_TYPE == '<ALG_UPPER>_AGENT'`.

## Step 4 — Configuration

In `<NEW_PKG>/config/CONFIG.py`:

- Keep all environment constants identical to the reference
  (`AVAILABLE_RESOURCES`, `MAX_ALLOC`, `PANDEMIC_PARAMETER`, block /
  sequence / trial counts, `SEED = 42`, etc.).
- Add a clearly-marked section for `<ALG_UPPER>_*` hyperparameters with
  sensible defaults for the new algorithm.
- Add `<ALG_UPPER>_OPTIMIZE_MODE` if the optimiser supports more than one
  search mode.
- Ensure `INPUTS_PATH`, `OUTPUTS_PATH`, `LOG_PATH`, and `PLAYER_TYPE` are
  defined consistently with the reference package.

## Step 5 — Documentation

In `<GROUP>/<NEW_PKG>/doc/`:

1. `<NEW_PKG>_explained.md` — implementation walkthrough mapped to the
   actual files in `ext/`. Include all `<ALG_UPPER>_*` hyperparameters
   and their meaning.
2. `<NEW_PKG>_theory.md` — theoretical foundations of the algorithm,
   APA-style references, and the mapping from theoretical objects to
   code symbols.

Write both documents in **Spanish**, using KaTeX-compatible math
(`$...$`, `$$...$$`).

Do **not** generate `.html` exports here — the existing
`@update-pkg-docs` prompt and `utils/scripts/_export_html.py` handle
that.

## Step 6 — Project-level updates

After scaffolding, update the following workspace-level files:

1. `README.md` — add `<GROUP>/<NEW_PKG>` to the package table with its
   algorithm description.
2. `.github/copilot-instructions.md` — add `<GROUP>/<NEW_PKG>` to the
   package table.
3. `.github/prompts/pkg-scope.prompt.md`,
   `.github/prompts/audit-project.prompt.md`, and
   `.github/prompts/update-pkg-docs.prompt.md` — add `<NEW_PKG>` to
   their "Available Packages" / usage examples.
4. `utils/scripts/_export_html.py` — add an entry
   `"<NEW_PKG>": "<GROUP>"` to the `PACKAGE_GROUPS` dictionary so the
   exporter discovers the new package.
5. `utils/config/requirements.txt` — append any new third-party
   dependency the algorithm needs (and call it out to the user).

## Step 7 — Validate

Before reporting completion:

1. Run pyright against `<GROUP>/<NEW_PKG>/` — must report **0 errors**.
2. Run pylint against `<GROUP>/<NEW_PKG>/` — must score **10.00/10**.
3. Confirm the package tree matches Step 2 exactly.
4. Confirm `python -m <GROUP>.<NEW_PKG>` would resolve all imports
   (static check — do **not** execute training).

## Conventions reminder

- Follow every rule from `.github/copilot-instructions.md` (style,
  imports, docstrings, NumPy alias `numpy`, max 120 cols, snake_case +
  PascalCase, NumPy-style docstrings on every public function/class).
- Explicit imports only — no wildcard imports outside `__init__.py`.
- Use `os.path.join()` for every path; never hard-code `/` or `\`.
- Do **not** cross-reference or import from other `pes_*` packages.
- Do **not** commit, push, or run training/optimisation. Scaffold only.
