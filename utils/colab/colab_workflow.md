# Bayesian-Optimisation on Colab Pro+ — Workflow Guide

> Last updated: 2026-05-04

## TL;DR

1. Open `utils/colab/launch.ipynb` in **Colab Pro+** *in a browser*
   ([colab.research.google.com](https://colab.research.google.com)) — one tab
   per package. **Background Execution is not exposed by any VS Code
   Colab extension; you must use the browser UI.**
2. Edit Cell 1: set `PKG`, `N_TRIALS`, `RESUME_DATE` (optional), `GIT_REPO`.
3. **Runtime → Run all**.
4. **Runtime → Manage sessions → Background execution** (Pro+ only).
5. Close the browser. Tail the log file from Drive whenever you want
   to check progress.

---

## Worked example: optimising `pes_ql`

This shows the two ways to run a Bayesian optimisation of the Q-Learning
package (`pes_ql`). Pick **one**.

### Option A — Locally (Windows or Linux)

```powershell
# Windows (PowerShell)
win_mpes_env\Scripts\Activate.ps1
$env:PYTHONIOENCODING = 'utf-8'
python -m tabular.pes_ql.ext.optimize_rl 100
```

```bash
# Linux
source linux_mpes_env/bin/activate
export PYTHONIOENCODING=utf-8
python -m tabular.pes_ql.ext.optimize_rl 100
```

Artifacts land in `tabular/pes_ql/inputs/<YYYY-MM-DD>_BAYESIAN_OPT/`:

- `optuna_study_<date>.db`  — Optuna SQLite storage (resumable)
- `trials.csv`              — per-trial hyperparameters + value
- `best_params.json`        — best hyperparameters found
- `study_plots/`            — history / importances / parallel-coord plots

**Resume a previous run** (keeps the existing DB, adds trials):

```bash
python -m tabular.pes_ql.ext.optimize_rl 100 --resume 2026-04-20
```

The wrapper scripts [`utils/win/run_bayesian_opt.ps1`](../win/run_bayesian_opt.ps1)
and [`utils/linux/run_bayesian_opt.sh`](../linux/run_bayesian_opt.sh) do the
same with env-var hygiene and alias resolution (`ql`, `dql`, `dqn`, `ac`).

### Option B — Headless on Colab Pro+

Use this when you want the optimisation to run for **many hours without
your PC being on**.

1. Open [`utils/colab/launch.ipynb`](launch.ipynb) in Colab.
2. Edit **Cell 1**:

   ```python
   PKG         = 'ql'                                    # <- pes_ql
   N_TRIALS    = 100
   RESUME_DATE = ''                                      # '' = fresh study
   GIT_REPO    = 'https://github.com/<user>/Win_mPES.git'
   GIT_BRANCH  = 'main'
   ```

3. **Runtime → Run all** (authorises Drive in Cell 2 the first time).
4. Click the ▾ next to **Connect** (top-right) → **Manage sessions** →
   toggle **Background execution** on for the active session.
   *(This menu lives in the Colab web UI only — VS Code's Colab
   extensions do not implement it.)*
5. Close the browser. The process keeps running.

Artifacts land in Google Drive at:

```
/content/drive/MyDrive/mPES/pes_ql/<YYYY-MM-DD>_BAYESIAN_OPT/
    run_meta.json              ← launch metadata (pkg, module, git sha,
                                  pid, n_trials, launch_ts, host, …)
    optuna_study_<date>.db
    bayesian_opt.log           ← starts with a structured banner echoing
                                  the same metadata, then per-trial lines
    bayesian_opt_err.log
    optimize.pid
    trials.csv
    best_params.json
    study_plots/
```

**Resume after a 24h VM reclaim**: re-open the notebook, set
`RESUME_DATE = '2026-04-20'` (the date stamped in the Drive folder name),
run all. Optuna re-opens the DB with `load_if_exists=True` and continues
from trial *N+1*.

### What the CLI flags do

`tabular.pes_ql.ext.optimize_rl` (and the other five `optimize_*.py` in
`tabular/` and `ml/`) accept:

```
python -m tabular.pes_ql.ext.optimize_rl [N_TRIALS] \
    [--resume YYYY-MM-DD]                   # re-open existing study
    [--out-dir   PATH]                      # redirect all artifacts
    [--storage   sqlite:///ABSOLUTE.db]     # redirect Optuna DB only
```

Cell 4 of the notebook invokes the scripts as:

```
python -m tabular.pes_ql.ext.optimize_rl 100 \
    --out-dir /content/drive/MyDrive/mPES/pes_ql/2026-04-20_BAYESIAN_OPT \
    --storage sqlite:////content/drive/MyDrive/mPES/pes_ql/2026-04-20_BAYESIAN_OPT/optuna_study_2026-04-20.db
```

The `or`-fallback on each flag means **local runs without flags behave
exactly as before** — these flags exist purely to let Colab redirect to
Drive.

---

## What this gives you

- **Headless 24/7 optimisation** on Colab Pro+ — your PC can be off.
- **Resumable studies**: each Optuna SQLite DB lives on Google Drive, so a
  Colab VM reclaim never loses progress.
- **Up to 4 parallel instances** (one per package: `pes_ql`, `pes_dql`,
  `pes_dqn`, `pes_rdqn`, `pes_a2c`, `pes_trf`), each with its own DB on Drive — no contention.

---

## One-time setup

### 1. Push the repo to GitHub

The Colab launcher clones the repo on every launch. Make sure your
working tree is pushed. Then edit `GIT_REPO` in Cell 1 of
`launch.ipynb` to point at your fork.

### 2. Open in Colab

- Push `utils/colab/launch.ipynb` to GitHub, then open it via
  `https://colab.research.google.com/github/<user>/Win_mPES/blob/main/utils/colab/launch.ipynb`,
  **or**
- Upload `launch.ipynb` directly to Colab.

---

## Running 4 packages in parallel

Open `launch.ipynb` in **four separate Colab tabs**. In each tab:

| Tab | Cell 1 → `PKG` | Suggested `N_TRIALS` |
|-----|---------------|---------------------|
| 1   | `'ql'`        | 150                 |
| 2   | `'dql'`       | 100                 |
| 3   | `'dqn'`       | 60                  |
| 4   | `'ac'`        | 100                 |

Each tab runs in an independent Colab VM, writes to a separate path on
Drive (`/content/drive/MyDrive/mPES/<pkg>/<date>_BAYESIAN_OPT/`), and uses
its own SQLite database. They never collide.

> **Note:** Colab Pro+ enforces a soft global limit on simultaneous active
> sessions. If a fourth tab fails to start, queue it and launch it after
> one of the others completes.

---

## What happens during a run

```
Cell 1  → Read parameters
Cell 2  → Mount Google Drive (auth pop-up the first time)
Cell 3  → git clone/pull → pip install -r utils/config/requirements.txt
Cell 4  → bash utils/colab/run_colab.sh <PKG> <N_TRIALS> <RESUME_DATE>
              ├── reattach if previous PID is still alive (idempotent)
              ├── supervisor (bg) loops up to MAX_RESTARTS=10 times:
              │     python -m <group>.<pkg>.ext.optimize_<x>           (PID A)
              │         --out-dir /content/drive/.../<date>_BAYESIAN_OPT
              │         --storage sqlite:////content/drive/.../optuna_study_<date>.db
              │         --resume  <date>     (auto-added)
              └── tail -f --pid <supervisor> bayesian_opt.log     (blocks cell)
```

The supervisor PID is written to `optimize.pid`. Stdout/stderr of every
relaunch is appended to `bayesian_opt.log` / `bayesian_opt_err.log` on
Drive, and the supervisor logs each restart attempt to
`supervisor.log`.

> **Why does Cell 4 block?** Colab Pro+ Background Execution only keeps
> the VM alive while at least one cell is **actively executing**. A
> detached `nohup` would let the cell return immediately, the notebook
> would look idle, and Colab would reclaim the VM within minutes of you
> closing the browser — killing the optimisation. Blocking the cell on
> the supervisor PID is what makes "close the browser" actually safe.
>
> **Why an auto-restart supervisor?** Even with Background Execution
> on, the VM occasionally gets reclaimed (idle network, soft-OOM, the
> 24h Pro+ limit). When you reconnect and re-run Cell 4, two things
> happen: (a) if the previous python process is still alive the cell
> just reattaches to its log; (b) if the python died, the supervisor
> relaunches it with `--resume <date>` and Optuna continues from the
> last completed trial — you only ever lose the seconds spent inside the
> trial that was running when the VM died.

---

## Resuming after a 24h timeout

When Colab kills your VM, all artifacts are already on Drive. To resume:

1. Open a fresh `launch.ipynb` tab.
2. Set `RESUME_DATE` in Cell 1 to the date the original study started
   (the date appears in the directory name on Drive: `YYYY-MM-DD_BAYESIAN_OPT`).
3. Run all cells.

Optuna re-opens the SQLite DB with `load_if_exists=True` and continues from
trial *N+1*.

---

## Monitoring without keeping a browser open

The notebook ships a **unified monitor helper** that works identically for
every package (`pes_ql`, `pes_dql`, `pes_dqn`, `pes_rdqn`, `pes_a2c`, `pes_trf`). It reads the
shared on-Drive artifacts (`run_meta.json`, `bayesian_opt.log`,
`optuna_study_<date>.db`) so you never have to remember a different
monitoring command per package.

### Option 1 — one-shot snapshot (`monitor()`)
Re-run the **monitor** cell at the bottom of `launch.ipynb` whenever you
want a status report:

```python
import sys; sys.path.insert(0, '/content/Win_mPES')
from utils.colab.monitor import monitor
monitor()                              # active run for PKG set in cell 1
monitor(pkg='dqn', date='2026-04-22')  # any run, from any tab/machine
monitor(n_log_lines=30)                # show more recent trial lines
```

The snapshot prints package + module, PID + alive/dead status, completed/
pruned/failed trial counts, current best value + trial number, throughput
(trials/h), ETA, the last per-trial log lines, and any recent stderr.

### Option 2 — auto-refreshing dashboard (`follow()`)
Runs the same snapshot in a loop, clearing the cell output between
refreshes. Press Ctrl-C to stop the loop — the optimisation keeps running.

```python
from utils.colab.monitor import follow
follow(refresh=15)                     # snapshot every 15 s
```

### Option 3 — raw `tail -f` on the log file
For when you only want the underlying log stream:

```bash
!tail -f /content/drive/MyDrive/mPES/pes_ql/<date>_BAYESIAN_OPT/bayesian_opt.log
```

The log file starts with a structured banner (package, module, git SHA,
launch timestamp, full command line) so any `tail` immediately shows which
run the lines belong to.

---

## Files reference

| File | Purpose |
|------|---------|
| `utils/colab/launch.ipynb`     | The notebook you open in Colab. |
| `utils/colab/setup_colab.sh`   | Installs deps + exports env vars. |
| `utils/colab/run_colab.sh`     | Resolves `PKG` alias, builds Drive paths, writes `run_meta.json` + log banner, `nohup`s the optimisation. |
| `utils/colab/monitor.py`       | Unified `monitor()` / `follow()` helpers — same output for every package. |

## Modifying the optimise scripts

All six `optimize_*.py` scripts accept two additional CLI flags so the
Colab launcher can redirect storage and outputs to Drive without touching
the package code:

```
python -m <group>.<pkg>.ext.optimize_<x> [N_TRIALS] [--resume YYYY-MM-DD] \
    [--out-dir   PATH]   # default: <group>/<pkg>/inputs/<date>_BAYESIAN_OPT
    [--storage   URL]    # default: sqlite:///<out-dir>/optuna_study_<date>.db
```

Used packages (group prefix is required since the 2026-05 reorg):

| Package   | Module                                | Default trials |
|-----------|---------------------------------------|----------------|
| `pes_ql`   | `tabular.pes_ql.ext.optimize_rl`     | 100 |
| `pes_dql`  | `tabular.pes_dql.ext.optimize_rl`    | 50  |
| `pes_dqn`  | `ml.pes_dqn.ext.optimize_dqn`        | 60  |
| `pes_rdqn` | `ml.pes_rdqn.ext.optimize_rdqn`      | 60  |
| `pes_a2c`  | `ml.pes_a2c.ext.optimize_a2c`        | 30  |
| `pes_trf`  | `ml.pes_trf.ext.optimize_tr`         | 60  |

---

## Troubleshooting

**"ERROR: Google Drive is not mounted"** → Cell 2 was skipped or the auth
flow was cancelled. Re-run Cell 2.

**Optuna study "already exists" on resume** → Expected. `load_if_exists=True`
re-opens the DB and continues. If you actually want a fresh study, delete
the `optuna_study_<date>.db` file on Drive first.

**Background execution greyed out** → Pro+ feature. Verify your
subscription tier under *Runtime → Resources*.

**Cannot find the Background Execution toggle** → It only exists in the
Colab web UI at [colab.research.google.com](https://colab.research.google.com).
No VS Code extension exposes it. Open the notebook in a browser, then go to
*Runtime → Manage sessions* (or click the ▾ next to **Connect** at the
top right). Edit the notebook anywhere you like, but launch it from the
browser if you need PC-off operation.

**Process dies unexpectedly** → Check `bayesian_opt_err.log` on Drive.