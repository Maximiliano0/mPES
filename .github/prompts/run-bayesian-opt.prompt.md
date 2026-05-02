# Run Bayesian Optimisation

> Last updated: 2026-05-02

Launch the Bayesian optimisation for a given package.

Both a **Linux (Bash)** and a **Windows (PowerShell)** workflow are provided.
Detect the current OS and follow the appropriate path.

## Inputs

- `$PACKAGE` — the target package.
  Valid values: `pes_ql`, `pes_dql`, `pes_dqn`, `pes_rdqn`, `pes_a2c`, `pes_trf`.
- `$N_TRIALS` — number of optimisation trials (default: **30**).
- `$RESUME_DATE` *(optional)* — `YYYY-MM-DD` date string to resume a previous
  run stored under that date.

## Package → Module Map

Each package has its own optimisation module:

| Package | Module | Alias(es) |
|---------|--------|-----------|
| `pes_ql` | `pes_ql.ext.optimize_rl` | `bayesian`, `bay`, `1` |
| `pes_dql` | `pes_dql.ext.optimize_rl` | `dql`, `ql`, `2` |
| `pes_dqn` | `pes_dqn.ext.optimize_dqn` | `dqn`, `3` |
| `pes_rdqn` | `pes_rdqn.ext.optimize_rdqn` | `rdqn`, `7` |
| `pes_a2c` | `pes_a2c.ext.optimize_a2c` | `ac`, `a2c`, `actor-critic`, `4` |
| `pes_trf` | `pes_trf.ext.optimize_tr` | `transformer`, `tr`, `5` |

If the user provides an alias instead of the full package name, resolve it
using the table above.

## Quick Launch (recommended)

Both platforms have a ready-made script in `utils/`:

**Linux:**
```bash
./utils/linux/run_bayesian_opt.sh dqn 110
./utils/linux/run_bayesian_opt.sh ac 100
./utils/linux/run_bayesian_opt.sh bayesian 100 2026-02-12   # resume
```

**Windows (PowerShell):**
```powershell
.\utils\win\run_bayesian_opt.ps1 dqn 110
.\utils\win\run_bayesian_opt.ps1 ac 100
.\utils\win\run_bayesian_opt.ps1 bayesian 100 2026-02-12  # resume
```

These scripts handle environment activation, power settings and background
launch automatically.

> **IMPORTANT — Background execution:** All launched processes use
> `Start-Process -WindowStyle Hidden` **without** `-RedirectStandardOutput`
> or `-RedirectStandardError`. This is critical on Windows: the `-Redirect*`
> flags force `CreateProcess` (child inside the caller's job group), which
> means the process **dies** when VS Code or the parent terminal closes.
> Without those flags, `Start-Process` uses `ShellExecute`, creating a fully
> independent process that survives VS Code being closed.
>
> On Windows, stdout/stderr redirection is handled **inside Python** by
> `utils/scripts/run_module.py`.

The sections below describe the manual step-by-step process for reference.

---

## Manual Workflow — Linux (Bash)

All commands run from the **workspace root** using **relative paths** only.

### Step 0 — Activate the virtual environment

```bash
source linux_mpes_env/bin/activate
```

### Step 1 — Resolve package and module

Using the table above, derive:

- `PKG` — full package name (e.g. `pes_dqn`).
- `OPT_MODULE` — Python module path (e.g. `pes_dqn.ext.optimize_dqn`).

### Step 2 — Prepare the log directory

```bash
LOG_DIR="${PKG}/inputs"
mkdir -p "$LOG_DIR"
```

### Step 3 — Prevent system suspension

```bash
# Lid close → do nothing (AC and battery)
gsettings set org.gnome.settings-daemon.plugins.power lid-close-ac-action 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power lid-close-battery-action 'nothing'

# Disable automatic suspend on AC and battery
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'

# Disable screen blanking / screen lock
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
```

> **Note:** these settings persist across sessions. Revert them manually
> after the optimisation finishes if desired.

### Step 4 — Launch the optimisation

```bash
ARGS="$N_TRIALS"                          # fresh run
ARGS="$N_TRIALS --resume $RESUME_DATE"    # resumed run (if date provided)

nohup python3 -m "$OPT_MODULE" $ARGS > "$LOG_DIR/bayesian_opt.log" 2>&1 &
OPT_PID=$!
```

### Step 5 — Launch the suspend/shutdown inhibitor

```bash
nohup systemd-inhibit \
    --what=idle:sleep:shutdown:handle-lid-switch \
    --who="mPES Bayesian Optimization ($PKG)" \
    --why="Running $N_TRIALS-trial Bayesian optimization" \
    --mode=block \
    tail --pid=$OPT_PID -f /dev/null > /dev/null 2>&1 &
```

### Step 6 — Monitoring commands

```bash
grep 'Trial' $LOG_DIR/bayesian_opt.log | tail -10   # progress
tail -f $LOG_DIR/bayesian_opt.log                     # live
kill -0 $OPT_PID && echo "running" || echo "done"    # alive?
```

---

## Manual Workflow — Windows (PowerShell)

All commands run from the **workspace root** using **relative paths** only.

### Step 0 — Set environment variables

```powershell
$env:VIRTUAL_ENV       = Join-Path $PWD 'win_mpes_env'
$env:PYTHONIOENCODING  = 'utf-8'
$env:TF_ENABLE_ONEDNN_OPTS = '0'
```

These are **required** to avoid the `__init__.py` "Press ENTER" prompt,
`UnicodeEncodeError` on cp1252, and oneDNN log noise.

### Step 1 — Resolve package and module

Same table as above. Derive `$PkgName` and `$OptModule`.

### Step 2 — Prepare the log directory

```powershell
$LogDir = Join-Path $PWD "$PkgName\inputs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
```

### Step 3 — Prevent system suspension

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
powercfg /change monitor-timeout-ac 0
powercfg /change monitor-timeout-dc 0
```

> **Note:** these settings persist. Revert them manually after the
> optimisation finishes if desired.

### Step 4 — Launch the optimisation

Use `utils/scripts/run_module.py` to redirect stdout/stderr at the Python level.
Do **NOT** use `-RedirectStandardOutput/-Error` — those create child
processes that die when VS Code closes.

```powershell
$Python    = Join-Path $PWD 'win_mpes_env\Scripts\python.exe'
$RunModule = Join-Path $PWD 'utils\scripts\run_module.py'
$LogFile   = Join-Path $LogDir 'bayesian_opt.log'
$ErrFile   = Join-Path $LogDir 'bayesian_opt_err.log'

$pyArgs = @($RunModule, $OptModule, $LogFile, $ErrFile, "$NTrials")
if ($ResumeDate) { $pyArgs += @('--resume', $ResumeDate) }

$optProc = Start-Process -FilePath $Python `
    -ArgumentList $pyArgs `
    -WorkingDirectory $PWD `
    -PassThru -WindowStyle Hidden
$OptPid = $optProc.Id
```

The process is fully detached (ShellExecute) — **survives VS Code close**.

### Step 5 — Monitoring commands

```powershell
Select-String 'Trial' (Join-Path $LogDir 'bayesian_opt.log') | Select-Object -Last 10   # progress
Get-Content (Join-Path $LogDir 'bayesian_opt.log') -Wait -Tail 20                        # live
Get-Process -Id $OptPid -ErrorAction SilentlyContinue                                     # alive?
Get-Content (Join-Path $LogDir 'bayesian_opt_err.log') -Tail 20                           # errors
```

---

## Rules

- **Relative paths only** — all paths must be relative to the workspace root.
  Never use absolute paths (no `/home/…` or `C:\Users\…`).
- **OS detection** — always determine the current OS and use the appropriate
  workflow. Never hard-code a single virtual environment name.
- **Environment variables** — on Windows, always set `VIRTUAL_ENV`,
  `PYTHONIOENCODING=utf-8`, and `TF_ENABLE_ONEDNN_OPTS=0` before launching.
- Always activate `linux_mpes_env` before running any command.
- Run all commands from the workspace root directory.
- If `$PACKAGE` is not in the module map, stop and ask the user for
  clarification.
- If `$N_TRIALS` is not specified, default to **30**.
- If `$RESUME_DATE` is not specified, start a **fresh** run.
- Do **not** modify any Python source files — this prompt only *launches*
  existing optimisation scripts.
