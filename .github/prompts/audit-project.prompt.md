# Audit mPES — Code, Config, and Docs

> Last updated: 2026-04-28

Perform a systematic correctness, consistency, and documentation audit of
the mPES workspace.  Detect drift between code and documentation, stale
references, and small bugs introduced during recent refactors.  Apply the
fixes you find unless an item is genuinely ambiguous or destructive — in
which case, surface it for review.

## Usage

When invoking this prompt, specify either a **single target package** or
the keyword `all` to audit the entire project.  Examples:

```text
@audit-project pes_base
@audit-project pes_ql
@audit-project pes_dql
@audit-project pes_dqn
@audit-project pes_a2c
@audit-project utils
@audit-project all
```

Throughout this prompt, `<TARGET>` refers to the value provided.  When
`<TARGET>` = `all`, run every step on each of the following packages, in
this canonical order:

```text
pes_base → pes_ql → pes_dql → pes_dqn → pes_rdqn → pes_a2c → pes_trf → pes_ens → utils
```

When `<TARGET>` is a single package, restrict every search and edit to
that package's tree (plus `utils/` only if the audit signal explicitly
points there).  Never edit a sibling package as a side-effect.

## Ground Rules

- **Verify before editing.**  Subagent / search results frequently flag
  false positives (e.g. claim a docstring is missing when it already
  exists).  Always `read_file` or `grep_search` to confirm a finding
  before applying any change.
- **Each package is self-contained.**  Do **not** introduce
  cross-package imports or shared modules outside `utils/`.  Per-package
  duplication of `src/` modules is by design; do not "deduplicate".
- **Implementation discipline.**  Apply only the audit fixes themselves.
  Do not add unrelated features, refactor untouched code, or rewrite
  comments / type hints in code you did not change.
- **Operational safety.**  Never commit, push, drop, force-push, or
  delete history.  Surface destructive recommendations for the user
  instead of executing them.
- **Style.**  Max 120 chars per line, PEP 8 indentation, NumPy alias is
  `numpy` (never `np`), `snake_case` and `PascalCase` both accepted.
  Public functions / classes need NumPy-style English docstrings.
- **Docs.**  `.md` content stays in Spanish, KaTeX-compatible LaTeX
  (`$...$`, `$$...$$`).  Never hand-edit `.html`; always re-export with
  `python utils/scripts/_export_html.py <PKG>`.

## Step 1 — Discover the target

For each package in scope, read:

| Item | Path |
|------|------|
| Init / exports | `<PKG>/__init__.py` |
| Config | `<PKG>/config/CONFIG.py` |
| Entry point | `<PKG>/__main__.py` |
| Core | `<PKG>/ext/*.py` |
| Support | `<PKG>/src/*.py` |
| Docs | `<PKG>/doc/*.md` |

For `utils`, read `utils/scripts/`, `utils/colab/`, `utils/linux/`,
`utils/win/`, `utils/config/`.

Run independent reads in parallel.  Do not read the same file twice.

## Step 2 — Static checks

Run, in parallel where possible:

1. `get_errors` on the entire package tree.
2. `grep_search` for each of the audit signals listed in Step 3.
3. Pyright (if `utils/config/pyrightconfig.json` exists):

   ```powershell
   $env:VIRTUAL_ENV='win_mpes_env'; $env:PYTHONIOENCODING='utf-8'
   .\win_mpes_env\Scripts\pyright.exe --project utils\config\pyrightconfig.json
   ```

   Linux equivalent uses `linux_mpes_env/bin/pyright`.  Result must be
   `0 errors, 0 warnings, 0 informations`.

(`utils/tests/` is not currently maintained; skip the pytest step
unless the directory exists.)

## Step 3 — Audit signals (search patterns)

Run these searches across the target.  Each match is a **candidate** —
inspect the surrounding lines before acting.

### 3.1 Stale documentation / metric drift

| Pattern | What it usually means |
|---------|----------------------|
| `BestCaseAllocations` | Old "max-allocation-everywhere" baseline; should reference DP optimum (`_best_feasible_sequence_severity`). |
| `asignación máxima en cada` / `max allocation .*every trial` | Same as above, in Spanish/English prose. |
| `MAX_ALLOCATABLE_RESOURCES` inside `doc/*.md` | Likely a stale formula / numeric example that pre-dates the DP backport. |
| `WorstCase == BestCase` | Sanity-check the surrounding text still matches the sanitisation logic in `objective()`. |
| Hard-coded numeric examples (e.g. `severity = 9 * L`) | Confirm against current `CONFIG.py` constants. |
| Module-structure diagrams in `doc/*.md` | Must list **every** top-level function exported by the matching `ext/*.py` (use `grep "^def "` to compare). |
| Hyperparameter tables in `doc/*.md` | Must match the current bounds in `optimize_*.py`. |

### 3.2 Code / signature drift

| Pattern | What it usually means |
|---------|----------------------|
| `def reset\(self,\s*seed` (positional `seed`) | Should be keyword-only: `def reset(self, *, seed=None, options=None)`. |
| `# type: ignore\[override\]` near `reset` / `step` | Shim left over after the signature fix; remove and verify pyright still passes. |
| `# type: ignore` without a justification comment | Either justify in-line or remove. |
| `from \.\. import \*` outside `__init__.py` | Violates explicit-import rule. |
| `import numpy as np` | Forbidden — must be `import numpy`. |
| Double-assignment typos: `(\w+)\s*=\s*\1\s*=\s*` on a single line | Real bug previously found in `log_utils.py` on multiple packages. |
| `FIXME\|XXX\|HACK` (not `TODO`) | Should be downgraded to `NOTE` or resolved. |
| `lambda *a, **kw: None` outside a `TYPE_CHECKING` fallback block | Replace with a `def` carrying a one-line NumPy docstring. |
| `except Exception` without re-raise or logging | Justify or narrow the exception type. |

### 3.3 Cross-platform / environment

| Pattern | What it usually means |
|---------|----------------------|
| Hard-coded `\\` or `/` in path joins | Should use `os.path.join`. |
| Hard-coded `win_mpes_env` / `linux_mpes_env` in `.py` | Should be OS-detected or templated.  In `utils/win/*.ps1` and `utils/linux/*.sh` it is by design. |
| Missing `VIRTUAL_ENV` / `PYTHONIOENCODING` / `TF_ENABLE_ONEDNN_OPTS` setup in launcher scripts | Required for non-interactive runs.  Verify parity between the `.sh` and `.ps1` variants. |
| `.sh` without matching `.ps1` (or vice-versa) under `utils/linux` / `utils/win` | Both variants must exist (compare `BaseName` lists). |

### 3.4 Optuna / persistence

| Pattern | What it usually means |
|---------|----------------------|
| Direct `pickle.dump` / `pickle.load` of trained artefacts | Should use NPZ + JSON helpers (`_save_best_artifacts` / `_load_best_artifacts`). |
| `optuna.create_study(.*storage=None` in production paths | Should use a SQLite (or RDB) storage URL so trials survive interruptions. |
| `study.best_value` without a `try/except ValueError` | Crashes the progress callback before any trial completes; wrap and fall back to `n/a`. |
| `f"…{trial.value:.4f}…"` without a None check | Pruned/failed trials have `value is None`; guard with an inline `if t.value is not None`. |

### 3.5 Configuration & dependencies

| Pattern | What it usually means |
|---------|----------------------|
| `utils/config/requirements.txt` last-updated date older than the workspace | Refresh against `pip freeze` (preserve section comments and ordering). |
| `utils/config/pyrightconfig.json` `exclude` missing `**/node_modules`, `**/__pycache__`, `**/.*` | Add Pylance default excludes alongside project-specific ones. |
| `utils/config/.pylintrc` | **Do not modify.** |

## Step 4 — Apply fixes

- Use `multi_replace_string_in_file` for batched independent edits.
- Re-run the relevant `grep_search` after each batch to confirm zero
  remaining matches.
- Run `get_errors` on edited files immediately to catch regressions.
- After Python edits: re-run pyright on the affected package(s).
- After `.md` edits: re-run pymarkdown:

  ```powershell
  .\win_mpes_env\Scripts\pymarkdown.exe scan -r <PKG>/doc
  ```

## Step 5 — Re-export documentation

For every package whose `doc/*.md` was modified:

```powershell
$env:VIRTUAL_ENV='win_mpes_env'; $env:PYTHONIOENCODING='utf-8'
.\win_mpes_env\Scripts\python.exe utils\scripts\_export_html.py <PKG>
```

(Linux: `python utils/scripts/_export_html.py <PKG>` after activating
`linux_mpes_env`.)

For `.md` files outside a `<PKG>/doc/` directory (e.g.
`utils/colab/colab_workflow.md`) call `convert_md_to_html` directly:

```powershell
.\win_mpes_env\Scripts\python.exe -c "import sys; sys.path.insert(0,'utils/scripts'); `
  from _export_html import convert_md_to_html; `
  convert_md_to_html('<path>.md','<path>.html')"
```

Do **not** hand-edit `.html` outputs.

## Step 6 — Verification

For `<TARGET>` = single package:

1. `get_errors` over the package — must be clean.
2. `grep_search` for each Step 3 pattern over the package — must be
   empty (or each remaining match explicitly justified in §7).
3. Pyright on the project config — `0 errors / 0 warnings / 0 informations`.
4. Confirm every modified `.md` has a refreshed `.html` sibling.

For `<TARGET>` = `all`, additionally:

5. Quick import smoke-test for each package:

   ```powershell
   foreach ($p in 'pes_base','pes_ql','pes_dql','pes_dqn','pes_rdqn','pes_a2c','pes_trf','pes_ens','utils') {
     .\win_mpes_env\Scripts\python.exe -c "import importlib; m=importlib.import_module('$p'); print('OK', m.__name__)"
   }
   ```

6. Cleanup any scratch files left in the workspace root (e.g. `_pr.txt`,
   `_md.txt`, `_plog.txt`, `pyright_full.txt`, `_freeze.txt`).

## Step 7 — Report

Produce a concise summary with:

- **Files modified** — workspace-relative markdown links with line
  numbers when useful.
- **Audit signals — clean** — table showing each Step 3 pattern with
  result count after fixes.
- **Findings rejected as false positives** — one-line justification each.
- **Surfaced for user review** — deferred / destructive items requiring
  user confirmation, with proposed action.
- **Verification** — pyright counts, pytest count, smoke-import results,
  HTML re-export status.

Keep the report short — bullets, not prose.  Do **not** create a
separate markdown document for the report unless the user asks for one.

## Known False Positives (do not "fix")

- Per-package `src/` duplication (`log_utils.py`, `terminal_utils.py`,
  `result_formatter.py`, `pygameMediator.py`, `exp_utils.py`) — by
  design, do not consolidate.
- Per-package `ext/exp_utils.py` reimplementation of
  `_best_feasible_sequence_severity` — by design, must stay
  package-local.
- `utils/scripts/run_module.py` "file-handle leak" — wrapper exits
  immediately, OS reclaims handles.
- `from .config.CONFIG import *` inside `<PKG>/__init__.py` — allowed
  re-export pattern, do not change.
- Both `snake_case` and `PascalCase` in scientific code — accepted by
  project convention.
- Hard-coded `win_mpes_env` in `utils/win/*.ps1` and `linux_mpes_env`
  in `utils/linux/*.sh` — OS-specific scripts may pin their own venv.
- `pes_trf` referenced in `utils/{linux,win}/run_bayesian_opt.{sh,ps1}`
  but missing from the workspace tree — placeholder for an unreleased
  package; surface for review, do not auto-remove.
- `# type: ignore[override]` on `Pandemic.step()` return tuple — was
  obsolete after the `reset()` signature fix; if pyright passes without
  it the shim should be removed (verified across all packages on
  2026-04-21).
