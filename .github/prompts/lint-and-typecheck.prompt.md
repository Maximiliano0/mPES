# Lint & Type-Check (Fix Loop)

> Last updated: 2026-09-01

Run the mandatory quality gates on a package and **iteratively fix every
issue in source code** until both tools report zero problems. This prompt
is the single canonical source for the pyright + pylint fix loop. Other
agents and prompts should not run automatically when this one is merely
mentioned by name; whatever references it must explicitly `read_file` this
document and follow its steps, since custom agents cannot invoke prompt
files the way a user invokes a slash command.

## Inputs

- `$PACKAGE_DIR` — the package directory to check, expressed as the full
  path relative to its experiment line (`h1/` or `h2/`), including the
  algorithm-family group:
  - `h1` tabular family: `tabular/pes_base`, `tabular/pes_ql`, `tabular/pes_dql`.
  - `h1` deep-learning family: `ml/pes_dqn`, `ml/pes_rdqn`, `ml/pes_a2c`,
    `ml/pes_trf`.
  - `h1` benchmark harness: `general`.
  - `h2` (suspended): `tabular_uq/ql_uq`.
  - Shared helpers: `utils` (checked from the workspace root, not from
    inside `h1/` or `h2/`).

  If only the short name is provided (e.g. `pes_dqn`), resolve it to its
  group directory (`ml/pes_dqn`) inside `h1/` unless the user specifies `h2`.
  If omitted entirely, infer the package from the file that was just edited.

## Workflow

Repeat the loop below until **both** targets are met in the same iteration.

```
while issues remain:
    1. Run pyright  → read output → fix every issue in source
    2. Run pylint   → read output → fix every issue in source
```

### Step 0 — Activate the virtual environment and set the working directory

```powershell
win_mpes_env\Scripts\Activate.ps1
```

`h1/` and `h2/` are plain directories, not Python packages (no `__init__.py`
at their own level). Run pyright/pylint against `$PACKAGE_DIR` using a path
that already includes the `h1\` or `h2\` prefix from the workspace root —
do **not** `cd` into `h1/`/`h2/` for this prompt, since `utils/config/*`
is resolved relative to the workspace root.

### Step 1 — Pyright (static type checking)

```powershell
pyright --project utils\config\pyrightconfig.json h1\$PACKAGE_DIR\
```

- Read the full output. For **each** error, warning, or information:
  1. Open the reported file and line.
  2. Fix the root cause in source (add/correct type hints, fix imports, etc.).
- Do **not** use `# type: ignore` — fix the code instead.
- Re-run pyright after fixes. Repeat until output is
  `0 errors, 0 warnings, 0 informations`.

### Step 2 — Pylint (linting with project standard)

```powershell
pylint --rcfile=utils\config\.pylintrc h1\$PACKAGE_DIR\
```

- Read the full output. For **each** reported message:
  1. Open the reported file and line.
  2. Apply the appropriate fix (see common fixes below).
- Re-run pylint after fixes. Repeat until the score is `10.00/10`.

#### Common fixes

| Pylint message | Fix |
|----------------|-----|
| Unused import (`W0611`) | Remove the import line. |
| Unused variable (`W0612`) | Remove the variable, or prefix with `_`. |
| Unused argument (`W0613`) | Prefix with `_` (e.g. `_state`). |
| Bare `except:` (`W0702`) | Specify the exception type (e.g. `except OSError:`). |
| Missing docstring (`C0114/C0115/C0116`) | Add a NumPy-style docstring. |
| Trailing whitespace (`C0303`) | Remove trailing spaces. |
| Missing final newline (`C0304`) | Add a newline at EOF. |
| Unnecessary pass (`W0107`) | Remove the `pass` after a docstring or code. |
| `raise` missing `from` (`W0707`) | Use `raise ... from e`. |
| Bad unary operand (`E1130`) | Fix the operand type. |
| Wildcard import unused (`W0611` on `*`) | Replace `from .. import *` with explicit imports. |

### Step 3 — Cross-check

If fixes from step 2 introduced new pyright issues (or vice-versa), go back
to step 1 and repeat the full loop.

## Targets

| Tool    | Target                                |
|---------|---------------------------------------|
| pyright | `0 errors, 0 warnings, 0 informations` |
| pylint  | `10.00/10`                            |

Both targets must be met **simultaneously** before the task is considered done.

## Rules

- Always activate the virtual environment first (see Step 0).
- Run checks from the **workspace root**, never from inside `h1/` or `h2/`.
- Fix issues **in source code** — do not suppress, silence, or work around them.
- Do **not** add `# pylint: disable=` for rules that are **enforced** in `utils/config/.pylintrc`.
- Do **not** add `# type: ignore` comments.
- Do **not** modify `utils/config/.pylintrc` or `utils/config/pyrightconfig.json` to work around failures.
- When replacing `from .. import *` with explicit imports, check which names
  the module actually uses (grep/search the file) and import only those.
- Preserve existing functionality — fixes must be behaviour-neutral.
