# Python 3 Script Template

> Last updated: 2026-05-04

Generate a new Python 3 script following the mPES project conventions.

## Packages

Packages are grouped under two top-level directories by algorithm family:

| Group | Package | Description |
|-------|---------|-------------|
| `tabular` | `pes_base` | Tabular Q-Learning baseline |
| `tabular` | `pes_ql` | Q-Learning + Bayesian optimisation (Optuna) |
| `tabular` | `pes_dql` | Double Q-Learning, ε-decay warm-up, PBRS |
| `ml` | `pes_dqn` | Deep Q-Network (experience replay + target net) |
| `ml` | `pes_rdqn` | Recurrent DQN (LSTM over trial history) |
| `ml` | `pes_a2c` | Advantage Actor-Critic (A2C, actor + critic nets) |
| `ml` | `pes_trf` | Causal Transformer encoder + DQN |
| `ens` | `pes_ens_sprb` | Confidence-weighted soft voting ensemble |
| `ens` | `pes_ens_accq` | Confidence-weighted action/Q-value ensemble |
| — | `utils` | Shared helpers |

When referencing a package in module imports or run commands, always
include the group prefix (`tabular.pes_base`, `ml.pes_dqn`, etc.).

## Structure

The generated script MUST follow this structure, in order:

1. **Module docstring** — Triple-quoted description of what the script does.
2. **External imports** — Under a `## Imports externos ##` section header comment.
3. **Internal imports** — Under a `## Imports internos ##` section header comment.
   Use **explicit** relative imports (`from .. import X, Y`, `from .module import Z`).
   Never use `from .. import *` outside `__init__.py`.
4. **Constants / configuration** — If needed, under a `## Configuración ##` section header.
5. **Functions / Classes** — Core logic, each with a NumPy-style docstring.
6. **`main()` function** — Entry point with a one-line docstring and the primary workflow.
7. **`if __name__ == '__main__': main()`** — Guard at the bottom.

## Conventions

- Use `numpy` instead of `np` for the numpy alias.
- Use `os.path.join()` for paths, never f-strings with `/`.
- Use `datetime.now().strftime("%Y-%m-%d")` for date stamps.
- Use terminal utilities from `..src.terminal_utils`: `header`, `section`, `success`, `info`, `list_item`.
- Save outputs to a date-stamped subdirectory under the package's `inputs/` or `outputs/` folder.
- Prefix unused variables with `_` (e.g., `_fig, ax = plt.subplots()`).
- PascalCase variables are acceptable (project convention for scientific code).
- 120-character max line length, 4-space indentation.
- Suppress non-critical warnings at the top:
  ```python
  warnings.filterwarnings('ignore', category=UserWarning, message='.*Box bound precision.*')
  ```
- Force CPU for TensorFlow if imported:
  ```python
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
  ```

## Quality gates

After generating the script, activate the correct virtual environment for the
current OS and run:

**Linux / macOS:**
```bash
source linux_mpes_env/bin/activate
pyright <PACKAGE_DIR>/
pylint --rcfile=utils/config/.pylintrc <PACKAGE_DIR>/
```

**Windows (PowerShell):**
```powershell
win_mpes_env\Scripts\Activate.ps1
pyright <PACKAGE_DIR>/
pylint --rcfile=utils/config/.pylintrc <PACKAGE_DIR>/
```

Targets: pyright → 0 errors, pylint → 10.00/10.

## Example skeleton

```python
'''
<PKG_NAME> — <Short description>

<Longer explanation of what this script does.>
'''

##########################
##  Imports externos    ##
##########################
import os
import sys
import numpy
import warnings
from datetime import datetime

##########################
##  Imports internos    ##
##########################
from .. import INPUTS_PATH
from ..src.terminal_utils import header, section, success, info, list_item

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box bound precision.*')

###################################
##             Main             ###
###################################
def main():
    """Run the <description> pipeline."""

    header("<SCRIPT TITLE>", width=80)

    run_date = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(INPUTS_PATH, f'{run_date}_<SUFFIX>')
    os.makedirs(out_dir, exist_ok=True)
    info(f"Output directory: {out_dir}")

    # ... core logic ...

    section("Complete", width=80)
    success("Pipeline finished successfully!")
    info(f"Output directory: {out_dir}")


if __name__ == '__main__':
    main()
```

## Workflow

1. **Use `tabular/pes_ql` as reference** — Before writing any new script,
   study the corresponding modules in `tabular/pes_ql/` (environment,
   training loop, optimisation, config, `__init__.py`, `__main__.py`)
   as the canonical implementation example. Mirror its patterns,
   naming, and structure unless the target package explicitly requires
   a different approach. For deep-learning targets, use `ml/pes_dqn`
   as a secondary reference.

2. **Write the documentation** — After finishing the code, create a
   Markdown file inside the target package's `doc/` directory (e.g.,
   `tabular/pes_dql/doc/pes_dql_explained.md` or
   `ml/pes_dqn/doc/pes_dqn_explained.md`). The document must:
   - Explain the **theoretical foundations** behind the algorithm or
     feature implemented (or place these in a sibling
     `<pkg>_theory.md`).
   - Map each theoretical concept to the **specific functions, classes,
     or code sections** where it is applied (include module paths and
     function names).
   - Follow the style of existing docs in `tabular/pes_base/doc/`
     (title, sections, equations where appropriate, code references).

## Rules

- Adapt the package name to its full grouped path
  (`tabular.pes_base`, `tabular.pes_ql`, `tabular.pes_dql`,
  `ml.pes_dqn`, `ml.pes_rdqn`, `ml.pes_a2c`, `ml.pes_trf`,
  `ens.pes_ens_sprb` or `ens.pes_ens_accq`) based on where the user wants to place the script.
- Do NOT cross-reference between packages — each package is
  self-contained. The intentional exceptions are `ens/pes_ens_sprb` and
  `ens/pes_ens_accq`,
  which loads pre-trained `.keras` artefacts from sibling packages
  under `ml/` via filesystem paths (no Python import).
- Ask the user for the script's purpose if not specified.
