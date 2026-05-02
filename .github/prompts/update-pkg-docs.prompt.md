# Update Package Documentation

> Last updated: 2026-04-28

Read a target package's source code, then update and export its Markdown
documentation so that it faithfully explains the current implementation.
Also refresh the workspace-level `README.md` whenever the changes affect
project-wide facts (package list, layout, dependencies, usage examples).

## Usage

When invoking this prompt, specify the **target package** name, or `all`
to process every package in the workspace. Examples:

```
@update-pkg-docs pes_base
@update-pkg-docs pes_ql
@update-pkg-docs pes_dql
@update-pkg-docs pes_dqn
@update-pkg-docs pes_a2c
@update-pkg-docs pes_trf
@update-pkg-docs pes_rdqn
@update-pkg-docs all
```

Throughout this prompt, `<PKG>` refers to the package name provided by the
user (e.g. `pes_base`, `pes_ql`, `pes_dql`, `pes_dqn`, `pes_rdqn`, `pes_a2c`, `pes_trf`, `pes_ens`). When
invoked with `all`, run Steps 1–4 once per existing
package (discover them by listing the workspace root for `pes_*/`
folders), then run Step 5 (HTML export) and Step 6 (README refresh) once
at the end.

## Scope

Discover the package layout dynamically:

| Item | Path |
|------|------|
| Package root | `<PKG>/` |
| Config | `<PKG>/config/CONFIG.py` |
| Init / exports | `<PKG>/__init__.py` |
| Experiment entry | `<PKG>/__main__.py` |
| Core algorithms | `<PKG>/ext/*.py` |
| Support modules | `<PKG>/src/*.py` |
| Documentation | `<PKG>/doc/*.md` → `<PKG>/doc/*.html` |

## Step 1 — Read the project

1. List the contents of `<PKG>/`, `<PKG>/ext/`, `<PKG>/src/`, and `<PKG>/doc/`
   to discover all files.
2. Read **every** `.py` file inside the package.
3. Build a mental model of:
   - The Gym environment (`Pandemic` class in `ext/pandemic.py`): state space,
     action space, reward, transition dynamics.
   - Training / optimisation scripts in `ext/` (e.g. `train_rl.py`,
     `optimize_rl.py`): hyperparameters, algorithm variant, output artefacts.
   - The experiment runner (`__main__.py`): block/sequence/trial hierarchy,
     agent response flow, performance metrics.
   - Support modules in `src/`: severity calculations, logging, Pygame bridge,
     result plots, terminal output.
   - All tuneable constants from `config/CONFIG.py`.
   - Any package-specific features (Double Q-Learning, Bayesian optimisation,
     Transformer model, PBRS, etc.) — these vary by package.

## Step 2 — Read the existing `.md` files

1. List `<PKG>/doc/*.md` to discover all Markdown documentation files.
2. Read each one in full.
3. Note the purpose and structure of each file (some packages may have
   different `.md` files than others).

## Step 3 — Update the `.md` files

For **each** `.md` file, compare its content against the actual code read in
Step 1 and fix every discrepancy. The goal is that a reader can trust the
documentation as a faithful mirror of the code.

### What to check and update

- **Hyperparameter values**: learning rate, discount factor, epsilon, episodes,
  decay parameters. Must match what `train_rl.py` actually uses.
- **Code snippets**: every code block must correspond to the real source. Update
  variable names, function signatures, and logic if the code has changed.
  Use the variable name exactly as it appears in the source (e.g.
  `resolved_decay_rate` vs `decay_rate`).
- **Formulas**: verify that mathematical formulas (LaTeX) match the code
  implementation. Check reward formulas, severity update formulas, Q-update
  rules, and epsilon decay formulas.
- **State/action spaces**: check dimensions, ranges, and descriptions against
  `Pandemic.__init__()` and `CONFIG.py`.
- **Module descriptions**: verify that each module/function mentioned still
  exists, has the described purpose, and uses the described signature.
- **Constant values**: `MAX_SEVERITY`, `SEED`, `AVAILABLE_RESOURCES_PER_SEQUENCE`,
  `NUM_BLOCKS`, `NUM_SEQUENCES`, multipliers, etc.
- **Pipeline flow**: verify the described workflow (train → experiment → results)
  matches the actual scripts and their outputs.
- **New content**: if the code contains significant functionality not covered
  by any existing `.md`, add a new section (or a new `.md` if the topic is
  large enough to warrant one). Follow the same structure and tone.

### Style rules for `.md` content

- Write in **Spanish** (matching existing docs).
- Use **KaTeX-compatible LaTeX**: `$...$` for inline, `$$...$$` for display.
- Use fenced code blocks with `python` language tag.
- Keep tables aligned with `|---|` separators.
- Section numbering: `## 1.`, `### 1.1`, `#### subtitle`.
- ASCII diagrams for architecture/flow (match existing style).
- Include academic references where relevant (author, year, title, venue).

## Step 4 — Update docstrings

Using the mental model built in Step 1, review **every** public function and
class across all `.py` files in the package and ensure their docstrings are
present, accurate, and follow the **NumPy-style** convention.

### What to check and update

- **Missing docstrings**: every public function, method, and class must have a
  docstring. Add one if missing.
- **Stale descriptions**: if the implementation has changed (parameters added or
  removed, return type changed, logic altered), update the docstring to match.
- **Parameter lists**: verify that `Parameters`, `Returns`, `Raises`, and
  `Attributes` sections list exactly the current parameters/returns, with
  correct types and descriptions.
- **Examples section**: if present, confirm any example code still runs
  correctly against the current implementation.
- **Internal helpers**: private functions (prefixed with `_`) do not require
  docstrings but may have them — if present, ensure they are correct.

### NumPy-style template

```python
def function_name(param_a: int, param_b: str = "default") -> bool:
    """One-line summary (imperative mood, no period).

    Extended description explaining behaviour, edge cases, or
    algorithmic notes when useful.

    Parameters
    ----------
    param_a : int
        Description of param_a.
    param_b : str, optional
        Description of param_b (default "default").

    Returns
    -------
    bool
        Description of the return value.

    Raises
    ------
    ValueError
        When ``param_a`` is negative.
    """
```

### Style rules for docstrings

- Write docstrings in **English**.
- Use triple double-quotes (`"""`).
- First line: imperative summary ≤ 79 characters.
- Blank line between summary and body.
- Wrap body text at **120 characters** (project standard).
- Reference `CONFIG.py` constants by name when describing default values
  sourced from configuration.

## Step 5 — Export each `.md` to `.html`

Run the shared export script to convert every `.md` inside `<PKG>/doc/` to a
matching `.html` in the same directory:

```bash
python utils/scripts/_export_html.py <PKG>
```

The script uses the project's standard HTML template (KaTeX math rendering,
dark-mode CSS, responsive layout). It automatically extracts the `# H1`
heading as the `<title>`.

To convert **all** packages at once (no arguments):

```bash
python utils/scripts/_export_html.py
```

> **Note:** Do **not** generate HTML inline or with ad-hoc scripts. Always
> use `utils/scripts/_export_html.py` so that every package produces identical
> styling.

## Step 6 — Refresh the workspace `README.md`

After per-package documentation is up to date, review the top-level
`README.md` and update any project-wide facts that may have drifted from
reality. The README is written in **English** and acts as the project's
front page; keep it concise and high-level — defer algorithm details to
each package's `doc/`.

### What to check and update

- **Package table**: list every `pes_*` folder that exists, with a one-line
  algorithm description and the key files. Drop entries for removed packages.
- **Package layout** ASCII tree: confirm it matches the actual directory
  structure shared by all packages.
- **Setup section**:
  - Python version per OS (must match `copilot-instructions.md`).
  - Dependency table versions (cross-check with `utils/config/requirements.txt`).
  - Virtual-env activation snippets (Windows + Linux).
  - Required environment variables.
- **Usage examples**: every `python -m <pkg>` and training/optimisation
  command must reference a package and script that still exists.
- **Pandemic Scenario summary**: state/action ranges, severity formula,
  reward sign, and total state count must match `Pandemic.__init__()` and
  `CONFIG.py`.
- **Experiment structure**: block/sequence/trial counts must match
  `CONFIG.py` constants (`NUM_BLOCKS`, `NUM_SEQUENCES`, `NUM_MAX_TRIALS`).
- **Documentation pointer**: include a brief note that per-package docs
  live in `<pkg>/doc/` and are regenerated with
  `python utils/scripts/_export_html.py`.
- Bump the implicit "last updated" cue if present (date headers, etc.).

### Style rules for `README.md`

- Write in **English**.
- Keep the README under ~200 lines; link out to `<pkg>/doc/*.md` for depth.
- Use the same Markdown table syntax and fenced code blocks as the rest
  of the project.

## Checklist

Before finishing, verify:

- [ ] Every `.md` in `<PKG>/doc/` has been read and compared against the source code.
- [ ] Hyperparameters, formulas, code snippets, and descriptions match the code.
- [ ] No dead references to functions/classes/variables that no longer exist.
- [ ] Every `.md` has a corresponding up-to-date `.html` in `<PKG>/doc/`.
- [ ] HTML files render math correctly (KaTeX delimiters, no broken `$`).
- [ ] No English leaking into Spanish-language documentation (except code/names).
- [ ] Every public function and class in `<PKG>/` has a NumPy-style docstring.
- [ ] Docstring parameter lists, return types, and descriptions match the code.
- [ ] No stale references in docstrings to removed parameters or changed behaviour.
- [ ] `README.md` package table, layout, setup, usage, and scenario summary match the workspace state.
