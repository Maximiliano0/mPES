# h3 — Experimental Line

`h3/` continues the experimental branch pattern used by the project and keeps
its own scaffolding separate from the validated `h1/` workflow.

## Status

**Currently not active.** No `h3` benchmark is in the active execution path.
The only package present at this stage is `tabular_uq/ql_uq`, which remains a
research prototype and should be considered experimental until a formal
benchmark is defined.

## Current structure

```text
h3/
├── general/                 # reserved for a future h3 benchmark
└── tabular_uq/
    └── ql_uq/               # experimental Q-Learning + UQ package
        ├── config/
        ├── doc/
        ├── ext/
        ├── inputs/
        ├── outputs/
        └── src/
```

The copied benchmark scaffolding and comparison documents are retained for
reference only. They should not be used to report active h3 results until the
associated workflow and model catalogue are revalidated.
