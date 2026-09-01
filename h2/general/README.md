# h2 — Experimental Line

`h2/` mirrors the organisation of `h1/` so experimental variants can be
added without changing the validated line.

## Status

**Temporarily suspended.** No h2 experiment or benchmark is currently active.

The only package present at this stage is `tabular_uq/ql_uq`. Its
uncertainty-quantification protocol, training configuration, evaluation
criteria, and comparison against h1 are still to be defined.

## Current structure

```text
h2/
├── general/                 # reserved for a future h2 benchmark
└── tabular_uq/
    └── ql_uq/               # experimental Q-Learning + UQ package
        ├── config/
        ├── doc/
        ├── ext/
        ├── inputs/
        ├── outputs/
        └── src/
```

The copied benchmark scripts and comparison document are retained as
scaffolding only. Do not use them to report h2 results until the experiment
is reactivated and its model catalogue is established.