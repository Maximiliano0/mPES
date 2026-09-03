<div align="center">

# h2 — Experimental Line

`h2/` mirrors the project structure used by the active branch and is reserved
for experimental variants that have not yet been promoted to the validated
workflow.

## Status

**Currently suspended.** No `h2` benchmark is active in the current workspace.
This line is kept as a staging area for experimental work such as the
`tabular_conf/ql_conf` package and related documentation scaffolding.

## Current structure

```text
h2/
├── general/                 # reserved for a future h2 benchmark
├── tabular_conf/
│   └── ql_conf/            # experimental tabular Q-Learning variant
│       ├── config/
│       ├── doc/
│       ├── ext/
│       ├── inputs/
│       ├── outputs/
│       └── src/
└── README.md               # this file
```

The benchmark scripts and comparative reports in this line are retained as
scaffolding only. They should not be used to report active h2 results until
that experiment is reactivated and the package catalogue is approved.
