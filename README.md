# preconditioned-npe-under-misspecification

.
├─ src/precond_npe_misspec/   # importable library code
│  ├─ examples  /             # SBI examples
│  ├─ algorithms/             # 
│  ├─ diagnostics/            # (? - may or may not include)
│  └─ cli.py                  # entrypoint: (TODO)
├─ configs/                   # YAML/dataclass presets for runs
├─ experiments/               # commands + notes
├─ scripts/                   # one-off utilities (e.g., data prep)
├─ results/                   # run artefacts (gitignored)
├─ data/                      # raw/interim/processed (gitignored)
├─ tests/                     # unit + smoke tests  (TODO: will see if need)
├─ docs/                      # mkdocs or sphinx (TODO: will see if need)
├─ jobs/                      # HPC launchers calling CLI
└─ assets/                    # figures for README/docs (TODO: will see if need)
