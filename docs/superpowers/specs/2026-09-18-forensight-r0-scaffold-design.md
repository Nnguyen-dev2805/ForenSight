# ForenSight R0 Project Scaffold Design

## Goal

Create the smallest Python project scaffold needed to start R0 — Dataset + Evaluation Protocol — without implementing Stage 1, Stage 2, Stage 3, or speculative R0 utilities.

## Scope

The scaffold uses a `src/` layout with package name `forensight`.

Only these package namespaces are created:

- `forensight`
- `forensight.data`
- `forensight.evaluation`

They remain intentionally empty except for package initializers. No model, trainer, projector, fusion, pipeline, registry, distributed, split-building, evaluation-runner, or dataset-inspection implementation is added in this task.

## Project structure

```text
ForenSight/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── docs/
├── src/
│   └── forensight/
│       ├── __init__.py
│       ├── data/
│       │   └── __init__.py
│       └── evaluation/
│           └── __init__.py
├── scripts/
├── tests/
│   └── test_package.py
├── configs/
│   └── r0/
├── data/
│   ├── raw/
│   ├── manifests/
│   └── derived/
├── results/
└── notebooks/
```

Empty directories are retained with `.gitkeep` where Git tracking is useful.

## Python packaging

`pyproject.toml` declares:

- Python `>=3.11`;
- runtime dependencies: `numpy`, `pandas`, `Pillow`, `scikit-learn`;
- dev dependency: `pytest`;
- pytest `pythonpath = ["src"]`;
- pytest `testpaths = ["tests"]`.

No PyTorch dependency is added because this scaffold contains no code that requires it.

## Data and artifact policy

`data/raw/` is immutable project input and its contents are ignored.

`data/derived/` contents are ignored because derived datasets may be large and must be reproducible.

`data/manifests/` remains version-controlled so deterministic split manifests can be committed during R0.

Generated contents under `results/` are ignored while a placeholder may remain tracked.

The ignore rules also cover macOS metadata, Python caches, virtual environments, `.env`, `.codegraph`, notebook checkpoints, common model checkpoint formats, and credentials/secrets.

## Branding and commands

`README.md` is renamed from the old project label to `# ForenSight`, and its opening identifies the project as ForenSight.

`AGENTS.md` Project purpose identifies the project as ForenSight.

After the smoke test command is verified successfully, `AGENTS.md` records the canonical test command:

```bash
python -m pytest
```

No local-skill management section is added.

## Validation

The task is complete only when:

1. the requested scaffold exists;
2. `tests/test_package.py` verifies `import forensight`;
3. the smoke/unit test passes;
4. the relevant directory tree is printed;
5. `git status` is inspected;
6. Markdown is scanned for stale project branding, while legitimate technical or paper terminology is left unchanged.

## Constraints

This scaffold does not alter research questions, architecture, dataset strategy, split policy, evaluation protocol, or later-stage implementation boundaries.

R0 remains the only active milestone.
