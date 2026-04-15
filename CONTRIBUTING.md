# Contributing to HelixForge

Thanks for helping improve HelixForge.

## What Helps Most

- bug reports with exact commands and stack traces
- feature requests tied to a real MLDE workflow need
- pull requests that keep scripts practical, explicit, and easy to debug

## Development Setup

Conda:

```bash
conda env create -f PET.yml
conda activate pet
```

or venv:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Before Opening a Pull Request

Please run the lightweight checks that apply to your change.

Syntax checks:

```bash
python -m py_compile scripts/*.py scripts/embeds_scripts/*.py
```

CLI checks:

```bash
python scripts/training3_optuna_mlde_uncertainty.py --help
python scripts/select_best_supervised_model.py --help
python scripts/rank_zero_shot_candidates.py --help
```

If you touched documentation, verify the commands still match the actual CLI.

## Pull Request Guidelines

- keep behavior changes intentional and narrow
- prefer small, debuggable scripts over clever abstractions
- preserve OOF-based model selection
- keep rank-based evaluation first-class
- do not silently change output schemas without updating docs

## Reporting Bugs

Include:

- the command you ran
- the dataset or file paths involved
- the exact traceback or error message
- what you expected to happen instead

## Style

- prefer explicit flags and readable defaults
- keep generated-file naming stable
- preserve compatibility with existing repo structure when reasonable
