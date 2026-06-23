# VASP INCAR Benchmark

`VASP INCAR Benchmark` is a benchmark for evaluating large language models on
task-aware generation and repair of VASP `INCAR` files.

The repository focuses on two benchmarked capabilities:

- `INCAR` generation from structure and compact task context
- `INCAR` repair from a task-aware but potentially corrupted INCAR draft

It is designed to measure workflow-semantic correctness, method-policy
alignment, minimum task-runnability, and repair-specific preservation behavior.

## What This Repository Contains

- benchmark definitions under `problems/`
- benchmark construction, run, scoring, and reporting scripts under `scripts/`
- safe example configuration templates under `config/`
- released metadata and leaderboard summaries for generation and repair

## What This Repository Does Not Benchmark

- `KPOINTS`
- `POTCAR`
- full VASP execution and convergence
- physical-observable quality such as energies, forces, magnetization, or bands

The results should therefore be interpreted as an evaluation of `INCAR`
construction competence at the workflow-preparation stage, not as a complete
electronic-structure benchmark.

## Benchmark Overview

The current curated benchmark version is defined by:

- `problems/problem_set_v1.0.csv`

The benchmark contains:

- `192` generation cases
- `576` repair cases derived from the generation benchmark

The case set spans:

- four task types: `static_scf`, `geometry_relax`, `line_mode_bands`,
  `dos_nscf`
- multiple material families
- multiple challenge types, including NSCF workflow, DFT+U, SOC, vdW,
  smearing, symmetry, and magnetic initialization
- difficulty tiers `L1`, `L2`, and `L3`

## Repository Layout

- `config/`: safe config templates and prompt templates
- `problems/`: benchmark problem-set CSVs
- `scripts/`: core builders, runners, scorers, and report generators
- `incar_generation_benchmark/`: released generation metadata and leaderboards
- `incar_repair_benchmark/`: released repair metadata and leaderboards

## Install

```bash
pip install .
```

Or for local development:

```bash
pip install -e .
```

## Minimal Workflow

Build generation benchmark cases:

```bash
vasp-incar-build-generation \
  --csv problems/problem_set_v1.0.csv
```

Build repair benchmark cases:

```bash
vasp-incar-build-repair
```

Run generation models:

```bash
vasp-incar-run-generation
```

Run repair models:

```bash
vasp-incar-run-repair
```

Score generation outputs:

```bash
vasp-incar-score-generation --model-name your_model_name
```

Score repair outputs:

```bash
vasp-incar-score-repair --model-name your_model_name
```

## Released Outputs

This repository includes released summary-level outputs such as:

- generation leaderboard summaries under
  `incar_generation_benchmark/leaderboards/`
- repair leaderboard summaries under
  `incar_repair_benchmark/leaderboards/`
- benchmark metadata indices and build reports

Full local run logs, private credentials, and full case-level model outputs are
not included in this public export.

## Included Scripts

The public export keeps the core scripts needed for benchmark construction,
model execution, scoring, and report generation. Helper modules required by
these entry points are also included under `scripts/`.

## Citation

If you use this benchmark, please cite the associated paper or repository
release. A machine-readable citation file is provided in `CITATION.cff`.

## License

This project is distributed under the `MIT` License. See `LICENSE`.
