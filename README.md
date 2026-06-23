# INCARBench

INCARBench is a benchmark for evaluating large language models on scientific
configuration tasks for VASP `INCAR` files. It covers both task-aware INCAR
generation and repair, with released benchmark metadata, scoring scripts, and
leaderboard summaries.

The benchmark is introduced in the arXiv paper:

- **INCARBench: A Benchmark for Scientific Configuration in VASP INCAR by Large
  Language Models**
- arXiv: [2606.23571](https://arxiv.org/abs/2606.23571)
- Authors: Bin Shao, Jixiang Li, Xinyue Zhang, Baishun Yang, Zhiyang Liu,
  Weichao Wang

## What This Repository Contains

- benchmark definitions under `problems/`
- construction, inference, scoring, and reporting scripts under `scripts/`
- safe example configuration templates under `config/`
- released metadata and leaderboard summaries for INCAR generation and repair

The current curated benchmark version is defined by:

- `problems/problem_set_v1.0.csv`

The released benchmark contains:

- `192` INCAR generation cases
- `576` INCAR repair cases derived from the generation benchmark

The case set spans four task types:

- `static_scf`
- `geometry_relax`
- `line_mode_bands`
- `dos_nscf`

It covers multiple material families and challenge types, including NSCF
workflow configuration, DFT+U, SOC, vdW corrections, smearing, symmetry, and
magnetic initialization.

## Scope

INCARBench evaluates workflow-level INCAR construction competence. It does not
benchmark:

- `KPOINTS`
- `POTCAR`
- full VASP execution and convergence
- downstream physical-observable quality such as energies, forces,
  magnetization, or band structures

The results should therefore be interpreted as an evaluation of
workflow-preparation and configuration correctness, not as a complete
electronic-structure benchmark.

## Repository Layout

- `config/`: example model/configuration templates and prompt templates
- `problems/`: benchmark problem-set CSVs
- `scripts/`: builders, runners, scorers, and report generators
- `incar_generation_benchmark/`: released generation metadata and leaderboards
- `incar_repair_benchmark/`: released repair metadata and leaderboards

## Install

```bash
pip install .
```

For local development:

```bash
pip install -e .
```

## Basic Usage

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

Generate Markdown reports:

```bash
vasp-incar-report-generation
vasp-incar-report-repair
```

## Released Outputs

This repository includes summary-level released outputs:

- generation leaderboard summaries under
  `incar_generation_benchmark/leaderboards/`
- repair leaderboard summaries under
  `incar_repair_benchmark/leaderboards/`
- benchmark metadata indices and build reports

Full local run logs, private credentials, and full case-level model outputs are
not included in this public export.

## Configuration

Copy the example configuration before running models:

```bash
cp config/llm_benchmark_config.example.json config/llm_benchmark_config.json
```

Then edit the local config with your model endpoints and keys. Local config
files containing credentials are ignored by Git.

## Citation

If you use INCARBench, please cite the paper:

```bibtex
@article{shao2026incarbench,
  title   = {INCARBench: A Benchmark for Scientific Configuration in VASP INCAR by Large Language Models},
  author  = {Shao, Bin and Li, Jixiang and Zhang, Xinyue and Yang, Baishun and Liu, Zhiyang and Wang, Weichao},
  journal = {arXiv preprint arXiv:2606.23571},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.23571}
}
```

A machine-readable citation file is also provided in `CITATION.cff`.

## License

This project is distributed under the MIT License. See `LICENSE`.
