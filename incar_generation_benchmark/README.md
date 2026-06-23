# INCAR Generation Benchmark

This benchmark is generated from a CSV problem file plus either:

- Materials Project source data
- local structure and INCAR seed files

## Workflow

1. Edit a CSV like `problems/problem_set_v1.0.csv`
2. Choose one source per row:
   - `source_kind=mp`
   - `source_kind=local`
3. Build the benchmark:

```bash
python3 ./scripts/build_incar_generation_benchmark.py
```

The builder is resumable by default. Re-running the same command will skip any case
whose output files already exist. Use `--overwrite` only when you want to rebuild
everything from scratch.

4. Run enabled models:

```bash
python3 ./scripts/run_llm_incar_batch.py
```

5. Score all generated INCAR files:

```bash
python3 ./scripts/score_incar_generation_batch.py --model-name gpt4o
```
