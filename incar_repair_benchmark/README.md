# INCAR Repair Benchmark

This standalone benchmark evaluates whether an LLM can repair a deliberately corrupted VASP INCAR.

Each case contains:

- `POSCAR`
- `INCAR_reference`
- `INCAR_bad`
- `error_manifest.json`
- the same metadata/scoring tags used by the generation benchmark

The model sees the broken INCAR and must output a corrected final INCAR.
