# Results note

The pre-existing CSV/PNG/PDF/DOCX outputs in this archive are historical single-seed results and are retained for traceability.

The **new runner and protocol are the authoritative experimental implementation** for the requested statistical validation. Run `python run_experiments.py --config configs/medium.json` to generate the 10-seed benchmark.

A small `results/smoke_validation/` run is included only as an execution/format validation (2 seeds, reduced episodes and evaluation). It is **not evidence for publication claims**.

The smoke validation exercised:
- repeated seeds,
- 95% CI table generation,
- `gamma=0` contextual-bandit baseline,
- imperfect CSI,
- feedback delay,
- automatic Zadoff–Chu validation/drop,
- 3GPP-inspired UMa channel generation.

In the smoke configuration, the Zadoff–Chu oracle-gain ratio was below the configured retention threshold, so the main smoke benchmark automatically used DFT only. Re-run the full protocol before reporting any numerical scientific conclusion.
