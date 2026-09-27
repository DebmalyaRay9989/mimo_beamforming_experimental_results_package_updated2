# MIMO Beamforming Optimizer — Statistically Defensible Reproducibility Package

This package extends the original PyTorch reproduction runner so that comparative claims can be supported by repeated independent seeds, confidence intervals, paired comparisons, imperfect-CSI tests, feedback-delay tests, codebook validation, and channel-model sensitivity.

## What changed

1. **Repeated independent seeds** — the default `configs/medium.json` uses seeds 42–51 (10 independent runs).
2. **95% confidence intervals** — two-sided Student-t intervals are computed across independent seeds for each condition.
3. **Paired seed comparisons** — algorithm deltas are matched by seed and condition, with 95% CIs.
4. **`γ=0` contextual-bandit baseline** — `Contextual Bandit (gamma=0)` is evaluated as a one-step reward learner; future rewards are removed from its target.
5. **Imperfect CSI** — CSI quality is swept over 1.0, 0.8, and 0.5 using a controlled noisy channel estimate.
6. **Feedback delay** — observations are delayed by 0, 1, and 3 environment steps.
7. **Zadoff–Chu validation** — ZC and DFT codebooks are compared on oracle beamforming gain before the main run. The ZC codebook is retained only when its mean oracle-gain ratio to DFT is at least the configured threshold (default 0.95).
8. **Channel-model sensitivity** — a documented 3GPP TR 38.901-inspired UMa stochastic surrogate is available, plus a loader for user-supplied DeepMIMO `.npy`/`.npz` channel samples.
9. **Reproducibility metadata** — configurations, seeds, codebook decision, channel-model caveat, and CI method are recorded in `results/run_metadata_extended.json`.

## Important channel-model qualification

The included UMa implementation is **3GPP TR 38.901-inspired, not a standards-complete 38.901 implementation**. It uses clustered geometric multipath, LOS/NLOS mixing, angular spreads, and normalized ULA steering to move the experiment materially beyond i.i.d. Rayleigh fading. For a standards-complete study, replace this adapter with a validated 38.901 implementation such as a trusted channel-simulation package.

The package can also consume exported DeepMIMO channel arrays. Place the data at the path specified by `deepmimo_path` and select `channel_model: "DeepMIMO"`.

3GPP TR 38.901 is the technical report titled “Study on channel model for frequencies from 0.5 to 100 GHz”; the current 3GPP portal records it as specification 38.901.

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python run_experiments.py --config configs/medium.json
```

For a short smoke test:

```bash
python run_experiments.py --config configs/medium_quick.json --quick
```

For explicit seeds:

```bash
python run_experiments.py --config configs/medium.json --seeds 42,43,44,45,46,47,48,49,50,51
```

## DeepMIMO

Use `configs/deepmimo_example.json` as a template. The loader accepts:

- `.npy`: complex array `[samples, antennas]`
- `.npz`: array named `H`, or the first array in the archive
- alternatively a real-valued `[samples, 2*antennas]` array containing real components followed by imaginary components.

Example:

```json
{
  "channel_model": "DeepMIMO",
  "deepmimo_path": "data/deepmimo_channels.npy"
}
```

The supplied package does not bundle a DeepMIMO dataset; this avoids redistributing external data without permission.

## Output tables

- `results/tables/codebook_validation_by_seed.csv`
- `results/tables/codebook_validation_summary.csv`
- `results/tables/training_history_extended.csv`
- `results/tables/snr_results_extended.csv`
- `results/tables/snr_summary_95ci.csv`
- `results/tables/paired_algorithm_differences_95ci.csv`
- `results/tables/final_training_summary_extended.csv`

Figures:

- `results/figures/imperfect_csi_95ci.png`
- `results/figures/feedback_delay_95ci.png`

The old single-seed CSVs and figures remain in the archive as historical outputs and should not be used as evidence for the new statistical claims.

## Interpretation rules

Do not describe a mean difference as statistically supported merely because its means differ. For the repeated-seed results, report the estimated difference together with its 95% CI and the number of independent seeds. If a CI spans zero, describe the comparison as inconclusive under this protocol rather than claiming superiority.

The gamma-zero baseline is a contextual-bandit baseline, not a discounted RL agent. It tests whether explicitly modeling future reward improves performance over an immediate-reward decision rule.

## References

- 3GPP TR 38.901, “Study on channel model for frequencies from 0.5 to 100 GHz.”
- DeepMIMO: use the dataset/scenario documentation corresponding to the exact channel export used in a paper.
