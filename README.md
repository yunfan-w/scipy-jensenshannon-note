# Numerical Case Study: SciPy Jensen–Shannon Stability

This repository documents a numerical stability issue in `scipy.spatial.distance.jensenshannon` and the corresponding fix submitted in SciPy:

🔗 PR: https://github.com/scipy/scipy/pull/24664

The issue arises when computing the Jensen–Shannon distance between proportional vectors (i.e., vectors that normalize to identical probability distributions). Due to floating-point round-off, the implementation may:

- Produce a tiny negative divergence leading to `NaN` after `sqrt`
- Yield a non-negligible positive distance under float32 precision

The full technical analysis, reproduction, and fix explanation can be found in:

📄 `docs/jensenshannon-numerical-stability.md`

This case study demonstrates:

- Numerical stability analysis in scientific computing
- Floating-point precision reasoning
- Root cause identification in production scientific code
- Design of regression tests for robustness
- Contribution to a large-scale open-source library (SciPy)


## Reproduction

To reproduce the issue (pre-fix behavior):

```bash
python repro/repro_jensenshannon.py
```

Observed output:
```bash
float64 10-d: nan
float32 3-d: 0.00021143198
```