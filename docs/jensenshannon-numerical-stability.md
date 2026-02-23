# Numerical Stability Analysis of scipy.spatial.distance.jensenshannon

##1. Overview

This document analyzes a numerical stability issue in scipy.spatial.distance.jensenshannon and describes the fix submitted in SciPy PR: https://github.com/scipy/scipy/pull/24664. The issue corresponds to gh-20083 and is related to several prior discussions including gh-19436, gh-20414, gh-20786, and gh-19438.

The problem occurs when computing the Jensen–Shannon distance between proportional vectors, i.e., vectors that normalize to identical probability distributions. In exact arithmetic, the distance between such vectors should be exactly zero. However, floating-point round-off can lead to two undesirable behaviors: (1) a tiny negative divergence that produces NaN after taking a square root, and (2) a non-negligible positive distance under float32 precision, even though the normalized distributions are identical.

This note documents the reproduction, root cause, and fix strategy.


## 2. Minimal Reproductions

All results below were obtained using SciPy 1.17.1 (conda-forge build on macOS ARM64).

### 2.1 Tiny Negative Divergence Leading to NaN

```python
import numpy as np
from scipy.spatial.distance import jensenshannon

p = np.full(10, 0.34, dtype=np.float64)
q = np.full(10, 0.42, dtype=np.float64)
print(jensenshannon(p, q))
```
Observed output:
```bash
RuntimeWarning: invalid value encountered in sqrt
nan
```
Although p and q normalize to identical probability distributions, floating-point round-off causes the internally computed Jensen–Shannon divergence to become slightly negative (on the order of 1e-16). Since the implementation directly takes the square root of this value, sqrt receives a negative input and produces NaN.

This violates the mathematical property that Jensen–Shannon divergence is non-negative.



### 2.2 Float32 Precision: Distance Not Sufficiently Close to Zero
```python
import numpy as np
from scipy.spatial.distance import jensenshannon
p = np.full(3, 0.1, dtype=np.float32)
q = np.full(3, 0.32, dtype=np.float32)
print(jensenshannon(p, q))
```

Observed output:
```bash
0.00021143198
```
In exact arithmetic, the result should be zero because the vectors are proportional and normalize to identical distributions. However, float32 precision introduces sufficient rounding error to produce a value on the order of 2e-4. While not catastrophic, this is significantly larger than typical floating-point noise and inconsistent with expected behavior for identical distributions.


## 3. Root Cause Analysis

The Jensen–Shannon distance is defined as:

JS(P, Q) = sqrt((KL(P || M) + KL(Q || M)) / 2), where M = (P + Q) / 2.

Several numerical effects contribute to instability:
	•	Subtractive cancellation in KL divergence terms.
	•	Accumulation of rounding error during normalization.
	•	Limited mantissa precision in float32.
	•	Small negative values produced due to finite precision arithmetic.
	•	No guard against negative divergence values before applying sqrt.

Although Jensen–Shannon divergence is mathematically non-negative, floating-point arithmetic does not strictly preserve that property.


## 4. Fix Strategy

The implemented fix consists of two components.

First, internal computation is promoted to float64 to improve numerical stability, even when inputs are float32. This reduces accumulated rounding error and improves KL divergence accuracy.

Second, tiny negative divergence values are clamped before taking the square root:
```python
js = np.clip(js, 0.0, None)
return np.sqrt(js / 2.0)
```
This guarantees that sqrt never receives a negative input due solely to floating-point error. The fix introduces no API changes and has negligible performance impact.


## 5. Regression Tests

Two reproducible numerical cases were added as regression tests:
	1.	Ten-dimensional proportional float64 vectors that previously produced slightly negative divergence and resulted in NaN.
	2.	Three-dimensional proportional float32 vectors that previously produced a value around 2e-4 instead of zero.

These tests ensure that proportional vectors consistently return finite values numerically close to zero under both float64 and float32 inputs. This prevents recurrence of the issue in future modifications.


## 6. Validation

The fix passes the local SciPy test suite and does not alter public API behavior. Continuous Integration builds validate cross-platform behavior across Linux, macOS, and Windows. No performance-sensitive code paths were modified beyond dtype promotion and clipping.

## 7. References

SciPy PR: https://github.com/scipy/scipy/pull/24664
Primary issue closed: gh-20083
Related discussions: gh-19436, gh-20414, gh-20786, gh-19438

