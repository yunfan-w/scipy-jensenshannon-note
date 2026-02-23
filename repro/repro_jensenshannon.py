import numpy as np
from scipy.spatial.distance import jensenshannon

def case_negative_divergence():
    # proportional vectors -> normalize to identical distributions
    p = np.full(10, 0.34, dtype=np.float32)
    q = np.full(10, 0.42, dtype=np.float32)
    d = jensenshannon(p, q)
    print("float64 10-d:", d)

def case_float32_not_small_enough():
    p = np.full(3, 0.1, dtype=np.float32)
    q = np.full(3, 0.32, dtype=np.float32)
    d = jensenshannon(p, q)
    print("float32 3-d:", d)

if __name__ == "__main__":
    case_negative_divergence()
    case_float32_not_small_enough()