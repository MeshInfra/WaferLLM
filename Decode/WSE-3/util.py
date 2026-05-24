import sys
from pathlib import Path
project_root = Path(__file__).resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import numpy as np

def calculate_ulp_distance_fp32(a, b):
    """
    Vectorized ULP distance calculation for arrays

    Args:
        a, b: numpy arrays of float32

    Returns:
        numpy array of ULP distances
    """
    # Handle non-finite values
    valid_mask = np.isfinite(a) & np.isfinite(b)
    ulp_distances = np.full_like(a, fill_value=np.inf, dtype=np.float64)

    # Handle exact matches
    exact_match = a == b
    ulp_distances[exact_match] = 0.0

    # Calculate ULP for valid, non-matching values
    calc_mask = valid_mask & ~exact_match

    if np.any(calc_mask):
        # Convert to uint32 view for bit manipulation
        a_bits = np.frombuffer(a[calc_mask].tobytes(), dtype=np.uint32)
        b_bits = np.frombuffer(b[calc_mask].tobytes(), dtype=np.uint32)

        # Calculate ULP distance
        ulp_distances[calc_mask] = np.abs(
            a_bits.astype(np.int64) - b_bits.astype(np.int64)
        ).astype(np.float64)

    return ulp_distances

def calculate_ulp_distance_fp16(a, b):
    """
    Vectorized ULP distance calculation for float16 arrays

    Args:
        a, b: numpy arrays of float16

    Returns:
        numpy array of ULP distances (float64)
    """
    a = np.asarray(a, dtype=np.float16)
    b = np.asarray(b, dtype=np.float16)

    valid_mask = np.isfinite(a) & np.isfinite(b)
    ulp_distances = np.full(a.shape, fill_value=np.inf, dtype=np.float64)

    exact_match = a == b
    ulp_distances[exact_match] = 0.0

    calc_mask = valid_mask & ~exact_match

    if np.any(calc_mask):
        a_bits = np.frombuffer(a[calc_mask].tobytes(), dtype=np.uint16)
        b_bits = np.frombuffer(b[calc_mask].tobytes(), dtype=np.uint16)

        ulp_distances[calc_mask] = np.abs(
            a_bits.astype(np.int32) - b_bits.astype(np.int32)
        ).astype(np.float64)

    return ulp_distances