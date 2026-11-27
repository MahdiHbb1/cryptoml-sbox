"""Polynomial-free initialization and bijective mutations for issue #17."""

import numpy as np


def initialize(rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(256).astype(np.uint8)


def mutate(
    parent: np.ndarray,
    operator: str,
    rng: np.random.Generator,
    intensity: int | None = None,
) -> np.ndarray:
    if parent.shape != (256,) or parent.dtype != np.uint8 or np.unique(parent).size != 256:
        raise ValueError("Expected a uint8 permutation of 0–255")
    child = parent.copy()
    if operator == "single_swap":
        positions = rng.choice(256, 2, replace=False)
        child[positions] = child[positions[::-1]]
    elif operator == "multi_swap":
        count = 2 if intensity is None else intensity
        if not isinstance(count, (int, np.integer)) or not 2 <= count <= 128:
            raise ValueError("multi_swap intensity must be an integer in [2, 128]")
        pairs = rng.choice(256, 2 * count, replace=False).reshape(-1, 2)
        child[pairs] = child[pairs[:, ::-1]]
    elif operator == "three_cycle":
        positions = rng.choice(256, 3, replace=False)
        child[positions] = child[np.roll(positions, 1)]
    elif operator == "segment_reversal":
        length = 4 if intensity is None else intensity
        if not isinstance(length, (int, np.integer)) or not 2 <= length <= 256:
            raise ValueError("segment_reversal intensity must be an integer in [2, 256]")
        start = int(rng.integers(257 - length))
        segment = slice(start, start + length)
        child[segment] = child[segment][::-1]
    else:
        raise ValueError(f"Unknown mutation operator: {operator}")
    return child
