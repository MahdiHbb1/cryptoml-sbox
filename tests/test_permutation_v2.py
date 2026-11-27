import numpy as np
import pytest

from src.permutation_v2 import initialize, mutate


def test_initialization_is_seeded_permutation():
    first = initialize(np.random.default_rng(42))
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(np.sort(first), np.arange(256))
    np.testing.assert_array_equal(first, initialize(np.random.default_rng(42)))


@pytest.mark.parametrize(('operator', 'changed'), [('single_swap', 2), ('multi_swap', 4), ('three_cycle', 3), ('segment_reversal', 4)])
def test_mutation_changes_candidate_without_touching_parent(operator, changed):
    parent = np.arange(256, dtype=np.uint8)
    child = mutate(parent, operator, np.random.default_rng(42), intensity=2 if operator == 'multi_swap' else 4 if operator == 'segment_reversal' else None)
    np.testing.assert_array_equal(parent, np.arange(256))
    np.testing.assert_array_equal(np.sort(child), parent)
    assert child.dtype == np.uint8
    assert np.count_nonzero(child != parent) == changed


def test_invalid_parent_is_rejected():
    with pytest.raises(ValueError):
        mutate(np.zeros(256, dtype=np.uint8), 'single_swap', np.random.default_rng(42))
