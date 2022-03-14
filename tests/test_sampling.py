from bo_protein.utils import weighted_resampling
import numpy as np


def test_weighted_resampling():
    np.random.seed(1)
    k = 1.
    scores = np.array([
        [0., 1.],
        [1., 0.],
        [2., 2.],
        [3., 1.],
    ])
    true_ranks = np.array([0, 0, 2, 2])
    weights = 1 / (k * 4 + true_ranks + 1e-6)
    true_weights = weights / weights.sum()

    ranks, weights, resampled_idxs = weighted_resampling(scores, k=k)

    assert np.all(ranks == true_ranks), f'{true_ranks} != {ranks}'
    assert np.all(weights == true_weights), f'{true_weights} != {weights}'
    assert np.all(resampled_idxs == np.array([1, 2, 0, 1]))
