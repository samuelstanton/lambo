import numpy as np
import torch

from boss.code.GPflow_wrappers.Batch_SSK import Batch_SSK
from bo_protein.models.sskernel import SSKernel

def test_setup(num_strs = 5, str_len = 85, maxdepth=8):
    alph_size = 4

    random_strs = np.array(["A", "C", "T", "G"])
    inputs = np.array(
        [np.random.choice(random_strs, size=(str_len,)) for _ in range(num_strs)]
    )

    inputs_as_match = torch.from_numpy(
        np.stack([np.stack([inputs[j] == random_strs[i] for i in range(alph_size)]).T for j in range(num_strs)])
    )

    kernel = SSKernel(max_depth = maxdepth, match_decay = 0.53, gap_decay = 0.99)
    kernel.lengthscale = 1.0

    gp_kernel = kernel(inputs_as_match, inputs_as_match)
    norm_kmat = gp_kernel.evaluate()

    kernel = Batch_SSK(
        alphabet=list(random_strs), 
        gap_decay=0.99, match_decay=0.53, 
        maxlen=str_len, 
        max_subsequence_length=maxdepth
    )
    inputs_to_kern = np.array(["".join(list(x)) for x in inputs])
    X_train = np.array([" ".join(list(x)) for x in inputs_to_kern])[:,None]
    true_kernel = kernel.K(X_train).numpy()
    print(torch.norm(torch.from_numpy(true_kernel) - norm_kmat) < 1e-4, "testing kernel closeness")

test_setup()