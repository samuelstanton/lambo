import torch


class ExpectedObjVal(object):
    def __init__(self, surrogate, num_samples, obj_dim, **kwargs):
        self.surrogate = surrogate
        self.num_samples = num_samples
        self.out_dim = obj_dim
        self.batch_size = 1

    def __call__(self, candidates):
        if torch.is_tensor(candidates):
            pass

        if candidates.shape[-1] == 1:
            candidates = candidates.squeeze(-1)

        pred_samples = self.surrogate.posterior(candidates).rsample(torch.Size([self.num_samples]))

        return pred_samples.mean(0)
