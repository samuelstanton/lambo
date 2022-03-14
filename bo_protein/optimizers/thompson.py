import numpy as np
import torch
import math

from scipy.special import softmax as np_softmax

from torch.nn import functional as F

from bo_protein.optimizers.genetic import draw_candidates
from bo_protein.models import surrogates


class DiscreteMaxPosteriorSampling(torch.nn.Module):
    def __init__(self, num_evolutions, population_size, weight_pop=False, **kwargs):
        super().__init__()
        self.fantasy_model = None
        self.num_evolutions = num_evolutions
        self.population_size = population_size
        self.weight_pop = weight_pop
        # self._log_alpha = torch.nn.Parameter(torch.zeros(1))
        # self._alpha_optimizer = torch.optim.Adam([self._log_alpha], lr=1e-4)
        # self._target_entropy = 0.1

    def forward(self, X_cand: np.ndarray, num_samples, expand_shape):
        self.fantasy_model.eval()

        if isinstance(self.fantasy_model, surrogates.EmbeddingGP):
            # X_cand features will be expanded
            f_samples, _, _ = self.fantasy_model(X_cand, bs=32, num_samples=1,
                                                 expand_shape=expand_shape)
            f_samples = f_samples.squeeze(0)
        elif isinstance(self.fantasy_model, surrogates.DeepEnsemble):
            assert num_samples <= self.fantasy_model.ensemble_size
            f_samples, _, _ = self.fantasy_model(X_cand, bs=self.population_size, num_samples=num_samples)
        else:
            raise NotImplementedError('unsupported fantasy model')

        max_idxs = np.argmax(f_samples.cpu(), axis=-1)
        X_max = np.tile(X_cand, (num_samples, *X_cand.shape))
        X_max = X_max[np.arange(num_samples), max_idxs]
        f_samples = f_samples[np.arange(num_samples), max_idxs]
        return X_max, f_samples

    def optimize(self, X, Y, model, num_samples: int = 1, log_prefix=''):
        self.fantasy_model = model
        expand_shape = [num_samples]

        X_trajec, f_trajec = [], []
        for i in range(self.num_evolutions):
            if self.weight_pop and i == 0:
                weights = self._get_weights(Y)
            else:
                weights = None
            _, X_cand = draw_candidates(X, self.population_size, weights)

            with torch.no_grad():
                X, f = self(X_cand, num_samples, expand_shape)

            # condition each sample path on the last step
            if hasattr(self.fantasy_model, 'condition_on_observations'):
                self.fantasy_model = self.fantasy_model.condition_on_observations(X, f.view(-1, 1, 1))
                expand_shape = [num_samples]

            X_trajec.append(np.copy(X))
            f_trajec.append(np.copy(f.cpu()))

        X_trajec = np.stack(X_trajec, axis=1)  # [num_samples x num_steps]
        f_trajec = np.stack(f_trajec, axis=1)  # [num_samples x num_steps]

        max_idxs = np.argmax(f_trajec, axis=1)
        X_best = X_trajec[np.arange(num_samples), max_idxs]
        f_best = f_trajec[np.arange(num_samples), max_idxs]

        self.fantasy_model = None  # don't keep the model around to save memory
        return X_best, f_best

    def _get_weights(self, Y):
        return np_softmax(Y.reshape(-1))
        # Y = Y if torch.is_tensor(Y) else torch.tensor(Y)
        # weights = F.softmax(Y.view(-1) / self._log_alpha.exp())
        # weights = F.softmax(Y.view(-1))
        # entropy = -(weights * weights.log()).mean()
        # alpha_loss = (entropy - self._target_entropy) ** 2
        # alpha_loss.backward()
        # self._alpha_optimizer.step()
        # return weights.detach().cpu().numpy()
