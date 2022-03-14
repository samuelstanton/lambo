import itertools
import torch
import torch.nn as nn
import numpy as np
import wandb
import abc

from collections import OrderedDict

from sklearn.model_selection import train_test_split

from torch.nn import functional as F

from scipy.stats import spearmanr

from botorch.models import SingleTaskGP, SingleTaskVariationalGP

from gpytorch.mlls import ExactMarginalLogLikelihood

import bo_protein.utils
from bo_protein.models.surrogates import SurrogateModel
from bo_protein.gfp_data import utils as gfp_utils
from bo_protein.gfp_data import transforms as gfp_transforms
from bo_protein.models.trainer import quantile_calibration, check_early_stopping


class BOWFeaturizer:
    def __init__(self, alphabet, max_feature_len=3):
        self.alphabet = alphabet
        self.max_feature_len = max_feature_len
        self._get_features()

    def _get_features(self):
        tuples = []
        for i in range(1, self.max_feature_len + 1):
            tuples.extend(
                list(itertools.combinations_with_replacement(self.alphabet, i))
            )

        features = []
        for t in tuples:
            features.append("".join(list(t)))

        self.features = features

        self.num_features = len(self.features)

    def featurize(self, X):
        batch_size = X.shape[0]
        representations = np.zeros((batch_size, self.num_features))
        for i in range(batch_size):
            for j in range(self.num_features):
                representations[i][j] = len(
                    re.findall("(?={})".format(self.features[j]), X[i])
                )

        return representations


class BOWRBFGP:
    def __init__(self, alphabet):
        self.featurizer = BOWFeaturizer(alphabet, max_feature_len=2)

    def _preprocess_x(self, X):
        X = self.featurizer.featurize(X)
        X = torch.from_numpy(X).double()

        self._min, self._max = X.min(0)[0], X.max(0)[0]
        X = (X - self._min) / torch.where(
            self._max != self._min, self._max - self._min, 1.0
        )
        X = 2 * X - 1

        return X

    def _preprocess_y(self, Y):
        Y = Y.reshape(-1, 1)
        Y = torch.from_numpy(Y).double()
        return Y

    def train(self, X, Y):
        X = self._preprocess_x(X)
        Y = self._preprocess_y(Y)
        self.gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        return self

    def forward(self, X):
        X = self._preprocess_x(X)

        with torch.no_grad():
            mvn = self.gp(X)

        return mvn


class SSKGP:
    def __init__(self, alphabet, max_len):
        self.alphabet = alphabet
        self.max_len = max_len

    def train(self, X, Y):
        from boss.code.parameters.string_parameter import StringParameter
        from boss.code.emukit_models.emukit_ssk_model import SSK_model
        from emukit.core import ParameterSpace

        X_pad = np.array([string.ljust(self.max_len, "#") for string in X])
        X_sep = np.array([" ".join(string.strip()) for string in X_pad]).reshape(-1, 1)

        Y = Y.reshape(-1, 1)

        space = ParameterSpace(
            [StringParameter("string", length=2 * self.max_len, alphabet=self.alphabet)]
        )
        self.model = SSK_model(space, X_sep, Y, num_splits=25, max_subsequence_length=3)
        return self

    def forward(self, X):
        X_pad = np.array([string.ljust(self.max_len, "#") for string in X])
        X_sep = np.array([" ".join(string.strip()) for string in X_pad]).reshape(-1, 1)

        mean, var = self.model.predict(X_sep)

        return mean, var


# import gpflow
# from gpflow.mean_functions import Constant
# from gpflow import set_trainable
# from gpflow.utilities import positive

# from boss.code.GPflow_wrappers.Batch_SSK import Batch_SSK

# k = Batch_SSK(batch_size=3000,gap_decay=0.99,match_decay=0.53,alphabet=alphabet,max_subsequence_length = max_subsequence_length, maxlen=85)
# cst = gpflow.kernels.Constant(1.77)
# m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(0.2), kernel= cst*k, noise_variance=0.003)
# loss=m.log_marginal_likelihood()

# # fit model (turned off for quick demo, good hyper-parameters are already selected)
# optimizer = gpflow.optimizers.Scipy()
# optimizer.minimize(m.training_loss , m.trainable_variables,options=dict(ftol=0.0001),compile=False)
