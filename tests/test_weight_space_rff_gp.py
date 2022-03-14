#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import torch
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.testing import BotorchTestCase
from botorch_fb.experimental.models import ApproximateGPyTorchModel, RFFWeightSpaceModel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO


class TestRFFWeightSpaceGP(BotorchTestCase):
    def setUp(self):
        super().setUp()
        train_x = torch.rand(10, 1)
        train_y = torch.sin(train_x) + torch.randn(train_x.size()) * 0.2

        train_x = train_x.to(device=self.device)
        train_y = train_y.to(device=self.device)

        init_model = RFFWeightSpaceModel()
        self.model = ApproximateGPyTorchModel(
            model=init_model, likelihood=GaussianLikelihood()
        )

        mll = VariationalELBO(self.model.likelihood, self.model.model, num_data=10)
        loss = -mll(self.model.likelihood(self.model(train_x)), train_y).sum()
        loss.backward()

    def test_posterior(self):
        test_x = torch.rand(30).unsqueeze(1).to(device=self.device)
        posterior = self.model.posterior(test_x)
        self.assertIsInstance(posterior, GPyTorchPosterior)

    def test_batched_posterior(self):
        train_x = torch.rand(10, 1, device=self.device)
        train_y = torch.randn(10, 3, device=self.device)

        batched_gpytorch_model = RFFWeightSpaceModel(batch_shape=torch.Size((3,))).to(
            self.device
        )
        batched_model = ApproximateGPyTorchModel(model=batched_gpytorch_model).to(
            self.device
        )
        mll = VariationalELBO(
            batched_model.likelihood, batched_gpytorch_model, num_data=10
        )

        with torch.enable_grad():
            loss = -mll(
                batched_model.likelihood(batched_gpytorch_model(train_x)), train_y.t()
            ).sum()
            loss.backward()

        # but that the covariance does have a gradient
        self.assertIsNotNone(batched_model.model.covar_module.raw_lengthscale.grad)
