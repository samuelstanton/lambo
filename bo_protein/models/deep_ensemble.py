import math

import numpy as np
import torch
import wandb
from botorch.posteriors import Posterior
from scipy.stats import spearmanr
from torch import optim as optim, nn as nn
from torch.utils.data import DataLoader

import bo_protein.utils
from bo_protein import dataset as dataset_util
from bo_protein.models.base_surrogate import BaseSurrogate
from bo_protein.models.shared_elements import check_early_stopping
from bo_protein.models.surrogates import model_dict
from bo_protein.transforms import padding_collate_fn
from bo_protein.utils import to_cuda


class DeepEnsemble(BaseSurrogate):
    def __init__(self, tokenizer, model, model_kwargs, lr, bs, weight_decay,
                 ensemble_size, max_shift, mask_size, num_epochs, eval_period,
                 bootstrap_ratio, holdout_ratio, early_stopping, patience, min_num_train, **kwargs):
        super().__init__()

        self._set_transforms(tokenizer, max_shift, mask_size)

        network_constr = model_dict[model]
        model_kwargs.update(dict(tokenizer=tokenizer))

        trainer = EnsembleComponentTrainer(
            network_constr,
            network_kwargs=model_kwargs,
            dict_size=len(tokenizer.full_vocab),
            lr=lr,
            bs=bs,
            weight_decay=weight_decay,
        )

        self.trainer = trainer
        self.ensemble_size = ensemble_size
        self.models = torch.nn.ModuleList([
            network_constr(**model_kwargs) for _ in range(self.ensemble_size)
        ])

        self.num_epochs = num_epochs
        self.bootstrap_ratio = bootstrap_ratio
        self.holdout_ratio = holdout_ratio
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_num_train = min_num_train
        self.num_outputs = model_kwargs['out_dim']  # BoTorch acquisition compatibility
        self.eval_period = eval_period

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, reset=False, log_prefix="deep_ens", **kwargs):
        super().fit(X_train, Y_train)
        if isinstance(Y_train, np.ndarray):
            Y_train = torch.from_numpy(Y_train).float()
        if isinstance(Y_test, np.ndarray):
            Y_test = torch.from_numpy(Y_test).float()

        print(f'{X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test')

        _train_dataset, _test_dataset = self._get_datasets(X_train, X_test, Y_train, Y_test)
        _, _val_dataset = self._get_datasets(X_train, X_val, Y_train, Y_val)
        prev_models = self.models
        new_models = torch.nn.ModuleList()
        for i in range(self.ensemble_size):
            # if bootstrapping, each component gets its own independent bootstrap
            if self.bootstrap_ratio is not None:
                X_train, Y_train = _train_dataset.tensors
                new_X, new_Y = bo_protein.utils.draw_bootstrap(
                    X_train, Y_train, bootstrap_ratio=self.bootstrap_ratio, min_samples=self.min_num_train
                )
                train_dataset = dataset_util.TransformTensorDataset(
                    [new_X, new_Y], self.train_transform
                )
            else:
                train_dataset = _train_dataset

            # if reset is False, attempt to resume training existing component
            if not reset and len(prev_models) > i:
                self.trainer.model = prev_models[i]
            # otherwise a new component will be created
            elif len(prev_models) <= i:
                reset = True

            _log_prefix = f"{log_prefix}/model_{i}"
            self.trainer.train(
                train_dataset,
                _val_dataset,
                _test_dataset,
                num_epochs=self.num_epochs,
                eval_period=self.eval_period,
                reset=reset,
                log_prefix=_log_prefix,
                early_stopping=self.early_stopping,
                stopping_patience=self.patience
            )
            if reset:
                new_models.append(self.trainer.model)
            else:
                new_models.append(prev_models[i])
        self.models = new_models

        # evaluate full ensemble
        metrics = dict(best_epoch=0)
        metrics.update(self.evaluate(X_test, Y_test, bs=32, log_prefix=log_prefix, split="test"))
        records = [metrics]

        return records

    def predict(self, X, num_samples, *args, **kwargs):
        return self(X, num_samples=num_samples)

    def posterior(self, X):
        return EnsemblePosterior(X, self)

    def forward(self, X, Y=None, bs=128, num_samples=None, **kwargs):
        for model in self.models:
            model.eval()

        ens_size = self.ensemble_size
        num_samples = ens_size if num_samples is None else num_samples
        if num_samples > ens_size:
            raise ValueError(
                f'[DeepEnsemble] a {ens_size}-component deep ensemble can do at most {ens_size} function draws'
            )
        sample_idxs = np.random.permutation(ens_size)[:num_samples]

        if Y is not None:
            if isinstance(Y, np.ndarray):
                Y = torch.from_numpy(Y).float()

            dataset = dataset_util.TransformTensorDataset([X, Y], self.test_transform)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=bs, collate_fn=padding_collate_fn
            )

            y_hat = []
            for x_batch, _ in loader:
                if torch.cuda.is_available():
                    x_batch = x_batch.to("cuda")

                with torch.no_grad():
                    _y_hat = torch.stack(
                        [model(x_batch) for model in self.models], dim=0
                    )

                y_hat.append(_y_hat)

            y_hat = torch.cat(y_hat, dim=1)[sample_idxs]
            return (y_hat[sample_idxs], y_hat.mean(0), y_hat.std(0)), Y

        else:
            dataset = dataset_util.TransformTensorDataset([X], self.test_transform)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=bs, collate_fn=padding_collate_fn
            )

            y_hat = []
            for x_batch in loader:
                if torch.cuda.is_available():
                    x_batch = x_batch.to("cuda")

                with torch.no_grad():
                    _y_hat = torch.stack(
                        [model(x_batch) for model in self.models], dim=0
                    )

                y_hat.append(_y_hat)

            y_hat = torch.cat(y_hat, dim=1)[sample_idxs]
            return (y_hat, y_hat.mean(0), y_hat.std(0))


class EnsemblePosterior(Posterior):
    def __init__(self, X, model):
        super().__init__()
        self.X = X
        self.model = model

    @property
    def device(self):
        return list(self.model.parameters())[0].device

    @property
    def dtype(self):
        return list(self.model.parameters())[0].dtype

    @property
    def event_shape(self):
        return self.X.shape[:-1] + (1,)

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        if self.X.ndim == 1:
            y_outputs, _, _ = self.model(self.X, num_samples=sample_shape[0])
            return y_outputs
        elif self.X.ndim == 2:
            y_output_list = []
            for x in self.X:
                y_outputs, _, _ = self.model(x, num_samples=sample_shape[0])
                y_output_list.append(y_outputs)
            # TODO: fix the hard-coded permute here
            return torch.stack(y_output_list).permute(1, 0, 2, 3)

    @property
    def mean(self):
        return self.model(self.X)[1]

    @property
    def variance(self):
        return self.model(self.X)[2]


class EnsembleComponentTrainer(object):
    def __init__(self, network, dict_size, network_kwargs={}, lr=1e-3, bs=100, weight_decay=0.0):
        self.network = network
        self.dict_size = dict_size

        self.lr = lr
        self.bs = bs
        self.weight_decay = weight_decay

        self.model = self.network(**network_kwargs)
        self.network_kwargs = network_kwargs
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )

    def train(
        self,
        train_dataset,
        holdout_dataset=None,
        eval_dataset=None,
        num_epochs=100,
        eval_period=1,
        reset=True,
        log_prefix="",
        early_stopping=False,
        stopping_patience=64,
    ):
        if reset:
            self.model = self.network(dict_size=self.dict_size, **self.network_kwargs)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=math.ceil(stopping_patience / 2.), threshold=1e-3
        )

        collate_fn = lambda x: padding_collate_fn(x, self.model.tokenizer.padding_idx)
        loader = DataLoader(
            train_dataset,
            batch_size=self.bs,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        criterion = nn.MSELoss()
        best_score, best_epoch, best_weights, stop = None, 0, None, False

        # print(f"train_dataset: {len(train_dataset)}")
        best_loss, best_loss_epoch = None, 0
        best_score, best_score_epoch = None, 0
        stop_crit_key = select_crit_key = 'val_rmse'
        stop = False
        for epoch_idx in range(num_epochs):
            metrics = {}
            self.model.train()
            avg_train_loss = 0.0
            for x, y in loader:
                x, y = to_cuda((x, y))
                self.optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                avg_train_loss += loss.item() / len(loader)

            metrics.update({
                "epoch": epoch_idx + 1,
                "train_loss": avg_train_loss,
            })
            lr_sched.step(avg_train_loss)

            if (epoch_idx + 1) % eval_period == 0:
                val_rmse, _ = self.evaluate(holdout_dataset)
                metrics.update(dict(val_rmse=val_rmse))

            select_crit = metrics.get(select_crit_key, None)
            if early_stopping and select_crit is not None:
                best_score, best_score_epoch, best_weights, _ = check_early_stopping(
                    model=self.model,
                    best_score=best_score,
                    best_epoch=best_score_epoch,
                    best_weights=best_weights,
                    curr_score=select_crit,
                    curr_epoch=epoch_idx + 1,
                    patience=stopping_patience,
                    save_weights=True,
                )
                metrics.update(dict(best_score=best_score, best_epoch=best_score_epoch))

            # use train loss to determine convergence
            stop_crit = metrics.get(stop_crit_key, None)
            if stop_crit is not None:
                best_loss, best_loss_epoch, _, stop = check_early_stopping(
                    model=self.model,
                    best_score=best_loss,
                    best_epoch=best_loss_epoch,
                    best_weights=None,
                    curr_score=stop_crit,
                    curr_epoch=epoch_idx + 1,
                    patience=stopping_patience,
                    save_weights=False,
                )
                metrics.update(dict(best_loss=best_loss, best_loss_epoch=best_loss_epoch))

            if len(log_prefix) > 0:
                metrics = {'/'.join((log_prefix, key)): val for key, val in metrics.items()}
            try:
                wandb.log(metrics)
            except Exception as e:
                pass

            if stop:
                break

        if early_stopping:
            assert holdout_dataset is not None
            self.model.load_state_dict(best_weights)

        self.model.eval()
        return self.model

    def evaluate(self, dataset):
        loader = DataLoader(
            dataset, batch_size=self.bs, shuffle=False, collate_fn=padding_collate_fn
        )

        Y_hat = []
        Y_test = []
        self.model.eval()
        for x, y in loader:
            x, y = to_cuda((x, y))

            with torch.no_grad():
                Y_hat.append(self.model(x))

            Y_test.append(y)

        if len(Y_hat) >= 1:
            Y_hat = torch.cat(Y_hat, dim=0).cpu().numpy()
            Y_test = torch.cat(Y_test, dim=0).cpu().numpy()

            rmse = np.sqrt(np.mean(np.power(Y_test - Y_hat, 2))).item()
            sr = spearmanr(Y_test, Y_hat).correlation
        else:
            rmse = None
            sr = None

        return rmse, sr