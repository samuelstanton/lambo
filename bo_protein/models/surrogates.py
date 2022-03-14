import sys

import bo_protein.utils

sys.path.append("../../scripts")
import copy
import tqdm
import torch
import numpy as np
import torchvision
import math
import hydra
import wandb
import gpytorch

# from typing import Sequence
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC, SVR
# from sklearn.gaussian_process import GaussianProcessRegressor
# from xgboost import XGBRegressor
# from tape import TAPETokenizer, ProteinBertModel, ProteinBertConfig
from tape import ProteinBertConfig
from scipy.stats import spearmanr
import botorch

from . import nn_models
from . import finetune
from . import transformers
from .trainer import Trainer, EmbeddingTrainer, quantile_calibration

from ..gfp_data import utils, transforms
from ..gfp_data import dataset as dataset_util
from ..gfp_data.transforms import padding_collate_fn

model_dict = {
    "CNN": nn_models.CNN,
    "mCNN": nn_models.mCNN,
    "TCN": nn_models.TCN,
    "Transformer": nn_models.Transformer,
    "RNN": nn_models.RNN,
    "MLP": nn_models.MLP
}


class SurrogateModel(torch.nn.Module):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def _set_transforms(self, tokenizer, max_shift, mask_size):
        # convert from string to LongTensor of token indexes
        # don't use the max_len arg here, will interfere with the translation augmentation
        train_transform = [transforms.StringToLongTensor(tokenizer, max_len=None)]
        # randomly substitute masking tokens
        if mask_size > 0:
            train_transform.append(
                transforms.RandomMask(mask_size, tokenizer.masking_idx, contiguous=False)
            )
        # random cycle rotation of the sequence
        if max_shift > 0:
            train_transform.append(
                transforms.SequenceTranslation(max_shift)
            )
        train_transform = torchvision.transforms.Compose(train_transform)

        # no data augmentation at test-time
        test_transform = transforms.StringToLongTensor(tokenizer)

        self.train_transform = train_transform
        self.test_transform = test_transform

    def _get_datasets(self, X_train, X_test, Y_train, Y_test):
        if isinstance(Y_train, np.ndarray):
            Y_train = torch.from_numpy(Y_train).float()
        if isinstance(Y_test, np.ndarray):
            Y_test = torch.from_numpy(Y_test).float()

        train_dataset = dataset_util.TransformTensorDataset(
            [X_train, Y_train], self.train_transform
        )

        val_dataset = dataset_util.TransformTensorDataset(
            [X_test, Y_test], self.test_transform
        )

        return train_dataset, val_dataset

    def fit(self, X_train, Y_train, *args, **kwargs):
        self.train_inputs = X_train
        self.train_targets = Y_train

    def evaluate(self, X, Y, bs, log_prefix="", split=""):
        self.eval()
        (_, mean, std), labels = self(X, Y, bs=bs)
        try:
            assert mean.shape == labels.shape
        except AssertionError:
            import pdb; pdb.set_trace()
        mean, std, labels = mean.cpu(), std.cpu(), labels.cpu()
        nll = -torch.distributions.Normal(mean, std).log_prob(labels).mean()
        ece = quantile_calibration(mean, std, labels)["ece"]

        if mean.ndim == 1:
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)
            
        spearman_rho = 0
        for idx in range(labels.size(-1)):
            spearman_rho += spearmanr(labels[..., idx], mean[..., idx]).correlation / labels.size(-1)

        # metrics = {
        #     "/".join([log_prefix, f"{split}_nll"]): nll.item(),
        #     "/".join([log_prefix, f"{split}_rmse"]): np.sqrt(np.power(mean - labels, 2).mean()).item(),
        #     "/".join([log_prefix, f"{split}_s_rho"]): spearman_rho,
        #     "/".join([log_prefix, f"{split}_ece"]): ece,
        #     "/".join([log_prefix, f"{split}_post_var"]): (std ** 2).mean().item()
        # }

        metrics = {
            f"{split}_nll": nll.item(),
            f"{split}_rmse": np.sqrt(np.power(mean - labels, 2).mean()).item(),
            f"{split}_s_rho": spearman_rho,
            f"{split}_ece": ece,
            f"{split}_post_var": (std ** 2).mean().item()
        }

        if len(log_prefix) > 0:
            metrics = {'/'.join((log_prefix, key)): val for key, val in metrics.items()}
        try:
            wandb.log(metrics)
        except:
            pass

        return metrics


class DeepEnsemble(SurrogateModel):
    def __init__(self, tokenizer, model, model_kwargs, lr, bs, weight_decay,
                 ensemble_size, max_shift, mask_size, num_epochs, eval_period,
                 bootstrap_ratio, holdout_ratio, early_stopping, patience, min_num_train, **kwargs):
        super().__init__()
    
        self._set_transforms(tokenizer, max_shift, mask_size)

        network_constr = model_dict[model]
        model_kwargs.update(dict(tokenizer=tokenizer))

        trainer = Trainer(
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

            # create holdout dataset for early stopping
            # if self.holdout_ratio > 0.0:
            #     num_holdout = math.ceil(len(_train_dataset) * self.holdout_ratio)
            #     num_train = len(_train_dataset) - num_holdout
            #     train_dataset, holdout_dataset = _train_dataset.random_split(
            #         num_train, num_holdout
            #     )
            # else:
            #     holdout_dataset = None

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
        from .ensemble_posterior import EnsemblePosterior
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

            # y_hat = torch.cat(y_hat, dim=-1).cpu().numpy()[sample_idxs]
            # y_gt = Y.cpu().numpy()
            # return (y_hat, y_hat.mean(0), y_hat.std(0)), y_gt

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

            # y_hat = torch.cat(y_hat, dim=1).cpu().numpy()[sample_idxs]
            y_hat = torch.cat(y_hat, dim=1)[sample_idxs]
            return (y_hat, y_hat.mean(0), y_hat.std(0))


class EmbeddingSurrogate(SurrogateModel):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embedding_type = config.get("embedding_type", "ESM")
        if self.embedding_type == "ESM":
            wrapper = transformers.ESMWrapper(config)
        elif self.embedding_type == "BERT":
            wrapper = transformers.BERTWrapper(config)

        self.tokenizer = wrapper
        self.model = wrapper

        finetune = config.get("finetune", None)
        self.do_unsupervised_finetune = False
        self.do_supervised_finetune = False
        if finetune:
            self.do_unsupervised_finetune = finetune == 'unsupervised'
            self.do_supervised_finetune = finetune == 'supervised'

        self.embedding_cache = {}

        self._set_transforms(config, self.tokenizer)
        
        projection = torch.nn.Linear(self.model.embedding_size, 10, bias=False)
        self.emb_projection = projection.cuda() if torch.cuda.is_available() else projection

    def unsupervised_finetune(self, X, reset=False, num_epochs=10, steps_cutoff=float("inf")):
        assert self.embedding_type == "BERT"

        model_config = ProteinBertConfig.from_pretrained("bert-base")
        mlm = finetune.TransformerForMaskedLM(model_config, self.model)

        mlm.train()
        if torch.cuda.is_available():
            mlm = mlm.cuda()

        dataset = finetune.MaskedLanguageModelingDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

        opt = torch.optim.Adam(mlm.parameters(), lr=1e-4)

        steps = 0
        for _ in range(num_epochs):
            if steps > steps_cutoff:
                break

            for x in tqdm.tqdm(loader):

                if torch.cuda.is_available():
                    x = tuple([_x.cuda() for _x in x])

                (loss, perplexity), *other = mlm(*x)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if steps > 0 and steps % 100 == 0:
                    print(loss.item())

                steps += 1

        self.model = mlm.transformer

        return self

    def supervised_finetune(self, X, Y, reset=False, bs=32, steps_cutoff=float("inf"), log_prefix='emb_sup_finetune'):
        model_config = ProteinBertConfig.from_pretrained("bert-base")
        model_config.input_size = self.model.embedding_size
        model_config.hidden_size = 256
        model_kwargs = self.config.get('model_kwargs', {})
        model_kwargs.setdefault('p', 0.)
        net = finetune.TransformerForValuePrediction(model_config, self.model, **model_kwargs)

        # print(net)

        if torch.cuda.is_available():
            net = net.cuda()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.config['holdout_ratio'])

        train_dataset, holdout_dataset = self._get_datasets(X_train, X_test, Y_train, Y_test)

        num_train = len(X)
        collate_fn = lambda x: padding_collate_fn(x, padding_value=self.tokenizer.padding_idx)

        if self.config['holdout_ratio'] > 0.0:
            holdout_loader = torch.utils.data.DataLoader(holdout_dataset, batch_size=bs,
                                                                          collate_fn=collate_fn)
        else:
            holdout_loader = None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                                                                 collate_fn=collate_fn)

        param_groups = [
            dict(params=net.transformer.parameters(), lr=1e-5, weight_decay=self.config.get('weight_decay', 1e-4)),
            dict(params=net.predict.parameters(), lr=1e-3, weight_decay=self.config.get('weight_decay', 1e-4))
        ] 
        opt = torch.optim.Adam(param_groups)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config['num_epochs'], eta_min=1e-6)
        steps = 0
        best_score, best_epoch, best_weights, stop = None, 0, None, False
        print(f'[finetuning] training for a max of {self.config["num_epochs"]} epoch(s) on {num_train} examples')
        for epoch in range(self.config['num_epochs']):
            if steps > steps_cutoff:
                break
            train_loss = 0
            net.train()
            for x, y in train_loader:

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                loss, *other = net(x, x.ne(0), y[:, None])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                steps += 1
                train_loss += loss.item() / len(train_loader)

            if holdout_loader is not None:
                net.eval()
                holdout_loss = 0.
                for x, y in holdout_loader:
                    x = x.cuda() if torch.cuda.is_available() else x
                    y = y.cuda() if torch.cuda.is_available() else y
                    with torch.no_grad():
                        loss, *other = net(x, x.ne(0), y[:, None])
                    holdout_loss += loss.item() / len(holdout_loader)
                best_score, best_epoch, best_weights, stop = self.early_stopping(
                    net, best_score, best_epoch, best_weights, holdout_loss, epoch,
                    patience=self.config.get('patience', 64)
                )

            metrics = {
                "/".join([log_prefix, "train_loss"]): train_loss,
                "/".join([log_prefix, "epoch"]): epoch + 1,
                '/'.join([log_prefix, 'holdout_loss']): holdout_loss,
                '/'.join([log_prefix, 'lr']): lr_sched.get_last_lr()[-1]
            }
            lr_sched.step()
            try:
                wandb.log(metrics)
            except Exception as e:
                pass

            if self.config['early_stopping'] and stop:
                assert holdout_loader is not None
                # print(f'stopped model training after {epoch + 1} epoch(s)')
                # print(f'loading checkpoint from {best_epoch} with score {best_score:0.4f}.')
                net.load_state_dict(best_weights)
                break

        self.transformer = net.transformer
        self.emb_projection = net.predict.projection

        return self

    def early_stopping(
        self,
        module,
        best_score,
        best_epoch,
        best_weights,
        curr_score,
        curr_epoch,
        patience,
        tol=1e-3,
    ):
        eps = 1e-6
        stop = False
        if (
            best_score is None
            or (best_score - curr_score) / (abs(best_score) + eps) > tol
        ):
            best_score, best_epoch = curr_score, curr_epoch
            best_weights = copy.deepcopy(module.state_dict())
        elif curr_epoch - best_epoch >= patience:
            stop = True
        return best_score, best_epoch, best_weights, stop

    def _get_bert_embeddings(self, X, bs=64):
        if isinstance(X, list) and all(isinstance(x, str) for x in X):
            inputs = torch.LongTensor(
                finetune.pad_sequences([np.array(self.tokenizer.encode(x)) for x in X],
                                       self.tokenizer.padding_idx)
            )
        elif isinstance(X, list) and all(isinstance(x, torch.Tensor) for x in X):
            inputs = torch.stack(X)
        else:
            raise RuntimeError('X should be an np.ndarray of strings or a list of tensors')

        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        embeddings = []
        for i in range((len(X) // bs) + 1):

            if i * bs >= len(X):
                break

            _inputs = inputs[i * bs : (i + 1) * bs]
            if torch.cuda.is_available():
                _inputs = _inputs.cuda()

            with torch.no_grad():
                embed = self.model(_inputs)[1]
            embeddings.append(embed)

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def _get_esm_embeddings(self, X, bs=64):
        if isinstance(X, list) and all(isinstance(x, str) for x in X):
            inputs = torch.LongTensor(
                finetune.pad_sequences([np.array(self.tokenizer.encode(x)) for x in X],
                                       self.tokenizer.padding_idx)
            )
        elif isinstance(X, list) and all(isinstance(x, torch.Tensor) for x in X):
            inputs = torch.stack(X)
        else:
            raise RuntimeError('X should be an np.ndarray of strings or a list of tensors')

        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        embeddings = []
        iterator = range((len(X) // bs) + 1)
        if (len(X) // bs) > 100:
            iterator = tqdm.tqdm(iterator)
        for i in iterator:
            if i * bs >= len(X):
                break

            _inputs = inputs[i * bs : (i + 1) * bs]
            if torch.cuda.is_available():
                _inputs = _inputs.cuda()

            with torch.no_grad():
                embed = self.model(_inputs)[1]
            embeddings.append(embed)

        embeddings = torch.cat(embeddings, dim=0)

        return embeddings

    def _get_embeddings(self, X, bs=64, update_cache=False, use_cache=False):
        cached_embeddings = []
        cached_indices = []
        _X = []
        for i, x in enumerate(X):
            if update_cache:
                _X.append(x)
            elif use_cache and x in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[x])
                cached_indices.append(i)
            else:
                _X.append(x)

        if len(cached_embeddings) > 0:
            cached_embeddings = torch.stack(cached_embeddings, dim=0)
            cached_indices = np.array(cached_indices)
            cached_embeddings = cached_embeddings.cuda() if torch.cuda.is_available() else cached_embeddings

        if len(_X) == 0:
            return cached_embeddings

        if self.embedding_type == "ESM":
            embeddings = self._get_esm_embeddings(_X, bs=bs)
        elif self.embedding_type == "BERT":
            embeddings = self._get_bert_embeddings(_X, bs=bs)

        if update_cache:
            self.embedding_cache.update({x: emb.detach().cpu() for x, emb in zip(_X, embeddings)})

        if torch.is_tensor(cached_embeddings):
            mask = np.ones(len(X), dtype=bool)
            mask[cached_indices] = 0

            _embeddings = torch.zeros(len(X), cached_embeddings.size(-1)).to(embeddings)
            _embeddings[mask] = embeddings
            _embeddings[~mask] = cached_embeddings
            embeddings = _embeddings

        return embeddings


class EmbeddingDeepEnsemble(EmbeddingSurrogate):

    def __init__(self, config):
        super().__init__(config)

        lr = config.get("lr", 1e-3)
        bs = config.get("bs", 200)
        weight_decay = config.get("weight_decay", 0.0)
        trainer = EmbeddingTrainer(
            model_dict["MLP"],
            embedding_size=self.model.embedding_size,
            lr=lr,
            bs=bs,
            weight_decay=weight_decay,
        )
        self.trainer = trainer

        self.ensemble_size = config.get("ensemble_size", 3)
        self.models = torch.nn.ModuleList()

        self.num_epochs = config.get("num_epochs", 10)
        self.bootstrap_ratio = config.get("bootstrap_ratio", None)
        self.holdout_ratio = config.get("holdout_ratio", 0.0)

    def fit(self, X_train, Y_train, X_test, Y_test, reset=False, log_prefix=""):
        super().fit(X_train, Y_train)
        if self.do_unsupervised_finetune:
            self.unsupervised_finetune(X_train, reset=reset, num_epochs=self.config['num_epochs'])
        if self.do_supervised_finetune:
            prefix = '/'.join([log_prefix, 'sup_finetune'])
            self.supervised_finetune(X_train, Y_train, reset=reset, log_prefix=prefix)

        with torch.no_grad():
            embeddings_train = self._get_embeddings(X_train)
            self.embedding_mean = embeddings_train.mean(0)
            self.embedding_std = embeddings_train.std(0)
            embeddings_train = (embeddings_train - self.embedding_mean) / (self.embedding_std + 1e-6)

            embeddings_test = self._get_embeddings(X_test)
            embeddings_test = (embeddings_test - self.embedding_mean) / (self.embedding_std + 1e-6)

        if isinstance(Y_train, np.ndarray):
            Y_train = torch.from_numpy(Y_train).float()
        if isinstance(Y_test, np.ndarray):
            Y_test = torch.from_numpy(Y_test).float()

        eval_dataset = dataset_util.TransformTensorDataset(
            [embeddings_test, Y_test], None
        )

        self.models = torch.nn.ModuleList()
        for i in range(self.ensemble_size):
            train_dataset = dataset_util.TransformTensorDataset(
                [embeddings_train, Y_train], None
            )
            if self.holdout_ratio > 0.0:
                num_holdout = math.ceil(len(train_dataset) * self.holdout_ratio)
                num_train = len(train_dataset) - num_holdout
                train_dataset, holdout_dataset = train_dataset.random_split(
                    num_train, num_holdout
                )
            else:
                holdout_dataset = None
            if self.bootstrap_ratio is not None:
                X_train, Y_train = train_dataset.tensors
                new_X, new_Y = bo_protein.utils.draw_bootstrap(
                    X_train, Y_train, bootstrap_ratio=self.bootstrap_ratio
                )
                train_dataset = dataset_util.TransformTensorDataset(
                    [new_X, new_Y], None
                )

            _log_prefix = f"{log_prefix}/model_{i}"
            self.trainer.train(
                train_dataset,
                holdout_dataset,
                eval_dataset,
                num_epochs=self.num_epochs,
                log_prefix=_log_prefix,
                early_stopping=self.config.get("early_stopping", False),
            )
            self.models.append(self.trainer.model)

        # print(X_test)
        # prefix = '/'.join([log_prefix, 'eval'])
        self.evaluate(X_test, Y_test, bs=32, log_prefix=log_prefix, split='test')
        return self

    def forward(self, X, Y=None, bs=100):
        for model in self.models:
            model.eval()

        embeddings = self._get_embeddings(X, bs=bs)
        embeddings = (embeddings - self.embedding_mean) / (self.embedding_std + 1e-6)

        if Y is not None:
            if isinstance(Y, np.ndarray):
                Y = torch.from_numpy(Y).float()

            dataset = dataset_util.TransformTensorDataset([embeddings, Y], None)
            loader = torch.utils.data.DataLoader(dataset, batch_size=bs)

            y_hat, y_gt = [], []
            for x_batch, y_batch in loader:
                if torch.cuda.is_available():
                    x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")

                with torch.no_grad():
                    _y_hat = torch.stack(
                        [model(x_batch) for model in self.models], dim=0
                    )

                y_hat.append(_y_hat)
                y_gt.append(y_batch)

            y_hat = torch.cat(y_hat, dim=1)
            y_gt = torch.cat(y_gt, dim=0)

            return (y_hat, y_hat.mean(0), y_hat.std(0)), y_gt
        else:
            dataset = dataset_util.TransformTensorDataset([embeddings], None)
            loader = torch.utils.data.DataLoader(dataset, batch_size=bs)

            y_hat = []
            for x_batch in loader:
                if torch.cuda.is_available():
                    x_batch = x_batch.to("cuda")

                with torch.no_grad():
                    _y_hat = torch.stack(
                        [model(x_batch) for model in self.models], dim=0
                    )

                y_hat.append(_y_hat)

            y_hat = torch.cat(y_hat, dim=1)

            return (y_hat, y_hat.mean(0), y_hat.std(0))

class EmbeddingGP(EmbeddingSurrogate):
    def __init__(self, config, emb_projection=None, gp=None, feature_dim=10):
        super().__init__(config)
        self.config = config
        if emb_projection is not None:
            self.emb_projection = emb_projection
        self.gp = gp
        self.register_buffer('feature_mean', torch.zeros(feature_dim))
        self.register_buffer('feature_std', torch.ones(feature_dim))

    def fit(self, X_train, Y_train, X_test, Y_test, reset=False, log_prefix="emb_gp"):
        self.train()
        super().fit(X_train, Y_train)
        if self.do_unsupervised_finetune:
            self.unsupervised_finetune(X_train, reset, num_epochs=self.config['num_epochs'])
        if self.do_supervised_finetune:
            prefix = '/'.join([log_prefix, 'sup_finetune'])
            # print(X_train[:10])
            # print(Y_train[:10])
            # print(len(X_train))
            # print(len(Y_train))
            self.supervised_finetune(X_train, Y_train, reset, log_prefix=prefix)

        self.eval()
        with torch.no_grad():
            embeddings = self._get_embeddings(X_train)
            features = self.emb_projection(embeddings)
        self.feature_mean = features.mean(0)
        self.feature_std = features.std(0)
        features = (features - self.feature_mean) / (self.feature_std + 1e-6)

        print(features.shape)

        Y_train = torch.tensor(Y_train).to(features).unsqueeze(-1)
        if reset or self.gp is None:
            gp = botorch.models.SingleTaskGP(features, Y_train)
            self.gp = gp.cuda() if torch.cuda.is_available() else gp
        else:
            # self.gp.set_train_data(features, Y_train, strict=False)

            gp = botorch.models.SingleTaskGP(features, Y_train)
            self.gp = gp.cuda() if torch.cuda.is_available() else gp

        self.gp.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        botorch.fit.fit_gpytorch_model(mll)

        self.evaluate(X_test, Y_test, bs=32, log_prefix=log_prefix)
        return self

    def _get_features(self, str_inputs, bs=32):
        self.eval()
        with torch.no_grad():
            embeddings = self._get_embeddings(str_inputs, bs=bs)
            features = self.emb_projection(embeddings)
            features = (features - self.feature_mean) / (self.feature_std + 1e-6)
        return features

    def set_train_data(self, inputs, targets, strict=True):
        features = self._get_features(inputs)
        self.gp.set_train_data(features, targets, strict)

    def condition_on_observations(self, inputs, targets):
        batch_shape = targets.shape[:-1]
        features = self._get_features(inputs).view(*batch_shape, -1)
        conditioned_gp = self.gp.condition_on_observations(features, targets)
        conditioned_model = EmbeddingGP(self.config, self.emb_projection, conditioned_gp)
        conditioned_model.feature_mean = copy.deepcopy(self.feature_mean)
        conditioned_model.feature_std = copy.deepcopy(self.feature_std)
        return conditioned_model

    def evaluate(self, X, Y, bs, log_prefix=""):
        self.eval()
        with torch.no_grad():
            features = self._get_features(X)
            # print(len(X))
            print(features.shape)
            pred_dist = self.gp(features)  # p(f | D)
            f_post_var = pred_dist.variance.mean().item()
            pred_dist = self.gp.likelihood(pred_dist)  # p(y | D)
        mean = pred_dist.mean.cpu()
        std = pred_dist.variance.sqrt().cpu()
        # std = std * 1e-1

        ece = quantile_calibration(mean, std, torch.tensor(Y).cpu())["ece"]

        metrics = {
            "/".join([log_prefix, "rmse"]): np.sqrt(np.power(mean.numpy() - Y, 2).mean()),
            "/".join([log_prefix, "s_rho"]): spearmanr(mean.numpy(), Y).correlation,
            "/".join([log_prefix, "ece"]): ece,
            "/".join([log_prefix, "post_var"]): f_post_var,
        }

        try:
            wandb.log(metrics)
        except Exception as e:
            pass

        return metrics

    def forward(self, X, Y=None, bs=32, update_cache=False, use_cache=False, num_samples=1,
                expand_shape=None, **kwargs):
        features = self._get_features(X, bs)
        features = features if expand_shape is None else features.expand(*expand_shape, -1, -1)
        pred_dist = self.gp(features)

        mean, std = pred_dist.mean, pred_dist.variance.sqrt()
        # std = std * 1e-1
        pred_dist = torch.distributions.Normal(mean, std)
        samples = pred_dist.sample(torch.Size((num_samples,)))

        # mean = pred_dist.mean.detach().cpu().numpy()
        # std = pred_dist.variance.sqrt().detach().cpu().numpy()
        # samples = np.random.normal(mean, std, (num_samples, *mean.shape))

        if Y is None:
            return (samples, mean, std)
        else:
            return (samples, mean, std), Y


class SSKGP(SurrogateModel):
    def __init__(self, config):
        super().__init__()

        self.alphabet = bo_protein.utils.AMINO_ACIDS + [' ']

        self.config = config

        self.ssk_batch_size = config.get("ssk_batch_size", 1000)
        self.gap_decay = config.get("gap_decay", 0.99)
        self.match_decay = config.get("match_decay", 0.53)
        self.max_subsequence_length = config.get("max_subsequence_length", 5)
        self.max_len = config.get("max_len", None)

        if not self.max_len:
            raise Exception('must pass max len to SSKGP')

    def fit(self, X_train, Y_train, X_test, Y_test, reset=False, log_prefix=""):
        import gpflow
        import gpflow
        # from gpflow import set_trainable
        # from gpflow.utilities import positive
        from boss.code.GPflow_wrappers.Batch_SSK import Batch_SSK

        super().fit(X_train, Y_train)
        X_train = np.array([" ".join(list(x)) for x in X_train])[:,None]
        Y_train = Y_train[:,None].astype(np.float64)

        k = Batch_SSK(batch_size=self.ssk_batch_size,
                      gap_decay=self.gap_decay,
                      match_decay=self.match_decay,
                      alphabet=self.alphabet,
                      max_subsequence_length=self.max_subsequence_length, 
                      maxlen=self.max_len)
        
        mean = gpflow.mean_functions.Constant(0)
        scale = gpflow.kernels.Constant(1)
        m = gpflow.models.GPR(data=(X_train, Y_train), mean_function=mean, 
                                                       kernel=scale*k)#, noise_variance=0.003)
        loss = m.log_marginal_likelihood()

        if reset:
            # print(m.trainable_variables)

            optimizer = gpflow.optimizers.Scipy()
            train_vars = [v for v in m.trainable_variables if v.name != "gap_decay:0"]
            optimizer.minimize(m.training_loss, train_vars, options=dict(ftol=0.0001), compile=False)
            # optimizer.minimize(m.training_loss, m.trainable_variables, options=dict(ftol=0.0001), compile=False)
            # self.gap_decay = [v.numpy() for v in m.variables if v.name == "gap_decay:0"][0]
            self.match_decay = [v.numpy() for v in m.variables if v.name == "match_decay:0"][0]

            print(m.trainable_variables)
            # sys.exit(0)
            # print(self.gap_decay)
            # print(self.match_decay)

        # sys.exit(0)

        self.model = m

        return self

    def forward(self, X, Y=None, bs=100, **kwargs):
        X = np.array([" ".join(list(x)) for x in X])[:,None]

        mean, var = self.model.predict_f(X)
        mean, var = mean.numpy()[:,0], var.numpy()[:,0]

        std = np.sqrt(var)
        samples = np.random.normal(mean, std, (10, *mean.shape))

        if Y is None:
            return (samples, mean, std)
        else:
            return (samples, mean, std), Y
