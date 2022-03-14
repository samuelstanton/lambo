import sys
import wandb
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
import math

from ..gfp_data.transforms import padding_collate_fn


def to_cuda(batch):
    if torch.cuda.is_available():
        return tuple([x.to("cuda") for x in batch])
    else:
        return batch


def check_early_stopping(
        model,
        best_score,
        best_epoch,
        best_weights,
        curr_score,
        curr_epoch,
        patience,
        tol=1e-3,
        save_weights=True,
):
    eps = 1e-6
    stop = False
    if (
            best_score is None
            or (best_score - curr_score) / (abs(best_score) + eps) > tol
    ):
        best_score, best_epoch = curr_score, curr_epoch
    elif curr_epoch - best_epoch >= patience:
        stop = True
    else:
        pass

    if best_epoch == curr_epoch and save_weights:
        del best_weights
        model.cpu()  # avoid storing two copies of the weights on GPU
        best_weights = copy.deepcopy(model.state_dict())
        model.to(model.device)

    return best_score, best_epoch, best_weights, stop


class Trainer(object):
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

            # if eval_dataset is not None:
            #     rmse, s_rho = self.evaluate(eval_dataset)
            #     metrics.update(
            #         {
            #             "/".join([log_prefix, "eval_rmse"]): rmse,
            #             "/".join([log_prefix, "eval_sr"]): s_rho,
            #         }
            #     )

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


class EmbeddingTrainer(object):
    def __init__(self, network, embedding_size, lr=1e-3, bs=100, weight_decay=0.0):
        self.network = network
        self.embedding_size = embedding_size

        self.lr = lr
        self.bs = bs
        self.weight_decay = weight_decay

        self.model = self.network(chin=self.embedding_size)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )

    def train(
        self,
        train_dataset,
        holdout_dataset=None,
        eval_dataset=None,
        num_epochs=100,
        reset=True,
        log_prefix="",
        early_stopping=False,
    ):
        if reset:
            self.model = self.network(chin=self.embedding_size)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        loader = DataLoader(
            train_dataset,
            batch_size=self.bs,
            shuffle=True,
        )
        criterion = nn.MSELoss()
        best_score, best_epoch, best_weights, stop = None, 0, None, False

        # wandb.watch(self.model)
        best_loss, best_loss_epoch = None, 0
        best_score, best_score_epoch = None, 0
        for epoch_idx in range(num_epochs):
            self.model.train()
            avg_train_loss = 0.0
            for x, y in loader:
                if torch.cuda.is_available():
                    x, y = x.to("cuda"), y.to("cuda")

                self.optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item()  / len(loader)

            metrics = {
                "/".join([log_prefix, "train_loss"]): avg_train_loss,
                "/".join([log_prefix, "epoch"]): epoch_idx + 1,
            }

            if holdout_dataset is not None:
                val_rmse, _ = self.evaluate(holdout_dataset)
                metrics.update(
                    {
                        "/".join([log_prefix, "holdout_rmse"]): val_rmse,
                    }
                )

                best_score, best_score_epoch, best_weights, _ = check_early_stopping(
                    model=self,
                    best_score=best_score,
                    best_epoch=best_score_epoch,
                    best_weights=best_weights,
                    curr_score=val_rmse,
                    curr_epoch=epoch_idx + 1,
                    patience=2,
                    save_weights=True,
                )
                metrics.update(dict(best_score=best_score, best_epoch=best_score_epoch))

            # use train loss to determine convergence
            best_loss, best_loss_epoch, _, stop = check_early_stopping(
                model=self,
                best_score=best_loss,
                best_epoch=best_loss_epoch,
                best_weights=None,
                curr_score=avg_train_loss,
                curr_epoch=epoch_idx + 1,
                patience=2,
                save_weights=False,
            )
            metrics.update(dict(best_loss=best_loss, best_loss_epoch=best_loss_epoch))

            # if eval_dataset is not None:
            #     rmse, s_rho = self.evaluate(eval_dataset)
            #     metrics.update(
            #         {
            #             "/".join([log_prefix, "eval_rmse"]): rmse,
            #             "/".join([log_prefix, "eval_sr"]): s_rho,
            #         }
            #     )
            # try:
            #     wandb.log(metrics)
            # except Exception as e:
            #     print(e)

            if early_stopping and stop:
                assert holdout_dataset is not None
                # print(f'stopped model training after {epoch + 1} epoch(s)')
                # print(f'loading checkpoint from {best_epoch} with score {best_score:0.4f}.')
                self.model.load_state_dict(best_weights)
                break
        self.model.eval()

    def evaluate(self, dataset):
        loader = DataLoader(
            dataset, batch_size=self.bs, shuffle=False,
        )

        Y_hat = []
        Y_test = []
        self.model.eval()
        for x, y in loader:
            if torch.cuda.is_available():
                x, y = x.to("cuda"), y.to("cuda")

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

    def early_stopping(
        self,
        best_score,
        best_epoch,
        best_weights,
        curr_score,
        curr_epoch,
        patience=100,
        tol=1e-3,
    ):
        eps = 1e-6
        stop = False
        if (
            best_score is None
            or (best_score - curr_score) / (abs(best_score) + eps) > tol
        ):
            best_score, best_epoch = curr_score, curr_epoch
            best_weights = copy.deepcopy(self.model.state_dict())
        elif curr_epoch - best_epoch >= patience:
            stop = True
        return best_score, best_epoch, best_weights, stop



def quantile_calibration(mean, std, targets):
    quantiles = torch.linspace(0.05, 0.95, 10, device=mean.device).view(10, 1, 1)

    z_dist = torch.distributions.Normal(
        torch.tensor((0.0,), device=mean.device),
        torch.tensor((1.0,), device=mean.device),
    )
    tail_probs = (1 - quantiles) / 2
    z_scores = z_dist.icdf(1 - tail_probs)  # (num_quantiles, 1, 1)

    pred_mean = mean.unsqueeze(0)  # (1, batch_size, target_dim)
    pred_std = std.unsqueeze(0)
    lb = pred_mean - z_scores * pred_std
    ub = pred_mean + z_scores * pred_std

    targets = targets.unsqueeze(0)
    targets_in_region = torch.le(lb, targets) * torch.le(targets, ub)
    occupancy_rates = targets_in_region.float().mean(-1, keepdim=True)  # average over target dim
    occupancy_rates = occupancy_rates.mean(-2, keepdim=True)  # average over batch dim
    # import pdb; pdb.set_trace()
    ece = (occupancy_rates - quantiles).abs().mean().item()
    calibration_metrics = {
        f"{quantile.item():.2f}_quantile": occ_rate.item()
        for quantile, occ_rate in zip(quantiles, occupancy_rates)
    }
    calibration_metrics["ece"] = ece
    # in general,
    # over-confident --> quantiles > occupancy rates --> positive diff
    # under-confident --> quantiles < occupancy rates --> negative diff
    calibration_metrics["occ_diff"] = (quantiles - occupancy_rates).mean().item()
    return calibration_metrics


# if __name__=='__main__':
#     cfg = argupdated_config(makeTrainer.__kwdefaults__,namespace = models.nn_models)
#     trainer = makeTrainer(**cfg)
#     trainer.train(cfg['num_epochs'])
