import numpy as np
import torch
import torchvision
import wandb
from scipy.stats import spearmanr

from lambo import transforms, dataset as dataset_util
from lambo.models.metrics import quantile_calibration


class BaseSurrogate(torch.nn.Module):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def _set_transforms(self, tokenizer, max_shift, mask_size, train_prepend=None):
        # convert from string to LongTensor of token indexes
        # don't use the max_len arg here, will interfere with the translation augmentation
        train_transform = [] if train_prepend is None else train_prepend
        train_transform += [transforms.StringToLongTensor(tokenizer, max_len=None)]
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
