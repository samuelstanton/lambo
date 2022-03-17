import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import TensorDataset
import numpy as np

import bo_protein.utils
from . import utils
from . import transforms

class TransformTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, idx):
        x = self.tensors[0][idx]

        if self.transform:
            x = self.transform(x)

        if len(self.tensors) >= 2:
            rest = [self.tensors[i][idx] for i in range(1, len(self.tensors))]
            return (x,) + tuple(rest)
        else:
            return x

    def __len__(self):
        return len(self.tensors[0])

    def random_split(self, size_1, size_2):
        assert size_1 + size_2 == len(self)
        idxs = np.random.permutation(len(self))
        split_1 = TransformTensorDataset(
            [tensor[idxs[:size_1]] for tensor in self.tensors], self.transform
        )
        split_2 = TransformTensorDataset(
            [tensor[idxs[size_1:]] for tensor in self.tensors], self.transform
        )
        return split_1, split_2


def get_gfp_dataset(source, task):
    if source == "fpbase":
        X, Y = utils.load_fpbase_data(task)
    elif source == "localfl":
        X, Y = utils.load_localfl_data(task)

    max_len = np.max([len(x) for x in X]) + 2  # for start and end codes
    tokenizer = bo_protein.utils.RESIDUE_TOKENIZER
    transform = transforms.StringToPaddedLongTensor(tokenizer, max_len)

    Y = torch.from_numpy(Y).float()
    dataset = TransformTensorDataset([X, Y], transform)

    return dataset


# def get_gfp_transform_dataset(X, Y, max_len):
# #     max_len = np.max([len(x) for x in X]) + 2 #for start and end codes
#     tokenizer = utils.IntTokenizer(utils.RESIDUE_ALPHABET)
#     transform = transforms.StringToPaddedLongTensor(tokenizer, max_len)

#     Y = torch.from_numpy(Y).float()
#     dataset = TransformTensorDataset([X, Y], transform)
#     return dataset


def get_embedding_dataset(source, task, device=None):
    if source == "fpbase":
        X, Y = utils.load_fpbase_data(task)
    elif source == "localfl":
        X, Y = utils.load_localfl_data(task)

    max_len = np.max([len(x) for x in X]) + 2  # for start and end codes
    transform = transforms.get_bert_embed_transform(max_len, device)

    Y = torch.from_numpy(Y).float()
    dataset = TransformTensorDataset([X, Y], transform)

    # compute all embeddings up front
    _X, _Y = [], []
    for x, y in dataset:
        _X.append(x)
        _Y.append(y)
    X = torch.stack(_X, dim=0)
    Y = torch.stack(_Y, dim=0)

    return TensorDataset([X, Y])
