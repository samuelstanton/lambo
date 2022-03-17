from torch.utils.data import Dataset
import numpy as np


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
