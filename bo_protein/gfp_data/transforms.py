import torch
import numpy as np
from tape import TAPETokenizer, ProteinBertModel
import random
import torch.nn.functional as F

from scipy.stats import loguniform

from torch import LongTensor


def padding_collate_fn(batch, padding_value=0.0):
    with torch.no_grad():
        if isinstance(batch[0], tuple):
            k = len(batch[0])
            x = torch.nn.utils.rnn.pad_sequence(
                [b[0] for b in batch], batch_first=True, padding_value=padding_value
            )
            rest = [torch.stack([b[i] for b in batch]) for i in range(1, k)]
            return (x,) + tuple(rest)
        else:
            x = torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=padding_value
            )
            return x


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class StringToLongTensor:
    def __init__(self, tokenizer, max_len=None):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, x: str):
        tok_idxs = self.tokenizer.encode(x)
        tok_idxs = torch.LongTensor(tok_idxs)
        num_tokens = tok_idxs.size(0)
        if self.max_len is not None and num_tokens < self.max_len:
            len_diff = self.max_len - num_tokens
            padding = LongTensor(
                [self.tokenizer.padding_idx] * len_diff
            )
            tok_idxs = torch.cat([tok_idxs, padding])
        elif self.max_len is not None and num_tokens > self.max_len:
            tok_idxs = tok_idxs[:self.max_len]

        return tok_idxs


class SequenceTranslation(object):
    """
    Performs a random cycle rotation of a tokenized sequence up to
    `max_shift` tokens either left or right.
    Assumes the sequence has start and stop tokens and NO padding tokens at the end.
    """

    def __init__(self, max_shift: int):
        self.max_shift = max_shift

    def __call__(self, x: LongTensor, shift=None):
        """
        Args:
            x: LongTensor with shape (num_tokens,)
            shift: (optional) magnitude and direction of shift, randomly sampled if None
        """
        if shift is None:
            shift = random.randint(-self.max_shift, self.max_shift)
        else:
            shift = min(shift, self.max_shift)
            shift = max(shift, -self.max_shift)

        num_valid_tokens = x.size(0) - 2
        if shift < 0:
            shift = -(-shift % num_valid_tokens)
        elif shift > 0:
            shift = shift % num_valid_tokens

        if shift == 0:
            return x

        # don't include start/stop tokens in rotation
        trimmed_x = x[1:-1]
        rot_x = x.clone()
        # left shift
        if shift < 0:
            rot_x[1: num_valid_tokens + shift + 1] = trimmed_x[-shift:]
            rot_x[num_valid_tokens + shift + 1: -1] = trimmed_x[:-shift]
        # right shift
        else:
            rot_x[1: shift + 1] = trimmed_x[-shift:]
            rot_x[shift + 1: -1] = trimmed_x[:-shift]

        return rot_x


class RandomMask(object):
    """
    Randomly replaces original tokens with masking tokens.
    Assumes the sequence has start and stop tokens and NO padding tokens.
    """

    def __init__(self, mask_size: int, masking_idx: int, contiguous: bool = True):
        """
        Args:
            mask_size: number of tokens to mask
            masking_idx: mask token index from tokenizer
            contiguous: if True, consecutive tokens will be masked, otherwise positions will
                        be drawn independently without replacement.
        """
        self.mask_size = mask_size
        self.masking_idx = masking_idx
        self.contiguous = contiguous

    def __call__(self, x: LongTensor):
        """
        Args:
            x: LongTensor with shape (num_tokens,)
        """
        num_tokens = x.size(0)
        # don't mask start or stop tokens
        start_min = 1
        stop_max = num_tokens - 1
        # don't mask all tokens
        if stop_max - start_min <= self.mask_size:
            return x
        # mask consecutive tokens
        if self.contiguous:
            offset = np.random.randint(start_min, start_min + self.mask_size)
            mask_start = np.random.randint(offset, stop_max - self.mask_size)
            mask_stop = mask_start + self.mask_size
            if mask_stop >= stop_max:
                mask_start = stop_max - self.mask_size
                mask_stop = stop_max
            x[mask_start:mask_stop] = self.masking_idx
        # mask random tokens
        else:
            mask_idxs = np.random.choice(
                np.arange(start_min, stop_max), self.mask_size, replace=False
            )
            x[mask_idxs] = self.masking_idx
        return x


# TODO deprecate
class BertEmbedTransform:
    def __init__(self, device=None):
        self.device = device
        self.model = ProteinBertModel.from_pretrained("bert-base").eval().to(device)

    def __call__(self, x):
        x = x[None, :].to(self.device)

        with torch.no_grad():
            features = self.model(x)[0][0]

        return features.mean(0)


# TODO deprecate
def get_bert_embed_transform(max_len, device):
    tokenizer = TAPETokenizer(vocab="iupac")
    token_transform = StringToLongTensor(tokenizer, max_len)

    embed_transform = BertEmbedTransform(device=device)
    compose_transform = Compose([token_transform, embed_transform])

    return compose_transform


# TODO deprecate
class RandomTranslation(torch.nn.Module):
    def __init__(self, max_shift=4):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x):
        if len(x) == 0:
            return x

        S = self.max_shift
        if not self.training:
            return F.pad(x, (S // 2, S // 2), "constant", 0)
        bs, N = x.shape
        x = F.pad(x, (S, S), "constant", 0)
        shifts = np.random.randint(0, S, bs)
        x = torch.stack([x[i, shifts[i] : shifts[i] + N + S] for i in range(bs)])
        return x


# TODO deprecate
class RandomErasing1d(torch.nn.Module):
    def __init__(self, p=0.5, af=1 / 5, max_scale=3):
        self.p = p
        self.area_frac = af
        self.max_scale = max_scale
        super().__init__()

    def forward(self, x):
        if self.training:
            return self.random_erase(x)
        else:
            return x

    def random_erase(self, img):
        bs, n = img.shape
        area = n
        target_areas = (
                loguniform.rvs(1 / self.max_scale, self.max_scale, size=bs)
                * self.area_frac
                * area
        )

        do_erase = np.random.random(bs) < self.p
        cut_hs = target_areas * do_erase
        cut_i = np.random.randint(n, size=bs)
        i = np.arange(n)[None] + np.zeros((bs, n))

        ui = (cut_i + cut_hs / 2)[:, None]
        li = (cut_i - cut_hs / 2)[:, None]
        no_erase_mask = ~((li < i) & (i < ui))
        no_erase_tensor = torch.from_numpy(no_erase_mask.astype(np.float32)).to(
            img.device
        )
        return torch.where(no_erase_tensor > 0, img, 0 * img)
