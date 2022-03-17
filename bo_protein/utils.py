import math
import random
from collections.__init__ import namedtuple
import itertools

import numpy as np
import torch
from Levenshtein._levenshtein import distance as edit_distance, editops as edit_ops

from scipy.stats import rankdata
from scipy.special import softmax

from cachetools import cached, LRUCache

from bo_protein.transforms import padding_collate_fn

AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
RESIDUE_ALPHABET = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"] + AMINO_ACIDS + ["0"]


class IntTokenizer:
    def __init__(self, non_special_vocab, full_vocab, padding_token="[PAD]",
                 masking_token="[MASK]", bos_token="[CLS]", eos_token="[SEP]"):
        self.non_special_vocab = non_special_vocab
        self.full_vocab = full_vocab
        self.special_vocab = set(full_vocab) - set(non_special_vocab)
        self.lookup = {a: i for (i, a) in enumerate(full_vocab)}
        self.inverse_lookup = {i: a for (i, a) in enumerate(full_vocab)}
        self.padding_idx = self.lookup[padding_token]
        self.masking_idx = self.lookup[masking_token]
        self.bos_idx = self.lookup[bos_token]
        self.eos_idx = self.lookup[eos_token]

        self.sampling_vocab = non_special_vocab
        self.non_special_idxs = [self.convert_token_to_id(t) for t in non_special_vocab]
        self.special_idxs = [self.convert_token_to_id(t) for t in self.special_vocab]

    @cached(cache=LRUCache(maxsize=int(1e4)))
    def encode(self, seq):
        seq = ["[CLS]"] + list(seq) + ["[SEP]"]
        return [self.convert_token_to_id(c) for c in seq]

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            return self.convert_id_to_token(token_ids)

        tokens = []
        for t_id in token_ids:
            token = self.convert_id_to_token(t_id)
            if token in self.special_vocab and token not in ["[MASK]", "[UNK]"]:
                continue
            tokens.append(token)
        return ' '.join(tokens)

    def convert_id_to_token(self, token_id):
        if torch.is_tensor(token_id):
            token_id = token_id.item()
        assert isinstance(token_id, int)
        return self.inverse_lookup.get(token_id, '[UNK]')

    def convert_token_to_id(self, token):
        unk_idx = self.lookup["[UNK]"]
        return self.lookup.get(token, unk_idx)

    def set_sampling_vocab(self, sampling_vocab=None, max_ngram_size=1):
        if sampling_vocab is None:
            sampling_vocab = []
            for i in range(1, max_ngram_size + 1):
                prod_space = [self.non_special_vocab] * i
                for comb in itertools.product(*prod_space):
                    sampling_vocab.append("".join(comb))
        else:
            new_tokens = set(sampling_vocab) - set(self.full_vocab)
            self.full_vocab.extend(list(new_tokens))
            self.lookup = {a: i for (i, a) in enumerate(self.full_vocab)}
            self.inverse_lookup = {i: a for (i, a) in enumerate(self.full_vocab)}

        self.sampling_vocab = sampling_vocab


class ResidueTokenizer(IntTokenizer):
    def __init__(self):
        super().__init__(AMINO_ACIDS, RESIDUE_ALPHABET)


def random_proteins(num, min_len=200, max_len=250):
    alphabet = AMINO_ACIDS

    proteins = []
    for _ in range(num):
        length = np.random.randint(min_len, max_len + 1)
        idx = np.random.choice(len(alphabet), size=length, replace=True)
        proteins.append("".join([alphabet[i] for i in idx]))
    proteins = np.array(proteins)

    return proteins


class Expression(torch.nn.Module):
    def __init__(self, lambda_fn):
        super().__init__()
        self.fn = lambda_fn

    def forward(self, x):
        return self.fn(x)


def draw_bootstrap(*arrays, bootstrap_ratio=0.632, min_samples=1):
    """
    Returns bootstrapped arrays that (in expectation) have `bootstrap_ratio` proportion
    of the original rows. The size of the bootstrap is computed automatically.
    For large input arrays, the default value will produce a bootstrap
    the same size as the original arrays.

    :param arrays: indexable arrays (e.g. np.ndarray, torch.Tensor)
    :param bootstrap_ratio: float in the interval (0, 1)
    :param min_samples: (optional) instead specify the minimum size of the bootstrap
    :return: bootstrapped arrays
    """

    num_data = arrays[0].shape[0]
    assert all(arr.shape[0] == num_data for arr in arrays)

    if bootstrap_ratio is None:
        num_samples = min_samples
    else:
        assert bootstrap_ratio < 1
        num_samples = int(math.log(1 - bootstrap_ratio) / math.log(1 - 1 / num_data))
        num_samples = max(min_samples, num_samples)

    idxs = random.choices(range(num_data), k=num_samples)
    res = [arr[idxs] for arr in arrays]
    return res


def to_tensor(*arrays, device=torch.device('cpu')):
    tensors = []
    for arr in arrays:
        if isinstance(arr, torch.Tensor):
            tensors.append(arr.to(device))
        else:
            tensors.append(torch.tensor(arr, device=device))

    if len(arrays) == 1:
        return tensors[0]

    return tensors


def batched_call(fn, arg_array, batch_size, *args, **kwargs):
    batch_size = arg_array.shape[0] if batch_size is None else batch_size
    num_batches = max(1, arg_array.shape[0] // batch_size)

    if isinstance(arg_array, np.ndarray):
        arg_batches = np.array_split(arg_array, num_batches)
    elif isinstance(arg_array, torch.Tensor):
        arg_batches = torch.split(arg_array, num_batches)
    else:
        raise ValueError

    return [fn(batch, *args, **kwargs) for batch in arg_batches]


def mutation_list(src_str, tgt_str, tokenizer):
    # def to_unicode(seq):
    #     return ''.join([chr(int(x)) for x in seq]).encode('utf-8').decode('utf-8')

    src_token_id_seq = ''.join([chr(x) for x in tokenizer.encode(src_str)[1:-1]])
    tgt_token_id_seq = ''.join([chr(x) for x in tokenizer.encode(tgt_str)[1:-1]])

    if edit_distance(src_token_id_seq, tgt_token_id_seq) == 0:
        return []

    mutations = []
    tmp_tok_id_seq, trans_adj = src_token_id_seq, 0

    # TODO make sure insertion is properly supported, update tests

    for op_name, pos_1, pos_2 in edit_ops(src_token_id_seq, tgt_token_id_seq):
        tmp_pos = pos_1 + trans_adj

        if op_name == "delete":
            char_1 = ord(src_token_id_seq[pos_1])
            if pos_1 == len(src_token_id_seq) - 1:
                tmp_tok_id_seq = tmp_tok_id_seq[:-1]
            else:
                tmp_tok_id_seq = tmp_tok_id_seq[:tmp_pos] + tmp_tok_id_seq[tmp_pos + 1:]

            mutations.append(StringDeletion(char_1, tmp_pos, tokenizer))
            trans_adj -= 1

        if op_name == "replace":
            char_1 = ord(src_token_id_seq[pos_1])
            char_2 = ord(tgt_token_id_seq[pos_2])
            if pos_1 == len(src_token_id_seq) - 1:
                tmp_tok_id_seq = tmp_tok_id_seq[:-1] + chr(char_2)
            else:
                tmp_tok_id_seq = tmp_tok_id_seq[:tmp_pos] + chr(char_2) + tmp_tok_id_seq[tmp_pos + 1:]

            mutations.append(StringSubstitution(char_1, tmp_pos, char_2, tokenizer))

        if op_name == "insert":
            if pos_1 < len(src_token_id_seq):
                char_1 = ord(src_token_id_seq[pos_1])
            else:
                char_1 = None

            char_2 = ord(tgt_token_id_seq[pos_2])
            if pos_1 == len(src_token_id_seq) - 1:
                tmp_tok_id_seq = tmp_tok_id_seq[:-1] + chr(char_2) + tmp_tok_id_seq[-1]
            else:
                tmp_tok_id_seq = tmp_tok_id_seq[:tmp_pos] + chr(char_2) + tmp_tok_id_seq[tmp_pos:]

            mutations.append(StringInsertion(char_1, tmp_pos, char_2, tokenizer))
            trans_adj += 1

    # check output
    tmp_str = tokenizer.decode([ord(x) for x in tmp_tok_id_seq]).replace(" ", "")
    assert tmp_tok_id_seq == tgt_token_id_seq
    assert tmp_str == tgt_str, f'{tgt_str}\n{tmp_str}'

    return mutations


class StringSubstitution:
    def __init__(self, old_token_idx, token_pos, new_token_idx, tokenizer):
        self.old_token_idx = int(old_token_idx)
        self.old_token = tokenizer.decode(self.old_token_idx)

        self.token_pos = int(token_pos)

        self.new_token_idx = int(new_token_idx)
        self.new_token = tokenizer.decode(self.new_token_idx)

    def __str__(self):
        prefix = f"{self.token_pos}{self.old_token}-{self.token_pos}{self.old_token}_"
        return prefix + f"sub{self.new_token}"


class StringDeletion:
    def __init__(self, old_token_idx, token_pos, tokenizer):
        self.old_token_idx = int(old_token_idx)
        self.old_token = tokenizer.decode(self.old_token_idx)

        self.token_pos = int(token_pos)

    def __str__(self):
        prefix = f"{self.token_pos}{self.old_token_idx}-{self.token_pos}{self.old_token_idx}_"
        return prefix + "del"


class StringInsertion:
    def __init__(self, old_token_idx, token_pos, new_token_idx, tokenizer):
        if old_token_idx is None:
            self.old_token_idx = None
            self.old_token = ''
        else:
            self.old_token_idx = int(old_token_idx)
            self.old_token = tokenizer.decode(self.old_token_idx)

        self.token_pos = int(token_pos)

        self.new_token_idx = int(new_token_idx)
        self.new_token = tokenizer.decode(self.new_token_idx)

    def __str__(self):
        prefix = f"{self.token_pos}{self.old_token}-{self.token_pos}{self.old_token}_"
        return prefix + f"ins{self.new_token}"


# FoldxMutation = namedtuple("FoldxMutation", "wt_residue chain residue_number mutant_residue")
class FoldxMutation(StringSubstitution):
    def __init__(self, old_token_idx, chain, token_pos, new_token_idx, tokenizer):
        super().__init__(old_token_idx, token_pos, new_token_idx, tokenizer)
        self.chain = chain
        # syntactic sugar for compatibilty
        self.wt_residue = tokenizer.decode(old_token_idx)
        self.residue_number = token_pos
        self.mutant_residue = tokenizer.decode(new_token_idx)


def weighted_resampling(scores, k=1., num_samples=None):
    """
    Multi-objective ranked resampling weights.
    Assumes scores are being minimized.

    Args:
        scores: (num_rows, num_scores)
        k: softmax temperature
        num_samples: number of samples to draw (with replacement)
    """
    num_rows = scores.shape[0]
    scores = scores.reshape(num_rows, -1)

    ranks = rankdata(scores, method='dense', axis=0)  # starts from 1
    ranks = ranks.max(axis=-1)  # if A strictly dominates B it will have higher weight.

    weights = softmax(-np.log(ranks) / k)

    num_samples = num_rows if num_samples is None else num_samples
    resampled_idxs = np.random.choice(
        np.arange(num_rows), num_samples, replace=True, p=weights
    )
    return ranks, weights, resampled_idxs


fields = ("inputs", "targets")
defaults = (np.array([]), np.array([]))
DataSplit = namedtuple("DataSplit", fields, defaults=defaults)


def update_splits(
    train_split: DataSplit,
    val_split: DataSplit,
    test_split: DataSplit,
    new_split: DataSplit,
    holdout_ratio: float = 0.2,
):
    r"""
    This utility function updates train, validation and test data splits with
    new observations while preventing leakage from train back to val or test.
    New observations are allocated proportionally to prevent the
    distribution of the splits from drifting apart.

    New rows are added to the validation and test splits randomly according to
    a binomial distribution determined by the holdout ratio. This allows all splits
    to be updated with as few new points as desired. In the long run the split proportions
    will converge to the correct values.
    """
    train_inputs, train_targets = train_split
    val_inputs, val_targets = val_split
    test_inputs, test_targets = test_split

    # shuffle new data
    new_inputs, new_targets = new_split
    new_perm = np.random.permutation(
        np.arange(new_inputs.shape[0])
    )
    new_inputs = new_inputs[new_perm]
    new_targets = new_targets[new_perm]

    unseen_inputs = safe_np_cat([test_inputs, new_inputs])
    unseen_targets = safe_np_cat([test_targets, new_targets])

    num_rows = train_inputs.shape[0] + val_inputs.shape[0] + unseen_inputs.shape[0]
    num_test = min(
        np.random.binomial(num_rows, holdout_ratio / 2.),
        unseen_inputs.shape[0],
    )
    num_test = max(test_inputs.shape[0], num_test) if test_inputs.size else max(1, num_test)

    # first allocate to test split
    test_split = DataSplit(unseen_inputs[:num_test], unseen_targets[:num_test])

    resid_inputs = unseen_inputs[num_test:]
    resid_targets = unseen_targets[num_test:]
    resid_inputs = safe_np_cat([val_inputs, resid_inputs])
    resid_targets = safe_np_cat([val_targets, resid_targets])

    # then allocate to val split
    num_val = min(
        np.random.binomial(num_rows, holdout_ratio / 2.),
        resid_inputs.shape[0],
    )
    num_val = max(val_inputs.shape[0], num_val) if val_inputs.size else max(1, num_val)
    val_split = DataSplit(resid_inputs[:num_val], resid_targets[:num_val])

    # train split gets whatever is left
    last_inputs = resid_inputs[num_val:]
    last_targets = resid_targets[num_val:]
    train_inputs = safe_np_cat([train_inputs, last_inputs])
    train_targets = safe_np_cat([train_targets, last_targets])
    train_split = DataSplit(train_inputs, train_targets)

    return train_split, val_split, test_split


def safe_np_cat(arrays, **kwargs):
    if all([arr.size == 0 for arr in arrays]):
        return np.array([])
    cat_arrays = [arr for arr in arrays if arr.size]
    return np.concatenate(cat_arrays, **kwargs)


def str_to_tokens(str_array, tokenizer):
    tokens = [
        torch.tensor(tokenizer.encode(x)) for x in str_array
    ]
    batch = padding_collate_fn(tokens, tokenizer.padding_idx)
    return batch


def tokens_to_str(tok_idx_array, tokenizer):
    str_array = np.array([
        tokenizer.decode(token_ids).replace(' ', '') for token_ids in tok_idx_array
    ])
    return str_array


def apply_mutation(base_seq, mut_pos, mut_res, op_type, tokenizer):
    tokens = tokenizer.decode(tokenizer.encode(base_seq)).split(" ")[1:-1]

    if op_type == 'sub':
        mut_seq = "".join(tokens[:mut_pos] + [mut_res] + tokens[(mut_pos + 1):])
    elif op_type == 'ins':
        mut_seq = "".join(tokens[:mut_pos] + [mut_res] + tokens[mut_pos:])
    elif op_type == 'del':
        mut_seq = "".join(tokens[:mut_pos] + tokens[(mut_pos + 1):])
    else:
        raise ValueError('unsupported operation')

    return mut_seq


def to_cuda(batch):
    if torch.cuda.is_available():
        return tuple([x.to("cuda") for x in batch])
    else:
        return batch