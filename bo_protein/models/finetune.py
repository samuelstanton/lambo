import typing
from typing import Union, List, Tuple, Sequence, Dict, Any

import sys
import math

from copy import copy
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.weight_norm import weight_norm

from tape import TAPETokenizer, ProteinBertModel, ProteinBertConfig

from ..gfp_data import transforms
from ..gfp_data import dataset as dataset_util
from ..gfp_data import utils as gfp_utils


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class MaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """
    def __init__(
        self, X, tokenizer: Union[str, TAPETokenizer] = "iupac", in_memory: bool = False
    ):
        super().__init__()
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.data = X

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        tokens = self.tokenizer.tokenize(self.data[index])
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64
        )
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64
        )

        return masked_token_ids, input_mask, labels

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_ids = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

        return input_ids, input_mask, lm_label_ids

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                pass

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1)
                    )
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def get_activation_fn(name: str) -> typing.Callable:
    if name == "gelu":
        return gelu
    elif name == "relu":
        return torch.nn.functional.relu
    elif name == "swish":
        return swish
    else:
        raise ValueError(f"Unrecognized activation fn: {name}")


class LayerNorm(nn.Module):  # type: ignore
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PredictionHeadTransform(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_act: typing.Union[str, typing.Callable] = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = get_activation_fn(hidden_act)
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MLMHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: typing.Union[str, typing.Callable] = "gelu",
        layer_norm_eps: float = 1e-12,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.transform = PredictionHeadTransform(
            hidden_size, hidden_act, layer_norm_eps
        )

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        self.vocab_size = vocab_size
        self._ignore_index = ignore_index

    def forward(self, hidden_states, targets=None):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        outputs = (hidden_states,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.view(-1, self.vocab_size), targets.view(-1)
            )
            metrics = {"perplexity": torch.exp(masked_lm_loss)}
            loss_and_metrics = (masked_lm_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), prediction_scores


class TransformerForMaskedLM(nn.Module):
    def __init__(self, config, transformer):
        super().__init__()

        self.config = config
        self.transformer = transformer  # ProteinBertModel(config)
        self.mlm = MLMHead(
            config.hidden_size,
            config.vocab_size,
            config.hidden_act,
            config.layer_norm_eps,
            ignore_index=-1,
        )

        # self.init_weights()
        self.tie_weights()

    def _tie_or_clone_weights(self, first_module, second_module):
        """Tie or clone module weights depending of weither we are using TorchScript or not"""
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(
            self.mlm.decoder, self.transformer.embeddings.word_embeddings
        )

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.transformer(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        outputs = self.mlm(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

class SimpleMLP(nn.Sequential):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hid_dim, out_dim)
        )

class ValuePredictionHead(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float = 0.0):
        super().__init__()
        projection = SimpleMLP(input_size, hidden_size, 10, dropout)
        self.projection = projection
        self.projection.add_module('batch_norm', nn.BatchNorm1d(10, affine=False))
        self.value_prediction = nn.Linear(10, 1)

    def forward(self, pooled_output, targets=None):
        features = self.projection(pooled_output)
        value_pred = self.value_prediction(features)
        outputs = (value_pred,)

        if targets is not None:
            loss_fct = nn.MSELoss()
            value_pred_loss = loss_fct(value_pred, targets)
            outputs = (value_pred_loss,) + outputs
        return outputs  # (loss), value_prediction


class TransformerForValuePrediction(nn.Module):
    def __init__(self, config, transformer, p):
        super().__init__()

        self.config = config
        self.transformer = transformer
        self.predict = ValuePredictionHead(config.input_size, config.hidden_size, p)

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.transformer(input_ids)#, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


def main(**config):
    (X_train, y_train), (X_test, y_test) = gfp_utils.load_data(
        "tape", None, split=0.9, train_wo_cutoff=False
    )
    # unsupervised_finetune_tape(config, X_train)

    net, loader, tokenizer = supervised_finetune({}, X_train, y_train)

    opt = optim.Adam(net.parameters(), lr=1e-4)

    steps = 0
    for x, y in loader:
        if steps > 300:
            break

        mask = x != 0

        if torch.cuda.is_available():
            x = x.cuda()
            mask = mask.cuda()
            y = y.cuda()

        loss, *other = net(x, mask, y[:, None])

        print(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        steps += 1


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
