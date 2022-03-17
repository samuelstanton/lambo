import os
import numpy as np
import pandas as pd
from rdkit import RDLogger
import selfies as sf
import math
from pathlib import Path

RDLogger.DisableLog('rdApp.*')

from deepchem.feat.smiles_tokenizer import SmilesTokenizer

from cachetools import cached, LRUCache

from lambo.utils import IntTokenizer
from lambo.utils import weighted_resampling


# https://github.com/aspuru-guzik-group/GA/blob/b948519b30bdb162f30e100b647e63bf46777d55/evolution_functions.py#L294
CUSTOM_SAMPLING_VOCAB = [
    '[Branch1]',
    '[#Branch1]',
    '[=Branch1]',
    '[epsilon]',
    '[Ring1]',
    '[Ring2]',
    '[Branch2]',
    '[#Branch2]',
    '[=Branch2]',
    '[F]',
    '[O]',
    '[=O]',
    '[N]',
    '[=N]',
    '[#N]',
    '[C]',
    '[=C]',
    '[#C]',
    '[S]',
    '[=S]',
    '[C][=C][C][=C][C][=C][Ring1][Branch1]'
]


class ChemWrapperModule:

    def __init__(self, num_start_examples=10000, worst_ratio=1., best_ratio=0.):
        file_loc = os.path.dirname(os.path.realpath(__file__))
        root_path = Path(file_loc).parents[1]
        zinc_asset_path = root_path / "assets" / "zinc.csv"
        self.df = pd.read_csv(os.path.join(zinc_asset_path))

        self.n_start_points = num_start_examples
        self.n_worst_points = math.ceil(num_start_examples * worst_ratio)
        self.n_best_points = math.ceil(num_start_examples * best_ratio)
        self.n_rand_points = num_start_examples - self.n_best_points - self.n_worst_points

    def get_worst_points(self, property_list):
        if self.n_worst_points == 0:
            return []

        # scores to be minimized
        obj_vals = np.stack(
            [-self.df[prop_name].values for prop_name in property_list],
            axis=-1
        )
        # lower rank --> strictly dominated by higher rank points
        ranks, _, _ = weighted_resampling(-obj_vals)
        rank_argsort = ranks.argsort()

        chosen_idxs = []
        for row_idx in range(self.n_worst_points):
            chosen_idxs.append(rank_argsort[row_idx])
            if len(chosen_idxs) >= self.n_worst_points:
                chosen_idxs = chosen_idxs[:self.n_worst_points]
                break
        return chosen_idxs

    def get_best_points(self, property_list):
        if self.n_best_points == 0:
            return []

        # scores to be minimized
        obj_vals = np.stack(
            [-self.df[prop_name].values for prop_name in property_list],
            axis=-1
        )
        # lower rank --> strictly dominates higher rank points
        ranks, _, _ = weighted_resampling(obj_vals)
        rank_argsort = ranks.argsort()

        chosen_idxs = []
        for row_idx in range(self.n_best_points):
            chosen_idxs.append(rank_argsort[row_idx])
            if len(chosen_idxs) >= self.n_best_points:
                chosen_idxs = chosen_idxs[:self.n_best_points]
                break
        return chosen_idxs

    def sample_points(self, property_list):
        candidate_rand_points = np.random.choice(
            self.df.shape[0],
            size=self.n_start_points,
            replace=False,
        )

        if self.n_best_points == 0 and self.n_worst_points == 0:
            return candidate_rand_points

        chosen_idxs = self.get_worst_points(property_list)
        best_idxs = self.get_best_points(property_list)
        chosen_idxs.extend([idx for idx in best_idxs if idx not in chosen_idxs])

        # Fill remainder with random points
        for select_idx in candidate_rand_points:
            if select_idx not in chosen_idxs:
                chosen_idxs.append(select_idx)
            if len(chosen_idxs) >= self.n_start_points:
                chosen_idxs = chosen_idxs[:self.n_start_points]
                break
        assert len(chosen_idxs) == self.n_start_points
        return np.array(chosen_idxs)

    def sample_dataset(self, property_list):
        chosen_indices = self.sample_points(property_list)
        smiles = self.df.iloc[chosen_indices]["smiles"].to_numpy()
        targets = np.stack([
            -self.df.iloc[chosen_indices][p_name].values for p_name in property_list
        ], axis=-1)
        if 'penalized_logP' in property_list:
            prop_idx = property_list.index('penalized_logP')
            targets[..., prop_idx] = np.clip(
                targets[..., prop_idx], a_min=None, a_max=4.
            )
        return smiles, targets


class SMILESTokenizer:

    #MAYBE DO THIS WITH SUBCLASSING INSTEAD
    def __init__(self):
        #seyonec/SmilesTokenizer_ChemBERTa_zinc250k_40k
        #seyonec/SMILES_tokenized_PubChem_shard00_160k
        dir_path = os.path.dirname(os.path.realpath(__file__))
        tokenizer = SmilesTokenizer(os.path.join(dir_path,"vocab.txt"))

        with open(os.path.join(dir_path,"vocab.txt"), 'r') as fd:
            self.full_vocab = [x.strip() for x in fd.readlines()]

        with open(os.path.join(dir_path,"restricted_vocab.txt"), 'r') as fd:
            self.non_special_vocab = [x.strip() for x in fd.readlines()]

        self.tokenizer = tokenizer

        self.converter = IntTokenizer(self.non_special_vocab, self.full_vocab)
        self.padding_idx = self.converter.padding_idx
        self.masking_idx = self.converter.masking_idx
        self.eos_idx = self.converter.eos_idx
        self.special_vocab = set(self.full_vocab) - set(self.non_special_vocab)

    def encode(self, s):
        tokens = self.tokenizer.decode(self.tokenizer.encode(s)).split(" ")[1:-1]
        return self.converter.encode(tokens)

    def decode(self, s):
        return self.converter.decode(s)

    def convert_id_to_token(self, token_id):
        return self.converter.convert_id_to_token(token_id)

    def convert_token_to_id(self, token):
        return self.converter.convert_token_to_id(token)


class SELFIESTokenizer(IntTokenizer):
    def __init__(self, smiles_data=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if smiles_data is None:
            try:
                with open(os.path.join(dir_path, "selfies_vocab.txt"), 'r') as fd:
                    non_special_vocab = [x.strip() for x in fd.readlines()]
            except FileNotFoundError:
                smiles_df = pd.read_csv(os.path.join(dir_path, "smiles.csv"))
                selfies_data = list(map(sf.encoder, smiles_df.smiles))
                selfies_alphabet = sf.get_alphabet_from_selfies(selfies_data)
                non_special_vocab = list(sorted(selfies_alphabet))
        else:
            selfies_data = list(map(sf.encoder, smiles_data))
            selfies_alphabet = sf.get_alphabet_from_selfies(selfies_data)
            non_special_vocab = list(sorted(selfies_alphabet))

        full_vocab = ["[nop]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"] + non_special_vocab
        super().__init__(non_special_vocab, full_vocab, padding_token="[nop]")

    @cached(cache=LRUCache(maxsize=int(1e4)))
    def encode(self, seq):
        token_ids = sf.selfies_to_encoding(seq, self.lookup, enc_type='label')
        return [self.bos_idx] + token_ids + [self.eos_idx]

    def to_smiles(self, seq):
        return sf.decoder(seq)
