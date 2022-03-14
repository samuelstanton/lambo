import os
import re
import copy
import pickle
import subprocess
import collections
import numpy as np
from Levenshtein import distance

import torch
from hmmlearn.hmm import MultinomialHMM
import esm

import bo_protein.utils
from bo_protein.acquisition import SingleFidelityAcquisition
from bo_protein.models.surrogates import DeepEnsemble
from bo_protein.gfp_data import utils
from bo_protein.mavedb_data import utils as mdb_utils


class RegexRewardFunc:
    def __init__(self, regex):
        self.regex = regex

    def score(self, X):
        return np.array([len(re.findall(self.regex, str(x))) for x in X])


class MostCommonBigramRewardFunc:
    def __init__(self, X):
        bigram = self._find_most_common_bigram(X)
        self.reward_func = RegexRewardFunc(f"(?={bigram})")

    def _find_most_common_bigram(self, X):
        bigrams = [x[i : i + 2] for x in X for i in range(len(x) - 2)]
        counter = collections.Counter(bigrams)
        return counter.most_common(1)[0][0]

    def score(self, X):
        return self.reward_func.score(X)


class EditDistRewardFunc:
    def __init__(self, target):
        self.target = target

    def score(self, X):
        return np.array([-distance(self.target, x) for x in X])


class NNRewardFunc:
    def __init__(self, data, config, save_fn=None):
        if save_fn and os.path.exists(save_fn):
            if config["num_epochs"] != 0:
                config = copy.deepcopy(config)
                config["num_epochs"] = 0
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.ensemble = DeepEnsemble(config)
            self.ensemble.load_state_dict(torch.load(save_fn, map_location=device))
        else:
            self.ensemble = DeepEnsemble(config).fit(*data, log_prefix="rew_func")
            if save_fn:
                torch.save(self.ensemble.state_dict(), save_fn)

        self.ensemble = SingleFidelityAcquisition(self.ensemble, {})

        if torch.cuda.is_available():
            self.ensemble = self.ensemble.cuda()

    def score(self, X):
        with torch.no_grad():
            _, mean, _ = self.ensemble(X)
        return mean.cpu().numpy()


class HMMRewardFunc:
    def __init__(self, data, config, save_fn=None):
        self.tokenizer = bo_protein.utils.RESIDUE_TOKENIZER

        if save_fn and os.path.exists(save_fn):
            with open(save_fn, "rb") as fd:
                self.hmm = pickle.load(fd)
            return

        X = [
            np.array(self.tokenizer.encode(x))
            for x in np.concatenate([data[0], data[2]])
        ]
        if len(X) > 5000:
            X = [X[i] for i in np.random.choice(len(X), size=5000, replace=False)]

        flat_X = np.concatenate(X).reshape(-1, 1)
        lengths = [len(x) for x in X]

        n_hidden = config.get("n_hidden", 10)
        self.hmm = MultinomialHMM(n_components=n_hidden, verbose=True, n_iter=100).fit(
            flat_X, lengths
        )

        with open(save_fn, "wb") as fd:
            pickle.dump(self.hmm, fd)

    def score(self, X):
        lengths = [len(x) for x in X]
        X = [np.array(self.tokenizer.encode(x)) for x in X]
        scores = np.array([self.hmm.score(np.array(x).reshape(-1, 1)) for x in X])
        scores = scores / lengths
        # print(scores)
        scores[np.isneginf(scores)] = -3 #np.amin(scores[~np.isneginf(scores)])
        return scores

class EmbedDistRewardFunc:
    def __init__(self, config, prototype_seq=utils.avGFP):
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()

        if torch.cuda.is_available():
            model = model.cuda()

        self.model = model

        self.prototype_seq = prototype_seq

        _, _, batch_tokens = self.batch_converter([(0, prototype_seq)])
        
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            rep = model(batch_tokens, repr_layers=[33])
        rep = rep["representations"][33].cpu()

        self.prototype_rep = rep.mean(1)

    def _get_embeddings(self, X, bs=16):
        X = list(zip(range(len(X)), X))
        _, batch_strs, batch_tokens = self.batch_converter(X)

        token_representations = []
        for i in range((len(batch_tokens) // bs) + 1):
            if i * bs >= len(batch_tokens):
                break

            _batch_tokens = batch_tokens[i * bs : (i + 1) * bs]

            if torch.cuda.is_available():
                _batch_tokens = _batch_tokens.cuda()

            with torch.no_grad():
                results = self.model(_batch_tokens, repr_layers=[33])
            token_representations.append(results["representations"][33].cpu())

        token_representations = torch.cat(token_representations, dim=0)

        sequence_representations = []
        for i, seq in enumerate(batch_strs):
            if i >= len(token_representations):
                break
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        embeddings = torch.stack(sequence_representations, dim=0).cpu()

        return embeddings


    def _dprod_dist(self, X, bs=16):
        embeddings = self._get_embeddings(X, bs=bs)

        dists = torch.bmm(
            self.prototype_rep.expand_as(embeddings).unsqueeze(1),
            embeddings.unsqueeze(-1),
        )[:, 0, 0]

        return dists.numpy()

    def _l2_dist(self, X, bs=16):
        embeddings = self._get_embeddings(X, bs=bs)

        dists = torch.cdist(
            self.prototype_rep.expand_as(embeddings).unsqueeze(1),
            embeddings.unsqueeze(1),
        )[:, 0, 0]

        return dists.numpy()

    def score(self, X, bs=16):
        dists = self._dprod_dist(X, bs=bs)
        return -1 * dists


class RBFKernelRewardFunc(EmbedDistRewardFunc):

    def __init__(self, config, prototype_seq=utils.avGFP):
        super().__init__(config, prototype_seq)

        self.temperature = config.get("temperature", 1e-1)

    def score(self, X, bs=16):
        print(self.prototype_seq[0])
        # dists = np.array([distance(self.prototype_seq[0], x) for x in X])
        dists = self._l2_dist(X, bs=bs)
        return np.exp(-self.temperature * dists)


class RNAfoldRewardFunc:
    def __init__(self, config):
        self.bin_fn = "./RNAfold" #os.path.join(os.environ["HOME"], "src/vienna_rna/usr/bin/RNAfold")
        if not os.path.exists(self.bin_fn):
            self.bin_fn = "../RNAfold"

    def _call_bin(self, S):
        str_input = "\n".join(S)
        p = subprocess.Popen(self.bin_fn, stdin=subprocess.PIPE, stdout=subprocess.PIPE) 
        ans=p.communicate(str_input.encode())
        p.terminate()
        scores = [float(x.split("(")[-1].split(")")[0]) for x in str(ans[0]).split("\\n")[1::2]]
        return scores

    def score(self, X, bs=500):
        rna = [mdb_utils.convert_aa_to_nucleotides(x).replace('T','U') for x in X]
        scores = []
        for idx in range(0, len(rna), bs):
            if idx+bs > len(rna):
                _scores = -1 * np.array(self._call_bin(rna[idx:]))
                assert len(_scores) == len(rna[idx:])
            else:
                _scores = -1 * np.array(self._call_bin(rna[idx:idx+bs]))
                assert len(_scores) == bs
            scores.append(_scores)
        return np.concatenate(scores)
        # return -1 * np.array(self._call_bin(rna))


class NoisyReward(object):
    def __init__(self, reward_fn, noise_model):
        self.reward_fn = reward_fn
        self.noise_model = noise_model

    def score(self, query_set, *args, **kwargs):
        true_reward = self.reward_fn.score(query_set, *args, **kwargs)
        noise_variance = self.noise_model(query_set)
        noise_samples = np.random.normal(size=query_set.shape)
        return true_reward + np.sqrt(noise_variance) * noise_samples
