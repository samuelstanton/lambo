import copy
import numpy as np
import pandas as pd
import os
import random

aa_3_to_1 = {
    "Cys": "C",
    "Asp": "D",
    "Ser": "S",
    "Gln": "Q",
    "Lys": "K",
    "Ile": "I",
    "Pro": "P",
    "Thr": "T",
    "Phe": "F",
    "Asn": "N",
    "Gly": "G",
    "His": "H",
    "Leu": "L",
    "Arg": "R",
    "Trp": "W",
    "Ala": "A",
    "Val": "V",
    "Glu": "E",
    "Tyr": "Y",
    "Met": "M",
    "Ter": "*",
}

codon_to_aa = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "_",
    "TAG": "_",
    "TGC": "C",
    "TGT": "C",
    "TGA": "_",
    "TGG": "W",
}

aa_to_codon = {
    "A": list("GCT,GCC,GCA,GCG".split(",")),
    "R": list("CGT,CGC,CGA,CGG,AGA,AGG".split(",")),
    "N": list("AAT,AAC".split(",")),
    "D": list("GAT,GAC".split(",")),
    "C": list("TGT,TGC".split(",")),
    "Q": list("CAA,CAG".split(",")),
    "E": list("GAA,GAG".split(",")),
    "G": list("GGT,GGC,GGA,GGG".split(",")),
    "H": list("CAT,CAC".split(",")),
    "I": list("ATT,ATC,ATA".split(",")),
    "L": list("TTA,TTG,CTT,CTC,CTA,CTG".split(",")),
    "K": list("AAA,AAG".split(",")),
    "M": list("ATG".split(",")),
    "F": list("TTT,TTC".split(",")),
    "P": list("CCT,CCC,CCA,CCG".split(",")),
    "S": list("TCT,TCC,TCA,TCG,AGT,AGC".split(",")),
    "T": list("ACT,ACC,ACA,ACG".split(",")),
    "W": list("TGG".split(",")),
    "Y": list("TAT,TAC".split(",")),
    "V": list("GTT,GTC,GTA,GTG".split(",")),
    "*": list("TAA,TGA,TAG".split(","))
}

brca1_ref_nucleo = "GATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGCCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAGGAGCCTACAAGAAAGTACGAGATTTAGTCAACTTGTTGAAGAGCTATTGAAAATCATTTGTGCTTTTCAGCTTGACACAGGTTTGGAGTATGCAAACAGCTATAATTTTGCAAAAAAGGAAAATAACTCTCCTGAACATCTAAAAGATGAAGTTTCTATCATCCAAAGTATGGGCTACAGAAACCGTGCCAAAAGACTTCTACAGAGTGAACCCGAAAATCCTTCCTTGCAGGAAACCAGTCTCAGTGTCCAACTCTCTAACCTTGGAACTGTGAGAACTCTGAGGACAAAGCAGCGGATACAACCTCAAAGGACGTCTGTCTACATTGAATTGGGATCTGATTCTTCTGAAGATACCGTTAATAAGGCAACTTATTGCAGTGTGGGAGATCAAGAATTGTTACAAATCACCCCTCAAGGAACCAGGGATGAAATCAGTTTGGATTCTGCAAAAAAGGCTGCTTGTGAATTTTCTGAGACGGATGTAACAAATACTGAACATCATCAACCCAGTAATAATGATTTGAACACCACTGAGAAGCGTGCAGCTGAGAGGCATCCAGAAAAGTATCAGGGTAGTTCTGTTTCAAACTTGCATGTGGAGCCATGTGGCACAAATACTCATGCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACTAAAGACAGAATGAATGTAGAAAAGGCTGAGTTC"

brca1_ref_aa = "DLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQRTSVYIELGSDSSEDTVNKATYCSVGDQELLQITPQGTRDEISLDSAKKAACEFSETDVTNTEHHQPSNNDLNTTEKRAAERHPEKYQGSSVSNLHVEPCGTNTHASSLQHENSSLLLTKDRMNVEKAEF"

e4b_ref_nucleo = "ATAGAGAAGTTTAAACTTCTTGCAGAGAAAGTGGAGGAAATCGTGGCAAAGAATGCGCGGGCAGAAATAGACTACAGCGATGCCCCGGACGAGTTCAGAGACCCTCTGATGGACACCCTGATGACCGATCCCGTGAGACTGCCCTCTGGCACCGTCATGGACCGTTCTATCATCCTGCGGCATCTGCTCAACTCCCCCACCGACCCCTTCAACCGCCAGATGCTGACTGAGAGCATGCTGGAGCCAGTGCCAGAGCTAAAGGAGCAGATTCAGGCCTGGATGAGAGAGAAACAGAGCAGTGACCACTGA"

e4b_ref_aa = "IEKFKLLAEKVEEIVAKNARAEIDYSDAPDEFRDPLMDTLMTDPVRLPSGTVMDRSIILRHLLNSPTDPFNRQMLTESMLEPVPELKEQIQAWMREKQSSDH"

ube2i_ref_nucleo = "ATGTCGGGGATCGCCCTCAGCAGACTCGCCCAGGAGAGGAAAGCATGGAGGAAAGACCACCCATTTGGTTTCGTGGCTGTCCCAACAAAAAATCCCGATGGCACGATGAACCTCATGAACTGGGAGTGCGCCATTCCAGGAAAGAAAGGGACTCCGTGGGAAGGAGGCTTGTTTAAACTACGGATGCTTTTCAAAGATGATTATCCATCTTCGCCACCAAAATGTAAATTCGAACCACCATTATTTCACCCGAATGTGTACCCTTCGGGGACAGTGTGCCTGTCCATCTTAGAGGAGGACAAGGACTGGAGGCCAGCCATCACAATCAAACAGATCCTATTAGGAATACAGGAACTTCTAAATGAACCAAATATCCAAGACCCAGCTCAAGCAGAGGCCTACACGATTTACTGCCAAAACAGAGTGGAGTACGAGAAAAGGGTCCGAGCACAAGCCAAGAAGTTTGCGCCCTCATAA"

ube2i_ref_aa = "MSGIALSRLAQERKAWRKDHPFGFVAVPTKNPDGTMNLMNWECAIPGKKGTPWEGGLFKLRMLFKDDYPSSPPKCKFEPPLFHPNVYPSGTVCLSILEEDKDWRPAITIKQILLGIQELLNEPNIQDPAQAEAYTIYCQNRVEYEKRVRAQAKKFAPS"

yap65_ref_nucleo = "GACGTTCCACTGCCGGCTGGTTGGGAAATGGCTAAAACTAGTTCTGGTCAGCGTTACTTCCTGAACCACATCGACCAGACCACCACGTGGCAGGACCCGCGT"

yap65_ref_aa = "DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPR"


def convert_nucleotides_to_aa(nucleotide_seq):
    aa_seq = ""
    for i in range(0, len(nucleotide_seq), 3):
        aa_seq += codon_to_aa[nucleotide_seq[i : i + 3]]
    return aa_seq

def convert_aa_to_nucleotides(aa_seq):
    na_seq = [aa_to_codon[c][0] for c in aa_seq]
    # na_seq = [random.choice(aa_to_codon.get(c, ["---"])) for c in aa_seq]
    return "".join(na_seq)


def str_to_mutation(base_seq, s):
    s = s.replace("p.", "").replace("]", "").replace("[", "")
    mutations = list(set(s.split(";")) - {"="})
    if len(mutations) == 0:
        return None

    swaps = {}
    for m in mutations:
        if len(m) <= 6 or "=" in m:
            return None

        aa1 = aa_3_to_1[m[:3]]
        loc = int(m[3:-3])
        aa2 = aa_3_to_1[m[-3:]]

        if loc > len(base_seq):
            print(s)

        assert aa1 == base_seq[loc - 1]

        swaps[loc - 1] = aa2

    return swaps


def mut_df_to_seq_df(base_seq, mut_df):
    labeled_seqs = []
    for mut_str, score in mut_df.values:
        swaps = str_to_mutation(base_seq, mut_str)
        if swaps is None or "*" in swaps.values():
            continue
        seq = copy.deepcopy(base_seq)
        for loc in sorted(swaps.keys()):
            seq = seq[:loc] + swaps[loc] + seq[loc + 1 :]

        labeled_seqs.append([seq, score, len(swaps)])

    df = pd.DataFrame(labeled_seqs, columns=["seq", "score", "num_mutations"])
    return df


def _load_data(root, source, cutoff_dist=20.0):
    file_path = os.path.join(root, f"mavedb_data/{source}_data/{source}_processed.csv")
    df = pd.read_csv(file_path)
    X, Y = df["seq"].to_numpy(), df["score"].to_numpy()
    mutations = df["num_mutations"].to_numpy()
    return X, Y, mutations


def _normalize(Y):
    Y_mean, Y_std = Y.mean(), Y.std()
    Y = (Y - Y_mean) / Y_std
    return Y


def _shuffle(X, Y, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]
    return X, Y, idx


def _divide_by_mutations(X, Y, mutations, lower_perc=0.5, upper_perc=0.5):
    mutations_upper = np.quantile(mutations, lower_perc)
    mutations_lower = np.quantile(mutations, 1 - upper_perc)

    train_mask = mutations <= mutations_upper
    test_mask = mutations > mutations_lower

    X, Y, idx = _shuffle(X, Y)

    mutations = mutations[idx]
    train_mask = train_mask[idx]
    test_mask = test_mask[idx]

    return (X[train_mask], Y[train_mask], mutations[train_mask]), (
        X[test_mask],
        Y[test_mask],
        mutations[test_mask],
    )


def load_data(root, source, split=0.9, train_wo_cutoff=False, seed=0):
    X, Y, _ = _load_data(root, source)
    X, Y, _ = _shuffle(X, Y, seed=seed)

    num_train = int(split * len(X))
    X_train, Y_train = X[:num_train], Y[:num_train]
    X_test, Y_test = X[num_train:], Y[num_train:]

    if train_wo_cutoff:
        X, Y, _ = _load_data(root, source, cutoff_dist=float("inf"))
        X_train = np.concatenate([X_train, X])
        Y_train = np.concatenate([Y_train, Y])

    Y = _normalize(np.concatenate([Y_train, Y_test]))
    Y_train, Y_test = Y[: len(Y_train)], Y[len(Y_train) :]

    return (X_train, Y_train), (X_test, Y_test)


def load_data_mutation_split(
    root, source, lower_perc=0.5, upper_perc=0.5, train_wo_cutoff=False
):
    assert (lower_perc + upper_perc) <= 1.0
    X, Y, mutations = _load_data(source)
    splits = _divide_by_mutations(X, Y, mutations)

    if train_wo_cutoff:
        X, Y, mutations = _load_data(root, source, cutoff_dist=float("inf"))

        train, test = splits
        X_train, Y_train, mutations_train = train

        X_train = np.concatenate([X_train, X])
        Y_train = np.concatenate([Y_train, Y])
        mutations = np.concatenate([mutations_train, mutations])

        train = (X_train, Y_train, mutations)
        splits = (train, test)

    (X_train, Y_train, mutations_train), (X_test, Y_test, mutations_test) = splits
    Y = _normalize(np.concatenate([Y_train, Y_test]))
    Y_train, Y_test = Y[: len(Y_train)], Y[len(Y_train) :]
    splits = (X_train, Y_train, mutations_train), (X_test, Y_test, mutations_test)

    return splits
