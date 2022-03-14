from bo_protein.candidate import StringCandidate, StringSubstitution, StringInsertion, StringDeletion
from bo_protein.utils import mutation_list, IntTokenizer


def check_candidates(b_seq, n_seq, manual_cand, auto_cand):
    assert manual_cand.wild_residue_seq == auto_cand.wild_residue_seq
    assert manual_cand.wild_residue_seq == b_seq

    assert manual_cand.mutant_residue_seq == auto_cand.mutant_residue_seq
    assert manual_cand.mutant_residue_seq == n_seq


VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', '[PAD]', '[MASK]', '[UNK]']


def test_string_substitution():
    tokenizer = IntTokenizer(VOCAB, VOCAB)
    b_seq = 'ABA'
    b_cand = StringCandidate(b_seq, mutation_list=[], tokenizer=tokenizer)

    n_seq = 'ACA'
    manual_mut_ops = [
        StringSubstitution(
            old_token_idx=1, token_pos=1, new_token_idx=2, tokenizer=tokenizer
        ),
    ]
    manual_cand = b_cand.new_candidate(manual_mut_ops, tokenizer)

    auto_mut_ops = mutation_list(b_seq, n_seq, tokenizer)
    auto_cand = b_cand.new_candidate(auto_mut_ops, tokenizer)

    check_candidates(b_seq, n_seq, manual_cand, auto_cand)


def test_string_insertion():
    tokenizer = IntTokenizer(VOCAB, VOCAB)
    b_seq = 'ABA'
    b_cand = StringCandidate(b_seq, mutation_list=[], tokenizer=tokenizer)

    n_seq = 'ACBA'
    manual_mut_ops = [
        StringInsertion(
            old_token_idx=1, token_pos=1, new_token_idx=2, tokenizer=tokenizer
        ),
    ]
    manual_cand = b_cand.new_candidate(manual_mut_ops, tokenizer)

    auto_mut_ops = mutation_list(b_seq, n_seq, tokenizer)
    auto_cand = b_cand.new_candidate(auto_mut_ops, tokenizer)

    check_candidates(b_seq, n_seq, manual_cand, auto_cand)


def test_string_deletion():
    tokenizer = IntTokenizer(VOCAB, VOCAB)
    b_seq = 'ABA'
    b_cand = StringCandidate(b_seq, mutation_list=[], tokenizer=tokenizer)

    n_seq = 'AB'
    manual_mut_ops = [
        StringDeletion(old_token_idx=0, token_pos=2, tokenizer=tokenizer),
    ]
    manual_cand = b_cand.new_candidate(manual_mut_ops, tokenizer)

    auto_mut_ops = mutation_list(b_seq, n_seq, tokenizer)
    auto_cand = b_cand.new_candidate(auto_mut_ops, tokenizer)

    check_candidates(b_seq, n_seq, manual_cand, auto_cand)


def test_multiple_mutation():
    tokenizer = IntTokenizer(VOCAB, VOCAB)
    b_seq = 'AFBFAFC'
    b_cand = StringCandidate(b_seq, mutation_list=[], tokenizer=tokenizer)

    n_seq = 'FDBFDAFE'

    auto_mut_ops = mutation_list(b_seq, n_seq, tokenizer)
    # [print(op) for op in auto_mut_ops]
    auto_cand = b_cand.new_candidate(auto_mut_ops, tokenizer)

    manual_mut_ops = [
        StringDeletion(old_token_idx=0, token_pos=0, tokenizer=tokenizer),
        StringInsertion(old_token_idx=1, token_pos=1, new_token_idx=3, tokenizer=tokenizer),
        StringInsertion(old_token_idx=0, token_pos=4, new_token_idx=3, tokenizer=tokenizer),
        StringSubstitution(old_token_idx=2, token_pos=7, new_token_idx=4, tokenizer=tokenizer),
    ]
    manual_cand = b_cand.new_candidate(manual_mut_ops, tokenizer)

    check_candidates(b_seq, n_seq, manual_cand, auto_cand)
