import copy
import uuid
from Bio import PDB
from pathlib import Path
from bo_protein.tasks.proxy_rfp.foldx import FoldxManager
from bo_protein.tasks.proxy_rfp.sasa import SurfaceArea
from Bio.SeqUtils import seq1

from bo_protein.utils import StringSubstitution, StringDeletion, StringInsertion, FoldxMutation


def apply_mutation(base_seq, mut_pos, mut_res, tokenizer):
    tokens = tokenizer.decode(tokenizer.encode(base_seq)).split(" ")[1:-1]
    mut_seq = "".join(tokens[:mut_pos] + [mut_res] + tokens[(mut_pos + 1):])
    return mut_seq


def pdb_to_residues(pdb_path, chain_id='A'):
    """
    :param pdb_path: path to pdb file (str or Path)
    :param chain_id: (str)
    :return: residues: (Bio.Seq)
    :return: idxs: (list) residue indexes in the PDB
    """
    parser = PDB.PDBParser()
    pdb_path = Path(pdb_path).expanduser()
    struct = parser.get_structure(pdb_path.stem, pdb_path)
    chain_residues = {
        chain.get_id(): seq1(''.join(x.resname for x in chain)) for chain in struct.get_chains()
    }
    chain_idxs = {
        chain.get_id(): [x.get_id()[1] for x in chain] for chain in struct.get_chains()
    }
    residues = chain_residues[chain_id]
    idxs = chain_idxs[chain_id]
    return residues, idxs


class StringCandidate:
    def __init__(self, wild_seq, mutation_list, tokenizer, wild_name=None, dist_from_wild=0.):
        self.wild_residue_seq = wild_seq
        self.uuid = uuid.uuid4().hex
        self.wild_name = 'unnamed' if wild_name is None else wild_name
        self.mutant_residue_seq = self.apply_mutations(mutation_list, tokenizer)
        self.dist_from_wild = dist_from_wild
        self.tokenizer = tokenizer

    def __len__(self):
        tok_idxs = self.tokenizer.encode(self.mutant_residue_seq)
        return len(tok_idxs)

    def apply_mutations(self, mutation_list, tokenizer):
        if len(mutation_list) == 0:
            return self.wild_residue_seq

        mutant_seq = copy.deepcopy(self.wild_residue_seq)
        mutant_seq = tokenizer.encode(mutant_seq)[1:-1]
        for mutation_op in mutation_list:
            old_tok_idx = mutation_op.old_token_idx
            mut_pos = mutation_op.token_pos
            if mut_pos < len(mutant_seq):
                assert old_tok_idx == mutant_seq[mut_pos], str(mutation_op)
            if isinstance(mutation_op, StringSubstitution):
                new_tok_idx = mutation_op.new_token_idx
                mutant_seq = mutant_seq[:mut_pos] + [new_tok_idx] + mutant_seq[mut_pos + 1:]
            elif isinstance(mutation_op, StringDeletion):
                mutant_seq = mutant_seq[:mut_pos] + mutant_seq[mut_pos + 1:]
            elif isinstance(mutation_op, StringInsertion):
                new_tok_idx = mutation_op.new_token_idx
                mutant_seq = mutant_seq[:mut_pos] + [new_tok_idx] + mutant_seq[mut_pos:]
            else:
                raise RuntimeError('unrecognized mutation op')

        mutant_seq = tokenizer.decode(mutant_seq).replace(" ", "")
        return mutant_seq

    def new_candidate(self, mutation_list, tokenizer):
        cand_kwargs = dict(
            wild_seq=self.mutant_residue_seq,
            mutation_list=mutation_list,
            tokenizer=tokenizer,
            wild_name=self.wild_name,
            dist_from_wild=self.dist_from_wild + len(mutation_list),
        )
        return StringCandidate(**cand_kwargs)


class FoldedCandidate:
    def __init__(self, work_dir, wild_pdb_path, mutation_list, tokenizer,
                 skip_minimization=True, chain='A', wild_name=None, dist_from_wild=0.):
        """
        :param work_dir: (str or Path) output directory
        :param wild_pdb_path: (str or Path) path to pdb file (recommended to use FoldX repaired PDBs)
        :param mutation_list: list of Mutation named tuples
        :param skip_minimization: (bool) set to False to repair the wild PDB
        """
        self.work_dir = work_dir
        self.uuid = uuid.uuid4().hex
        self.mutation_list = mutation_list
        self.tokenizer = tokenizer
        self.chain = chain
        self.wild_name = 'unnamed' if wild_name is None else wild_name
        self.dist_from_wild = dist_from_wild
        foldx_manager = FoldxManager(wt_pdb=wild_pdb_path, work_dir=work_dir,
                                     skip_minimization=skip_minimization)

        # pass dummy mutation to foldx_manager if mutation_list is empty
        if len(mutation_list) == 0:
            wild_seq, wild_idxs = pdb_to_residues(wild_pdb_path, self.chain)
            tokens = tokenizer.encode(wild_seq)[1:-1]
            mutation_list = [
                # FoldxMutation(wild_seq[0], self.chain, wild_idxs[0], wild_seq[0])
                FoldxMutation(tokens[0], chain, wild_idxs[0], tokens[0], tokenizer)
            ]

        foldx_success = True
        try:
            metrics = foldx_manager(mutation_list, self.uuid)
            self.wild_pdb_path = Path(work_dir) / self.uuid / 'WT_wt_input_Repair_1.pdb'
            self.mutant_pdb_path = Path(work_dir) / self.uuid / 'wt_input_Repair_1.pdb'
        except RuntimeError:
            print(f'{wild_name}, {mutation_list}')
            foldx_success = False
            self.wild_pdb_path = wild_pdb_path
            self.mutant_pdb_path = None

        # predicted stability
        self.wild_total_energy = metrics['wild_total_energy'] if foldx_success else float('inf')
        self.mutant_total_energy = metrics['mutant_total_energy'] if foldx_success else float('inf')

        # solvent-accessible surface area
        sasa_fn = SurfaceArea()
        self.wild_surface_area = sasa_fn(self.uuid, self.wild_pdb_path)
        self.mutant_surface_area = sasa_fn(self.uuid, self.mutant_pdb_path) if foldx_success else -float('inf')

        # residue sequences
        self.wild_residue_seq, self.wild_residue_idxs = pdb_to_residues(
            self.wild_pdb_path,
            self.chain
        )
        self.mutant_residue_seq, self.mutant_residue_idxs = pdb_to_residues(
            self.mutant_pdb_path,
            self.chain
        ) if foldx_success else (None, None)

    def __len__(self):
        tok_idxs = self.tokenizer.encode(self.mutant_residue_seq)
        return len(tok_idxs)

    def new_mutation(self, seq_idx, mutant_residue, mutation_type='sub'):
        """
        formats the desired sequence substitution into a compatible FoldX mutation object
        :param seq_idx: position in the residue sequence
        :param mutant_residue: residue to be substituted
        :return: Mutation
        """
        assert mutation_type == 'sub', 'Foldx only allows substitutions'
        seq_idx = seq_idx % len(self.mutant_residue_seq)  # make indexes wrap around
        mutation_kwargs = dict(
            old_token_idx=self.tokenizer.encode(self.mutant_residue_seq[seq_idx])[1],
            chain=self.chain,
            token_pos=self.mutant_residue_idxs[seq_idx],
            new_token_idx=self.tokenizer.encode(mutant_residue)[1],
            tokenizer=self.tokenizer,
        )
        return FoldxMutation(**mutation_kwargs)

    def new_candidate(self, mutation_list):
        """
        Mutates the current mutant type into a new mutant type.
        Mutations should be formatted with `self.new_mutation`
        :param mutation_list: [Mutation objects]
        :return: MutationCandidate
        """
        mutation_ops = []
        for op in mutation_list:
            if isinstance(op, FoldxMutation):
                mutation_ops.append(op)
            elif isinstance(op, StringSubstitution):
                mutation_ops.append(self.new_mutation(
                    op.token_pos, op.new_token, mutation_type='sub'
                ))
            else:
                raise ValueError

        cand_kwargs = dict(
            work_dir=self.work_dir,
            wild_pdb_path=self.mutant_pdb_path,
            mutation_list=mutation_ops,
            tokenizer=self.tokenizer,
            skip_minimization=True,
            chain=self.chain,
            wild_name=self.wild_name,
            dist_from_wild=self.dist_from_wild + len(mutation_ops),
        )
        return FoldedCandidate(**cand_kwargs)
