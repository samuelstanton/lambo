from Bio import PDB
from pathlib import Path
from bo_protein.foldx_data.foldx import FoldxManager
from bo_protein.candidate import pdb_to_residues
from bo_protein.utils import FoldxMutation, RESIDUE_ALPHABET, IntTokenizer


class SelectChains(PDB.Select):
    """ Only accept the specified chains when saving. """

    def __init__(self, chain_letters):
        self.chain_letters = chain_letters

    def accept_chain(self, chain):
        return chain.get_id() in self.chain_letters


def extract_chain(path, chain="A"):
    """
    Extracts a chain from a PDB file and writes a new PDB file with just that chain

    Parameters
    ----------
    path: Path, str
        Path to initial pdb file
    chain: str

    Returns
    -------

    """
    parser = PDB.PDBParser()
    writer = PDB.PDBIO()
    path = Path(path).expanduser()
    struct = parser.get_structure(path.stem, path)
    writer.set_structure(struct)
    out_path = path.parent / f"{path.stem}_{chain}.pdb"
    writer.save(str(out_path), select=SelectChains([chain]))
    return out_path


def test_foldx():
    test_dir = Path(__file__).parent.resolve()
    test_pdb_asset = (test_dir / "./files/1ggx.pdb")
    pdb_path = extract_chain(test_pdb_asset, chain="A")
    work_dir = Path(__file__).parent / "tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    residue_seq, pos_seq = pdb_to_residues(pdb_path, chain_id='A')
    print(residue_seq)
    print(pos_seq)

    # Create FoldX Manager and initialize working directory
    # Note there is a default foldx directory in the user space,
    # set it manually if needed
    foldx_manager = FoldxManager(wt_pdb=pdb_path, work_dir=work_dir)

    #
    tokenizer = IntTokenizer(RESIDUE_ALPHABET, RESIDUE_ALPHABET)

    # Note that the first 6 amino acids are not included in the crystallographic model
    # So the first mutatable residue is number 7
    # Here is how we would pass 2 mutations to the FoldX manager
    mutation_list = [
        FoldxMutation(
            old_token_idx=tokenizer.encode("V")[1],
            chain="A",
            token_pos=pos_seq[0],
            new_token_idx=tokenizer.encode("G")[1],
            tokenizer=tokenizer
        ),
        FoldxMutation(
            old_token_idx=tokenizer.encode("K")[1],
            chain="A",
            token_pos=pos_seq[2],
            new_token_idx=tokenizer.encode("L")[1],
            tokenizer=tokenizer
        ),
    ]
    ddG = foldx_manager(mutation_list, id='1ggx')
    print(f"{ddG} kcal/mol")
    print(f"full cache saved at  {foldx_manager.cache_out}")
