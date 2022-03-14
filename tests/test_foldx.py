from Bio import PDB
from pathlib import Path
from bo_protein.foldx_data.foldx import FoldxManager
from bo_protein.utils import FoldxMutation


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


def test_foldx(pdb_loc):
    pdb_path = extract_chain(pdb_loc, chain="A")
    work_dir = Path(__file__).parent / "tmp"
    work_dir.mkdir(exist_ok=True)

    # Create FoldX Manager and initialize working directory
    # Note there is a default foldx directory in the user space,
    # set it manually if needed
    foldx_manager = FoldxManager(wt_pdb=pdb_path, work_dir=work_dir)

    # Note that the first 6 amino acids are not included in the crystallographic model
    # So the first mutatable residue is number 7
    # Here is how we would pass 2 mutations to the FoldX manager
    mutation_list = [
        FoldxMutation(old_token_idx="V", chain="A", token_pos=7, new_token_idx="G"),
        FoldxMutation(old_token_idx="K", chain="A", token_pos=9, new_token_idx="L"),
    ]
    ddG = foldx_manager(mutation_list)
    print(f"{ddG} kcal/mol")

    # And Again
    mutation_list = [
        FoldxMutation(old_token_idx="V", chain="A", token_pos=7, new_token_idx="L"),
        FoldxMutation(old_token_idx="K", chain="A", token_pos=9, new_token_idx="G"),
    ]
    ddG = foldx_manager(mutation_list)
    print(f"{ddG} kcal/mol")
    print(f"full cache saved at  {foldx_manager.cache_out}")


if __name__ == "__main__":
    test_foldx("./tests/files/1ggx.pdb")