import platform
import os
import shutil
from pathlib import Path
from subprocess import PIPE, Popen
import logging
import uuid
from Bio import PDB

_name = "FoldX"
if platform.system() == "Windows":
    _bin_name = "foldx.exe"
else:
    _bin_name = "foldx"


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


class FoldxManager:
    def __init__(
        self, *, wt_pdb, foldx_dir=None, work_dir="./", skip_minimization=False, ph=7.0
    ):
        """
        Callable FoldX management for a given wild type.
        Fills a working directory with subdirectories of simulations, named via UUID.

        The initialization sets up the directories, working space, and initial relaxation
        of the protein structure.

        Calling an instance of the manager on a list of mutations should be done using the
        Mutation named tuple.

        Parameters
        ----------
        wt_pdb: str, Path
            Path to wild type pdb file.
            This is copied into the working directory as 'wt_input.pdb'
        foldx_dir: str, Path
            Directory containing FoldX executable and rotabase.txt file
        work_dir: str, Path
            Top level working directory for storing the simulations and cache
        skip_minimization: bool
            Optional skipping of initial FoldX 'repair'.
            Not suggested for use, but available if insisted on.
        ph: float
            System pH for FoldX simulations. Defaults to 7.0.
        """
        if foldx_dir is None:
            self.bin_dir = Path("~/foldx").expanduser()
        self.work_dir = Path(work_dir).absolute()
        self.skip_minimization = skip_minimization
        self.wt_pdb = Path(wt_pdb).expanduser()
        os.makedirs(self.work_dir, exist_ok=True)
        shutil.copyfile(self.wt_pdb, self.work_dir / "wt_input.pdb")
        self.ph = ph

        # Hold run ncache in object, and append to file to allow restarts
        self.cache = {}
        self.cache_out = self.work_dir / "cache.csv"
        if not self.cache_out.is_file():
            with open(self.cache_out, "w") as f:
                f.write("uuid,mutation_list,wild_total_energy,mutant_total_energy\n")

        # Necessary for FoldX
        self._check_files()
        try:
            os.symlink(self.bin_dir / "rotabase.txt", self.work_dir / "rotabase.txt")
        except FileExistsError:
            pass

        # Repair with FoldX to start
        self.repair_cmd = [
            "--pdb",
            "wt_input.pdb",
            "--command",
            "RepairPDB",
            "--water",
            "-CRYSTAL",
            "--pH",
            f"{self.ph:.2f}",
        ]
        self.minimize_energy()

        # Hold permanent command for mutation runs
        self.build_mutant_cmd = [
            "--pdb",
            "wt_input_Repair.pdb",
            "--command",
            "BuildModel",
            "--pH",
            f"{self.ph:.2f}",
        ]

    def __call__(self, mutation_list, id=None):
        """
        Performs a single FoldX simulation on a mutant given a list of mutations
        separating the mutant from the wild type.

        Appends to the disk written cache the uuid, mutations, and result

        Parameters
        ----------
        mutation_list: list of Mutation
        id: UUID hex string (optional)

        Returns
        -------
        ddG: float
            Resulting change in folding free energy from mutation simulation

        """
        tag = self.build_mutant(mutation_list, id)
        metrics = self.read_result(self.work_dir / id)
        with open(self.cache_out, "a") as f:
            f.write(f"{tag},{mutation_list},{metrics['wild_total_energy']},{metrics['mutant_total_energy']}\n")
        return metrics

    def minimize_energy(self):
        """FoldX Repair to initialize a protein"""
        cwd = os.getcwd()
        os.chdir(self.work_dir)
        fold_cmd = [str(self.bin_dir / _bin_name)] + self.repair_cmd
        if not self.skip_minimization:
            print(' '.join(fold_cmd))
            self._run_foldx(cmd=fold_cmd, cwd=self.work_dir, outfile="finalRepair.out")
        else:
            shutil.move(
                self.work_dir / "wt_input.pdb", self.work_dir / "wt_input_Repair.pdb"
            )
        os.chdir(cwd)

    def build_mutant(self, mutation_list, id=None):
        """
        Builds mutant using FoldX in subdirectory named by UUID
        Parameters
        ----------
        mutation_list: list of Mutation
        id: UUID hex string (optional)

        Returns
        -------

        """
        cwd = os.getcwd()
        os.chdir(self.work_dir)
        id = uuid.uuid4().hex if id is None else id
        self.cache[id] = mutation_list
        subdir = self.work_dir / id
        subdir.mkdir(exist_ok=True)
        self._create_foldx_mutation_list(subdir / "individual_list.txt", mutation_list)

        foldx_cmd = (
            [str(self.bin_dir / _bin_name)]
            + self.build_mutant_cmd
            + [
                "--mutant-file",
                f"{subdir / 'individual_list.txt'}",
                "--output-dir",
                f"{subdir}",
            ]
        )

        self._run_foldx(cmd=foldx_cmd, cwd=self.work_dir, outfile=f"{subdir / 'buildFoldxMutant.out'}")
        os.chdir(cwd)
        return id

    @staticmethod
    def read_result(foldx_dir):
        """Get ddG result from subdir in kcal/mol"""
        try:
            with open(Path(foldx_dir) / "Raw_wt_input_Repair.fxout", "r") as f:
                mutant_metrics, wild_metrics = f.readlines()[-2:]
        except FileNotFoundError:
            raise RuntimeError('FoldX BuildModel call failed.')
        metrics = dict(
            mutant_total_energy=float(mutant_metrics.split()[1]),
            wild_total_energy=float(wild_metrics.split()[1]),
        )
        return metrics

    @staticmethod
    def _create_foldx_mutation_list(path, mutation_list):
        """
        Construct individual_list.txt file of mutations for FoldX run
        Parameters
        ----------
        path: str, Path
            Output path that has name beginning with 'individual_list'
        mutation_list: list of Mutation
            list of Mutation named tuples

        Returns
        -------

        """
        if Path(path).stem != "individual_list":
            raise RuntimeError(
                "FoldX mutation list must be in file named individual_list"
            )

        with open(path, "w") as f:
            f.write(
                ",".join(
                    [
                        f"{mutation.wt_residue}"
                        f"{mutation.chain}"
                        f"{mutation.residue_number}"
                        f"{mutation.mutant_residue}"
                        for mutation in mutation_list
                    ]
                )
            )
            f.write(";")

    @staticmethod
    def _run_foldx(cmd, cwd, outfile):
        with open(outfile, "wb") as f:
            out, err = Popen(cmd, stdout=f, stderr=PIPE,  cwd=cwd).communicate()
        if len(err.decode("utf-8")) > 0:
            logging.error(f"FoldX error: {err.decode('utf-8')}")

    def _check_files(self):
        """Sanity check to make sure all files and executables are accessible"""
        if not (
            (self.bin_dir / "rotabase.txt").is_file()
            and (self.bin_dir / _bin_name).is_file()
        ):
            raise FileNotFoundError(
                f"Provided FoldX directory ({self.bin_dir}) is missing either "
                f"rotabase.txt or {_bin_name}.\n"
                f"These files are required for FoldX to run"
            )
