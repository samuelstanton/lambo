from typing import Optional, Dict

from Bio.PDB import PDBParser, SASA


class SurfaceArea:
    def __init__(self, probe_radius: float = 1.4, n_points: int = 100, radii_dict: Optional[Dict] = None):
        """
            Computes solvent accessible surface area (SASA) via calling the implementation of the 
            Shrake Rupley algorithm in Biopython

            Args (copied over from biopython docs):
                probe_radius (float) – radius of the probe in A. Default is 1.40, roughly the radius of a water molecule.
                n_points (int) – resolution of the surface of each atom. Default is 100. 
                    A higher number of points results in more precise measurements, but slows down the calculation.
                radii_dict (dict) – user-provided dictionary of atomic radii to use in the calculation. 
                    Values will replace/complement those in the default ATOMIC_RADII dictionary.

            TODO: assess these arguments and work out api a bit better

        """
        # this is a workaround for a bug in BioPython
        # see https://github.com/biopython/biopython/pull/3777
        if radii_dict is None:
            radii_dict = {'X': 2.0}

        self.parser = PDBParser(QUIET=1)
        self.structure_computer = SASA.ShrakeRupley(probe_radius=probe_radius, n_points=n_points, radii_dict=radii_dict)

    def __call__(self, name, loc) -> float:
        struct = self.parser.get_structure(name, loc)
        self.structure_computer.compute(struct, level="S")
        return struct.sasa
