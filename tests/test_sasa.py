from pathlib import Path

from bo_protein.tasks.proxy_rfp.sasa import SurfaceArea


def test_sasa_forwards():
    test_dir = Path(__file__).parent.resolve()
    test_pdb_asset = (test_dir / "./files/1ggx.pdb").as_posix()

    sasa = SurfaceArea()

    name = test_pdb_asset.split(".")[-2].split("/")[-1]
    value = sasa(name, test_pdb_asset)
    assert isinstance(value, float)
