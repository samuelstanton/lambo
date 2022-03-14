from bo_protein.bp_data.sasa import SurfaceArea

def test_sasa_forwards(loc):
    sasa = SurfaceArea()

    name = loc.split(".")[-2].split("/")[-1]
    value = sasa(name, loc)
    print("Successfully computed surface area: ", value)

test_sasa_forwards("./files/1ggx.pdb")