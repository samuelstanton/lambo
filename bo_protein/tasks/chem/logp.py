from rdkit import Chem
from rdkit.Chem.QED import qed as qed_fn
from rdkit.Chem import Crippen
import networkx as nx
from rdkit.Chem import rdmolops

# My imports
from bo_protein.tasks.chem.SA_Score import sascorer


def get_mol(smiles_or_mol):
    '''                                                                                                                                       
    Loads SMILES/molecule into RDKit's object                                   
    '''                                                                                                                                       
    if isinstance(smiles_or_mol, str):                                          
        if len(smiles_or_mol) == 0:                                              
            return None                                                           
        mol = Chem.MolFromSmiles(smiles_or_mol)                                 
        if mol is None:                                                          
            return None                                                           
        try:                                                                    
            Chem.SanitizeMol(mol)                                                 
        except ValueError:                                                      
            return None                                                           
        return mol                                                              
    return smiles_or_mol

def standardize_smiles(smiles):
    """ Get standard smiles without stereo information """
    mol = get_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def SA(mol):
    return sascorer.calculateScore(mol)

def logP(smiles: str, min_score=-float("inf"), penalized=False) -> float:
    """ calculate penalized logP for a given smiles string """
    if smiles is None:
        return min_score

    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)

    if not penalized:
        return max(logp, min_score)

    sa = SA(mol)

    # Calculate cycle score
    cycle_length = _cycle_score(mol)

    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    score = (
        (logp - 2.45777691) / 1.43341767
        + (-sa + 3.05352042) / 0.83460587
        + (-cycle_length - -0.04861121) / 0.28746695
    )
    return max(score, min_score)


def _cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def prop_func(s, property_list, invalid_score=-100):
    mol = Chem.MolFromSmiles(s)

    # s = s if mol is not None else None

    mol_properties = [invalid_score for _ in property_list]
    if mol is None:
        return mol_properties

    for p_idx, p_name in enumerate(property_list):
        canonic_smiles = standardize_smiles(s)
        if p_name == 'penalized_logP':
            pen_logp = logP(canonic_smiles, penalized=True, min_score=-4)
            mol_properties[p_idx] = pen_logp
        elif p_name == 'logP':
            logp = logP(canonic_smiles, penalized=False)
            mol_properties[p_idx] = logp
        elif p_name == 'qed':
            mol_properties[p_idx] = qed_fn(mol)
        else:
            raise ValueError('unsupported molecule property')

    return mol_properties
