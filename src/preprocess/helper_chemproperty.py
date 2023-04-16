# from rdkit import Chem
# from rdkit.Chem import rdchem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import Crippen
# from rdkit.Chem import MolFromSmarts
# from rdkit.Chem import MolFromSmiles
import numpy as np


def atom_number(smile):
    """
	Given the Smile, this function counts the number of atoms for each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
		- number of atoms (int): number of atoms in the chemical
	"""
    return sum(1 for c in smile if c.isupper())


def alone_atom_number(smile):
    """
	Given the Smile, this function counts the number of atoms, which are seperated from other atoms, in each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
		- number of separeted atoms (int): number of separeted atoms in the chemical 
	"""
    return smile.count("[")


def count_doubleBond(smile):
    """
	Given the smile, this function count the number of double bonds for each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
		- number of double bonds (int): number of double bonds in the chemical 
	"""
    return smile.count("=")


def count_tripleBond(smile):
    """
	Given the smile, this function count the number of triple bonds for each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
		- number of triple bonds (int): number of triple bonds in the chemical 
	"""
    return smile.count("#")


def bonds_number(smile):
    """
	Given the smile, this function count the number of bonds for each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
		- bonds number (int): number of bonds in the chemical (NaN if not found)
	"""
    m = Chem.MolFromSmiles(smile)
    try:
        return rdchem.Mol.GetNumBonds(m)
    except:
        return np.nan


def ring_number(smile):
    """
	Given the smile, this function count the number of ring in each Chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
		- ring_number (int): number of ring in the chemical (NaN if not found)
	"""
    m = Chem.MolFromSmiles(smile)
    try:
        f = rdchem.Mol.GetRingInfo(m)
        return f.NumRings()
    except:
        return np.nan


def Mol(smile):
    """
    Given the smile, this function compute the Mol for each Chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - Mol (float): mol number of the chemical (NaN if not found)
    """
    smile = str(smile)
    try:
        m = Chem.MolFromSmiles(smile)
        return Descriptors.MolWt(m)
    except:
        return np.nan


def MorganDensity(smile):
    """
	Given the Smile, this function compute the Morgan density for each Chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
		- Morgan Density (float): Morgan density of the chemical (NaN if not found)
	"""
    smile = str(smile)
    m = Chem.MolFromSmiles(smile)
    try:
        return Descriptors.FpDensityMorgan1(m)
    except:
        return np.nan


def LogP(smile):
    """
	Given the Smile, this function compute the partition coefficient for each Chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
		- LogP (float): partition coefficient of the chemical (NaN if not found)
	"""
    smile = str(smile)
    try:
        m = Chem.MolFromSmiles(smile)
        return Crippen.MolLogP(m)
    except:
        return np.nan


def OH_count(smile):
    """
    Given the SMILES, this function compute the number of OH group in the chemical
    """
    try:
        m = MolFromSmiles(smile)
        patt = MolFromSmarts("[OX2H]")
        return len(m.GetSubstructMatches(patt))
    except:
        return np.nan


def adding_smiles_features(dataframe):
    print("Finding atom number...")
    dataframe["atom_number"] = dataframe["smiles"].apply(atom_number)

    print("Finding number of alone atoms...")
    dataframe["alone_atom_number"] = dataframe["smiles"].apply(alone_atom_number)

    print("Finding single bounds number...")
    dataframe["bonds_number"] = dataframe["smiles"].apply(bonds_number)

    print("Finding double bounds number...")
    dataframe["doubleBond"] = dataframe["smiles"].apply(count_doubleBond)

    print("Finding triple bounds number...")
    dataframe["tripleBond"] = dataframe["smiles"].apply(count_tripleBond)

    print("Finding ring number...")
    dataframe["ring_number"] = dataframe["smiles"].apply(ring_number)

    print("Finding morgan density...")
    dataframe["MorganDensity"] = dataframe["smiles"].apply(MorganDensity)

    print("Finding partition number (LogP)...")
    dataframe["LogP"] = dataframe["smiles"].apply(LogP)

    print("Finding number of OH group...")
    dataframe["oh_count"] = dataframe["smiles"].apply(OH_count)

    print("Finding melting point...")
    dataframe = dataframe[~dataframe.melting_point.isnull()]

    print("Finding water solubility...")
    dataframe = dataframe[~dataframe.water_solubility.isnull()]

    return dataframe
