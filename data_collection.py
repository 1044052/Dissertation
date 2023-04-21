# -*- coding: utf-8 -*-
"""Data Collection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hli9gQ3sb-iMOpmIoVzfQtSDec0bxtay
"""

import getpass 

user = input("DrugBank username: ")
pwd = getpass.getpass("DrugBank password: ")

!wget --user $user --password $pwd https://go.drugbank.com/releases/5-1-8/downloads/all-3d-structures
!wget --user $user --password $pwd https://go.drugbank.com/releases/5-1-10/downloads/all-metabolite-structures

!time pip install -q condacolab  # ~5s to run

import condacolab
condacolab.install()
!which conda
!conda --version
!which mamba
!mamba --version
!time mamba install -y -c rdkit rdkit
import numpy as np
import rdkit
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import rdEHTTools
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
!mamba install oddt
!mamba install -c conda-forge openbabel

# Commented out IPython magic to ensure Python compatibility.
# %ls
!unzip all-3d-structures

!mv 3D\ structures.sdf all-drugbank-3D.sdf
# %ls
filename = 'all-drugbank-3D.sdf'
drugbank = PandasTools.LoadSDF(filename)

ID_values = drugbank.DRUGBANK_ID
ID_values
drugbank.MOLECULAR_WEIGHT = pd.to_numeric(drugbank.MOLECULAR_WEIGHT)
df = drugbank.loc[drugbank.MOLECULAR_WEIGHT <= 1000]
# 9133 rows of small molecule drugs
# df is the dataframe with only small molecule values
# 4257 don't have a pka value
# 4876 values for the data set
test_poss_index = []
train_poss_index = []
for i in range(len(df.JCHEM_PKA)):
  pkas = df.JCHEM_PKA.values
  if str(float(pkas[i])) == 'nan':
    test_poss_index.append(i)
  else:
    train_poss_index.append(i)

test_set_idx = [x for x in test_poss_index[:1003] if x not in [1011, 1012]]
train_set_idx = [x for x in train_poss_index if x not in [1352, 3624, 6358, 6408, 6526, 8058, 8602, 8738, 8917]]

from google.colab import files
df
train_set_idx
df.to_csv('df.csv', encoding = 'utf-8-sig') 
files.download('df.csv')

def test_number(list_of_lists):
  non_numbers = []
  for i in range(len(list_of_lists)):
    for l in range(4):
      if list_of_lists[i][l].isnumeric() == False:
        non_numbers.append(i)
      else:
        pass
  return non_numbers

def bonds(m, n):
  mh = AddHyfromSmile(m)
  X = Chem.MolToMolBlock(mh)
  X = X.split("\n")
  filtered_bond_lines = [
    list(filter(lambda x: x != '', line.split(" ")))
    for line in X
  ]
  relevant_rows = filtered_bond_lines[4:]
  big_ls = list(map(lambda x: x[:4], relevant_rows))
  bond_ls_str = [x for x in big_ls[n+1:] if len(x) == 4]
  non_num_ids = test_number(bond_ls_str)
  ids = [x for x in range(len(bond_ls_str)) if x not in non_num_ids]
  bond_ls_final_str = [bond_ls_str[i] for i in ids]
  bond_ls = []
  y = []
  for bond in bond_ls_final_str:
    bond_ls.append([int(x) for x in bond[:3]])
  for i in range(n):
    y.append([i, [k[2] for k in bond_ls if i in bond_ls]])
  
  return bond_ls

# find index of carbons
# find index of hydrogens
# find hyrdrogens bonded to carbons, set that value to 0
# others get a 1

def set_atom_index_labels(mol, label, one_based=False):
  for atom in mol.GetAtoms():
    if one_based:
      str_index = str(atom.GetIdx()+1)
    else:
      str_index = str(atom.GetIdx())
    atom.SetProp(label, str_index)
  return mol

from rdkit.Chem import AllChem
from rdkit.Chem import rdqueries
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from oddt import toolkit
from openbabel import pybel, openbabel as ob
atoms = ['B', 'C', "N", "O", "P", "S", "Se"]
halogens = ['F', 'Cl', 'Br', "I", "At", "Ts"]
metals = ['Li', 'Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W',"Re",'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv']

def heavyvalence(m):
  mol = pybel.readstring('smi', m)
  pybel.Atom(mol.atoms[1])
  heavy_val = [mol.atoms[i].heavydegree for i in range(len(mol.atoms))]
  heavy = [float(x) for x in heavy_val]
  return heavy

def heterovalence(m):
  mol = pybel.readstring('smi', m)
  idx = []
  for i in range(len(mol.atoms)):
    if str(mol.atoms[i])[6:8] != '1 ':
      idx.append(i)
  hetero_val = [mol.atoms[i].heterodegree for i in idx]
  return hetero_val


def get_charge(m):
  _, res = rdEHTTools.RunMol(m)
  static_chgs = res.GetAtomicCharges()
  rdd_charge = [round(x,3) for x in static_chgs]
  return rdd_charge

def AddHyfromSmile(m):
  sm = Chem.MolFromSmiles(m)
  mh = Chem.AddHs(sm,addCoords=True)
  return mh

def AtomHyb(m):
  hyb = []
  sm = Chem.MolFromSmiles(m)
  for x in sm.GetAtoms():
      # hybridization
      hyb.append(x.GetHybridization())
  hyb2 = [float(x) for x in hyb]
  return hyb2

def Tot_charge(m):
  Total_chg = round(sum(get_charge(m)),2)
  Tot_chg = [Total_chg]*len(get_charge(m))
  return Tot_chg
  
def atom_count(m):
  for x in m.GetAtoms():
    y = x.GetIdx()
  return y

def get_coords(m, n):
  X = Chem.MolToMolBlock(m)
  X = X.split("\n")
  filtered_bond_lines = [
    list(filter(lambda x: x != '', line.split(" ")))
    for line in X
  ]
  relevant_rows = filtered_bond_lines[4:]
  rows = list(map(lambda x: x[:4], relevant_rows))[:n+1]
  coordsAtom = [[],[],[],[]]
  for row in rows:
    coords = [float(x) for x in row[:3]]
    for i in range(3):
      coordsAtom[i].append(coords[i])
    coordsAtom[3].append(row[3])
  return coordsAtom

def atom_col(m, n):
  coordsAtom = get_coords(m,n)
  atomvalue = [[],[],[],[],[],[],[],[],[]]
  for x in coordsAtom[3]:
    if x in atoms:
      ind = atoms.index(x)
      atomvalue[ind].append(1.0)
      for i in [x for x in range(len(atomvalue)) if x != ind]:
        atomvalue[i].append(0.0)
    elif x in halogens:
      ind = halogens.index(x)
      atomvalue[-2].append(1.0)
      for i in [x for x in range(len(atomvalue)) if x != 7]:
        atomvalue[i].append(0.0)
    elif x in metals:
      ind = metals.index(x)
      atomvalue[-1].append(1.0)
      for i in [x for x in range(len(atomvalue)) if x != 8]:
        atomvalue[i].append(0.0)
    else:
      for i in range(len(atomvalue)):
        atomvalue[i].append(0.0)
  return atomvalue

import itertools

def GetRingSystems(mol, includeSpiro=False):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems

def rings(m, n):
  mol = Chem.MolFromSmiles(m)
  sep_list = [list(x) for x in GetRingSystems(mol)]
  ring_list = list(itertools.chain.from_iterable(sep_list))
  rings = [1.0 if i in ring_list else 0.0 for i in range(n+1)]
  return rings

def aromatic(m, n):
    q = rdqueries.IsAromaticQueryAtom()
    arom = [x.GetIdx() for x in Chem.MolFromSmiles(m).GetAtomsMatchingQuery(q)]
    aromLs = [1.0 if i in arom else 0.0 for i in range(n+1)]
    return aromLs

def acceptor(m,n):
  fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
  factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
  feats = factory.GetFeaturesForMol(Chem.MolFromSmiles(m))
  x = [list(y.GetAtomIds()) for y in feats if y.GetFamily() == 'Acceptor']
  xy = [item for sublist in x for item in sublist]
  acc_ls = [1.0 if i in xy else 0.0 for i in range(n+1)]
  return acc_ls

def donor(m,n):
  fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
  factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
  feats = factory.GetFeaturesForMol(Chem.MolFromSmiles(m))
  x = [list(y.GetAtomIds()) for y in feats if y.GetFamily() == 'Donor']
  xy = [item for sublist in x for item in sublist]
  donor_ls = [1.0 if i in xy else 0.0 for i in range(n+1)]
  return donor_ls

def hyrdrophobe(m,n):
  fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
  factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
  feats = factory.GetFeaturesForMol(Chem.MolFromSmiles(m))
  x = [list(y.GetAtomIds()) for y in feats if y.GetFamily() == 'Hydrophobe']
  xy = [item for sublist in x for item in sublist]
  hydro_ls = [1.0 if i in xy else 0.0 for i in range(n+1)]
  return hydro_ls

def hyd_carb(m,n,df2):
  index_carbon = df2.index[df2['Atom']=='C'].tolist()
  index_hydr = df2.index[df2['Atom']=='H'].tolist()
  bonds1 = bonds(m, n)
  ls_one = [x[0] for x in bonds1]
  ls_two = [x[1] for x in bonds1]
  hyd_bond = []
  for idx in index_hydr:
    if idx in ls_one:
      if ls_two[ls_one.index(idx)] in index_carbon:
        hyd_bond.append(idx)
      else:
        pass
    else:
      if ls_one[ls_two.index(idx)] in index_carbon:
        hyd_bond.append(idx)
      else:
        pass
  hydr_carb = []
  for idx in index_hydr:
    if idx in hyd_bond:
      hydr_carb.append(0.0)
    else:
      hydr_carb.append(1.0)
  return hydr_carb

def hetero_total(m,n,df2):
  hydr_carb = hyd_carb(m,n,df2)
  hetero_val = heterovalence(m)
  hetero = [float(x) for x in hetero_val] + hydr_carb
  return hetero

def find_intramolecular_hbonds(mol,confId=-1,eligibleAtoms=[7,8],distTol=2.5):
    '''
    eligibleAtoms is the list of atomic numbers of eligible H-bond donors or acceptors
    distTol is the maximum accepted distance for an H bond
    '''
    res = []
    conf = mol.GetConformer(confId)
    for i in range(mol.GetNumAtoms()):
        atomi = mol.GetAtomWithIdx(i)
        if atomi.GetAtomicNum()==1:
            if atomi.GetDegree() != 1:
                continue
            nbr = atomi.GetNeighbors()[0]
            if nbr.GetAtomicNum() not in eligibleAtoms:
                continue
            # loop over all other atoms except ones we're bound to and other Hs:
            for j in range(mol.GetNumAtoms()):
                if j==i:
                    continue
                atomj = mol.GetAtomWithIdx(j)
                if atomj.GetAtomicNum() not in eligibleAtoms or mol.GetBondBetweenAtoms(i,j):
                    continue
                dist = (conf.GetAtomPosition(i)- conf.GetAtomPosition(j)).Length()
                if dist<distTol:
                    res.append((i,j,dist))
    return res

def intramolecular_hydr(mh,n):
  atoms = []
  x = find_intramolecular_hbonds(mh) 
  if len(x) != 0:
    for value in x:
      atoms.append(value[0])
      atoms.append(value[1])
  atom_list = []
  for i in range(n+1):
    r = atoms.count(i+1)
    atom_list.append(float(r))
  return atom_list

def dataframes(index_list):
  frames = []
  for i in index_list:
    m = df.SMILES[i]
    mh = AddHyfromSmile(m)
    n = atom_count(mh)
    pka = [round(float(df.JCHEM_PKA.to_list()[i]), 6)]*(n+1)
    r = set_atom_index_labels(Chem.MolFromSmiles(m), 'molAtomMapNumber', one_based=True)
    A = AllChem.EmbedMolecule(mh,randomSeed=0xf00d, maxAttempts = 5000)
    if A == 0:
      intra_hyd = intramolecular_hydr(mh,n)
      hyb = [float(x) for x in AtomHyb(m)] + [0.0]*(n-len(AtomHyb(m))+ 1)
      coordsAtom = get_coords(mh, n)
      atom_col1 = atom_col(mh, n)
      aromatic1 = aromatic(m,n)
      acceptor1 = acceptor(m,n)
      donor1 = donor(m,n)
      hydro = hyrdrophobe(m,n)
      heavy_val = heavyvalence(m) + [1]*(n+1 - len(heavyvalence(m)))
      file_name = ["B" + str(i)]*(n+1)
      new_idx = [index_list.index(i)] * (n+1)
      if index_list.index(i) % 100 == 0:
          print(index_list.index(i))
      df2 = pd.DataFrame({'Atom': coordsAtom[3], 'xcoord':coordsAtom[0], 'ycoord':coordsAtom[1], 'zcoord':coordsAtom[2],'B': atom_col1[0], 'C': atom_col1[1], 'N': atom_col1[2], 'O': atom_col1[3], 'P': atom_col1[4], 'S': atom_col1[5], 'Se': atom_col1[6], 'Halogen':atom_col1[7], 'Metal': atom_col1[8], 'hyb': hyb, 'charge': get_charge(mh), 'Tot_chg':Tot_charge(mh), 'rings':rings(m,n), 'aromatic':aromatic1, 'acceptor': acceptor1, 'donor': donor1, 'hydro': hydro, 'heavy': heavy_val})
      df_final = pd.DataFrame({'idx': new_idx, 'pka': pka, 'file_name': file_name, 'x': coordsAtom[0], 'y': coordsAtom[1], 'z': coordsAtom[2],'B': atom_col1[0], 'C': atom_col1[1], 'N': atom_col1[2], 'O': atom_col1[3], 'P': atom_col1[4], 'S': atom_col1[5], 'Se': atom_col1[6], 'halogen':atom_col1[7], 'metal': atom_col1[8], 'hyb': hyb, 'heavyvalence': heavy_val, 'heterovalence': hetero_total(m,n,df2), 'partialcharge':get_charge(mh), 'is_center_residue': Tot_charge(mh), 'res_type': intra_hyd, 'hydrophobic': hydro, 'aromatic':aromatic1, 'acceptor': acceptor1, 'donor': donor1, 'ring':rings(m,n)})
      frames.append(df_final)
  return pd.concat(frames)

train_df = dataframes(train_set_idx)

def set_atom_index_labels(mol, label, one_based=False):
  """Draws the molecule and labels the atoms with their index.
  
See https://stackoverflow.com/questions/53321453/rdkit-how-to-show-moleculars-atoms-number?answertab=active#tab-top

'mol':
    an RDKit molecule.
'label':
    'atomLabel' changes each element name to its numeric index.
    'molAtomMapNumber' labels each atom with a colon followed by the index.
    'atomNote' puts the index next to the element name.
'one_based', optional: 
    False by default, so atoms will be indexed from zero.
  
  """
  for atom in mol.GetAtoms():
    if one_based:
      str_index = str(atom.GetIdx()+1)
    else:
      str_index = str(atom.GetIdx())
    atom.SetProp(label, str_index)
  return mol
set_atom_index_labels(Chem.MolFromSmiles("[H]\C1=C2/[C@@]([H])(COC)CC[C@@]2([H])[C@@]([H])(C)[C@@]([H])(O)[C@]([H])(O[C@@]2([H])O[C@]([H])(COC(C)(C)C=C)[C@@]([H])(O)[C@]([H])(OC(C)=O)[C@@]2([H])O)C2=C(C[C@]([H])(O)[C@]12C)[C@]([H])(C)COC(C)=O"), 'atomLabel', one_based=True)

from google.colab import files
df.to_csv('output.csv', encoding = 'utf-8-sig') 

#test_df.to_csv('test_csv.csv',index=False, encoding = 'utf-8-sig')
train_df.to_csv('train_csv_intrahyd.csv',index=False, encoding = 'utf-8-sig')
#files.download('test_csv.csv')
files.download('train_csv_intrahyd.csv')