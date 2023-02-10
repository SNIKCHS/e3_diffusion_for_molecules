#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from rdkit import Chem
from rdkit.Chem import rdmolops


# bond mapping
bond_dict = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, "AROMATIC": 4}
number_to_bond= {1: Chem.rdchem.BondType.SINGLE, 2:Chem.rdchem.BondType.DOUBLE,
                 3: Chem.rdchem.BondType.TRIPLE, 4:Chem.rdchem.BondType.AROMATIC}

def dataset_info():
    return { 'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
            'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)','S4(0)', 'S6(0)'],
            'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5:1, 6:2, 7:3, 8:4, 9:1, 10:2, 11:2, 12:4, 13:6, 14:3},
            'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5:'I', 6:'N', 7:'N', 8:'N', 9:'O', 10:'O', 11:'S', 12:'S', 13:'S'},
            }

def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z
def onehot_to_idx(onehot):
    labels = torch.argmax(onehot, dim=-1)
    return labels

def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False

def to_graph(smiles, dataset):
    Chem.MolToSmiles()
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol is None:
        return [], []
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return [], []
    Chem.RemoveStereochemistry(mol)

    edges = []
    nodes = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        atom_str = "%s%i(%i)" % (symbol, valence, charge)

        if atom_str not in dataset_info(dataset)['atom_types']:
            return [], []
        nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom_str), len(dataset_info(dataset)['atom_types'])))
    return nodes, edges
def normalize_weight(adj_mat, weight):
    degree = [1 / math.sqrt(sum(np.abs(w))) for w in weight]
    for dst in range(len(adj_mat)):
        for src_idx in range(len(adj_mat[dst])):
            src = adj_mat[dst][src_idx]
            weight[dst][src_idx] = degree[dst] * weight[dst][src_idx] * degree[src]

def one_hot_vec(length, pos):
    vec = [0] * length
    vec[pos] = 1
    return vec
def pad_sequence(data_list, maxlen, value=0):
    return [row + [value] * (maxlen - len(row)) for row in data_list]

def graph_to_smile(onehot,adj):
    atom_dict = dataset_info()['number_to_atom']
    molecule = Chem.RWMol()
    atom_index = []
    atoms = onehot_to_idx(onehot)
    for atom_number in atoms:
        if atom_number == 0:
            continue

        atom = Chem.Atom(atom_dict[(atom_number-1).item()])  # 原子索引 atom_number
        molecular_index = molecule.AddAtom(atom)
        atom_index.append(molecular_index)

    # 在原子和原子直接加入指定种类的键
    for index_x, row_vector in enumerate(adj):
        for index_y, bond in enumerate(row_vector):
            if index_y <= index_x:
                continue
            if bond == 0:
                continue
            if bond == 5:
                continue
            else:
                molecule.AddBond(atom_index[index_x], atom_index[index_y], number_to_bond[bond.item()])

    return Chem.MolToSmiles(molecule)

