"""Small graph helpers used by featurization."""

from __future__ import annotations

import numpy as np
import networkx as nx
from rdkit import Chem


def pad_adjs(ori_adj: np.ndarray, node_number: int) -> np.ndarray:
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f"ori_len {ori_len} > node_number {node_number}")
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a


def count_bond_types(adj: np.ndarray):
    no_bonds = np.count_nonzero(adj == 0)
    single_bonds = np.count_nonzero(adj == 1)
    double_bonds = np.count_nonzero(adj == 2)
    triple_bonds = np.count_nonzero(adj == 3)
    return [no_bonds - adj.shape[0], single_bonds, double_bonds, triple_bonds], adj.shape[0]


def nx_to_rdkit(graph: nx.MultiGraph, hidrogenos: bool = False):
    m = Chem.MolFromSmiles("")
    mw = Chem.RWMol(m)
    atom_index = {}
    for n, d in graph.nodes(data=True):
        atomo = Chem.Atom(d["label"])
        atomo.SetFormalCharge(d["formal_charge"])
        atom_index[n] = mw.AddAtom(atomo)
    for a, b, _ in graph.edges(data=True):
        start = atom_index[a]
        end = atom_index[b]
        try:
            mw.AddBond(start, end, Chem.BondType.SINGLE)
        except RuntimeError:
            bond = mw.GetBondBetweenAtoms(start, end)
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondType(Chem.BondType.DOUBLE)
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetBondType(Chem.BondType.TRIPLE)
            else:
                raise ValueError("Quadruple bonds not supported")
    mol = mw.GetMol()
    try:
        if not hidrogenos:
            return Chem.RemoveHs(mol)
        return mol
    except Exception:
        return mol


def mol_to_graph(mol: Chem.Mol):
    Chem.rdmolops.Kekulize(mol)
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return None

    g = nx.MultiGraph(directed=False)
    atom_count = {"H": 0}
    mol2graph = {}

    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        atom_count.setdefault(atom_symbol, 0)
        atom_count[atom_symbol] += 1
        glabel = f"{atom_symbol}_{atom_count[atom_symbol]}"
        mol2graph[atom.GetIdx()] = glabel
        g.add_node(
            glabel,
            label=atom_symbol,
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
        )
        for _ in range(atom.GetNumImplicitHs()):
            atom_count["H"] += 1
            glabelh = f"H_{atom_count['H']}"
            g.add_node(glabelh, label="H", atomic_num=1, formal_charge=atom.GetFormalCharge())
            g.add_edge(glabel, glabelh)

    for bond in mol.GetBonds():
        bond_type = str(bond.GetBondType())
        if bond_type == "SINGLE":
            nbond = 1
        elif bond_type == "DOUBLE":
            nbond = 2
        elif bond_type == "TRIPLE":
            nbond = 3
        else:
            raise ValueError(f"Unsupported bond type: {bond_type}")
        for _ in range(nbond):
            g.add_edge(mol2graph[bond.GetBeginAtomIdx()], mol2graph[bond.GetEndAtomIdx()])
    return g


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_graph(mol)
