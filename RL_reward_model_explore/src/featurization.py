"""SMILES -> PyG Data featurization (CoCoGraph-compatible, self-contained)."""

from __future__ import annotations

import numpy as np
import networkx as nx
import torch
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from .config import CYCLE_MAX, ENCEL, MAX_ATOM, NGFEAT, NNFEAT
from .graph_utils import count_bond_types, nx_to_rdkit, pad_adjs, smiles_to_graph


def _all_simple_cycles(graph: nx.Graph):
    dg = nx.Graph(graph).to_directed()
    return nx.simple_cycles(dg)


def _create_cycle_dict(cycle_basis, cycle_max=CYCLE_MAX):
    edge_to_cycle_sizes = {}
    cycles_total = []
    for cycle in cycle_basis:
        cycle_size = len(cycle)
        cycles_total.append(cycle)
        for i in range(len(cycle)):
            edge = (cycle[i], cycle[(i + 1) % len(cycle)])
            edge_to_cycle_sizes.setdefault(edge, []).append(min(cycle_size, cycle_max))
    return edge_to_cycle_sizes, cycles_total


def _find_edge_cycles_with_sizes(g: nx.MultiGraph, cycle_max=CYCLE_MAX):
    cycle_basis = nx.cycle_basis(nx.Graph(g))
    return _create_cycle_dict(cycle_basis, cycle_max=cycle_max)


def embed_graph_nodes_norm(graph: nx.MultiGraph, encel=ENCEL, cycle_max=CYCLE_MAX):
    gsingle = nx.Graph(graph)
    the_cycles = [c for c in nx.cycle_basis(gsingle) if len(c) > 2]

    features = [0 for _ in range(1, cycle_max)]
    for cycle in the_cycles:
        try:
            features[len(cycle) - 2] += 1
        except IndexError:
            features[0] += 1

    total_cycles = sum(features)
    if total_cycles > 0:
        features = [count / total_cycles for count in features]

    features.append(int(nx.is_planar(graph)))
    features.append(1 if nx.number_connected_components(graph) == 1 else 0)
    features.append(len(list(nx.bridges(graph))) / graph.number_of_edges())
    features.append(len(list(nx.bridges(gsingle))) / gsingle.number_of_edges())

    multiedges = [0 for _ in range(4)]
    norm = 2 * graph.number_of_edges()
    for n1 in graph.nodes():
        for n2 in graph.nodes():
            num = min(graph.number_of_edges(n1, n2), 4)
            if num > 0:
                multiedges[num - 1] += num / norm
    features += multiedges
    gfeatures = torch.tensor(features, dtype=torch.float32)

    bridge_count = {node: 0 for node in graph.nodes()}
    for u, v in nx.bridges(graph):
        bridge_count[u] += 1
        bridge_count[v] += 1

    node_embeddings = []
    node_list = []
    for node in graph.nodes():
        embedding = [0] * len(encel)
        element = node.split("_")[0]
        if element in encel:
            embedding[encel.index(element)] = 1

        in_cycle_size = [0] * (cycle_max - 1)
        for cycle in the_cycles:
            if node in cycle:
                cycsize = len(cycle)
                if 2 < cycsize <= cycle_max:
                    in_cycle_size[cycsize - 3] += 1
                else:
                    in_cycle_size[cycle_max - 2] += 1

        total_node_cycles = sum(in_cycle_size)
        if total_node_cycles > 0:
            in_cycle_size = [count / total_node_cycles for count in in_cycle_size]
        embedding += in_cycle_size

        neighbors = [0, 0]
        vecinos = len(list(gsingle.neighbors(node))) or 1
        for ne in gsingle.neighbors(node):
            if ne.startswith("H_"):
                neighbors[1] += 1 / vecinos
            else:
                neighbors[0] += 1 / vecinos
        embedding += neighbors
        embedding.append(bridge_count[node] / vecinos)
        node_embeddings.append(embedding)
        node_list.append(node)

    node_embeddings = np.array(node_embeddings)
    node_embeddings = np.concatenate(
        [node_embeddings, np.zeros([MAX_ATOM - node_embeddings.shape[0], node_embeddings.shape[1]])],
        axis=0,
    )
    nfeatures = torch.tensor(node_embeddings, dtype=torch.float32)

    all_pairs = dict(nx.shortest_path_length(gsingle))
    one_hot_distances = np.zeros((MAX_ATOM, MAX_ATOM, 12), dtype=np.float32)
    for i, node1 in enumerate(node_list):
        for j, node2 in enumerate(node_list):
            if node1 in all_pairs and node2 in all_pairs[node1]:
                distance = all_pairs[node1][node2]
                if distance > 10:
                    one_hot_distances[i, j, 10] = 1
                else:
                    one_hot_distances[i, j, distance - 1] = 1
            else:
                one_hot_distances[i, j, 11] = 1

    return gfeatures, nfeatures, one_hot_distances


def embed_edges_with_cycle_sizes_norm(g: nx.MultiGraph, cycle_max=CYCLE_MAX):
    nodes = list(g.nodes())
    edges, edge_attributes = [], []
    edge_to_cycle_sizes, _ = _find_edge_cycles_with_sizes(g, cycle_max=cycle_max)
    bridges = set(nx.bridges(g))

    for n1, n2 in g.edges():
        nn1, nn2 = nodes.index(n1), nodes.index(n2)
        if (nn1, nn2) in edges or (nn2, nn1) in edges:
            continue

        bond_type = g.number_of_edges(n1, n2) - 1
        cycle_sizes = edge_to_cycle_sizes.get((n1, n2), []) + edge_to_cycle_sizes.get((n2, n1), [])
        cycle_size_attributes = [
            cycle_sizes.count(i) / 2 if i in cycle_sizes else 0 for i in range(3, cycle_max + 1)
        ]
        total_cycles = sum(cycle_size_attributes)
        if total_cycles > 0:
            cycle_size_attributes = [count / total_cycles for count in cycle_size_attributes]

        is_bridge = 1 if (n1, n2) in bridges or (n2, n1) in bridges else 0
        num_paths = len(list(nx.all_simple_paths(nx.Graph(g), n1, n2, cutoff=10)))
        num_paths = np.log1p(num_paths) / np.log1p(150)

        edge_attr = [1 if i == bond_type else 0 for i in range(3)] + cycle_size_attributes + [is_bridge, num_paths]
        edges.append((nn1, nn2))
        edge_attributes.append(edge_attr)
        edges.append((nn2, nn1))
        edge_attributes.append(edge_attr)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)
    return edge_index, edge_attr


def calculate_2d_distances_ordered(graph: nx.MultiGraph, original_node_order):
    mol = nx_to_rdkit(graph, hidrogenos=True)
    AllChem.Compute2DCoords(mol)
    node_to_atom_index = {node: i for i, node in enumerate(original_node_order)}

    distances = np.zeros((MAX_ATOM, MAX_ATOM), dtype=np.float32)
    conf = mol.GetConformer()
    for i, node_i in enumerate(original_node_order):
        pos_i = conf.GetAtomPosition(node_to_atom_index[node_i])
        for j, node_j in enumerate(original_node_order):
            pos_j = conf.GetAtomPosition(node_to_atom_index[node_j])
            distances[i, j] = ((pos_i.x - pos_j.x) ** 2 + (pos_i.y - pos_j.y) ** 2) ** 0.5

    std_dist = np.std(distances)
    if std_dist != 0:
        distances = (distances - np.mean(distances)) / std_dist
    return distances


def smiles_to_pyg_data(smiles: str, min_atoms: int = 5, max_atoms: int = MAX_ATOM) -> Data | None:
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    if graph.number_of_nodes() < min_atoms or graph.number_of_nodes() > max_atoms:
        return None
    if nx.number_connected_components(graph) != 1:
        return None

    node_list = list(graph.nodes())
    _, natoms = count_bond_types(pad_adjs(nx.to_numpy_array(graph, nodelist=node_list), MAX_ATOM))
    gemb, nemb, distances = embed_graph_nodes_norm(graph)
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(graph)
    dosd = calculate_2d_distances_ordered(graph, node_list)

    assert gemb.numel() == NGFEAT
    assert nemb.shape == (MAX_ATOM, NNFEAT)

    return Data(
        x=nemb,
        edge_index=edge_index,
        edge_attr=edge_attr,
        xA=gemb,
        distances=torch.tensor(distances, dtype=torch.float32),
        dosd_distances=torch.tensor(dosd, dtype=torch.float32),
        num_atoms=torch.tensor([natoms], dtype=torch.long),
        smiles=smiles,
    )
