from lib_functions.libraries import *
from typing import Optional
from rdkit import Chem
from lib_functions.formula_utils import element_max_valence

import pandas as pd

import networkx as nx
from collections import Counter

from tqdm import tqdm
import os

def atom_num_radical_electrons(atom: Chem.Atom) -> int:
    """Returns RDKit's num radical electrons for an atom."""
    try:
        return int(atom.GetNumRadicalElectrons())
    except Exception:
        return 0

def atom_observed_valence(atom: Chem.Atom, include_hydrogens: bool = False, aromatic_as_double: bool = False) -> int:
    """Returns observed valence for an RDKit atom.
    Counts bond orders to heavy neighbors (single=1, double=2, triple=3; aromatic as 1).
    If include_hydrogens=True, adds the atom's implicit hydrogens (GetTotalNumHs()).
    """
    val = 0
    for b in atom.GetBonds():
        bt = b.GetBondType()
        if bt == Chem.BondType.SINGLE:
            val += 1
        elif bt == Chem.BondType.DOUBLE:
            val += 2
        elif bt == Chem.BondType.TRIPLE:
            val += 3
        elif bt == Chem.BondType.AROMATIC:
            val += 2 if aromatic_as_double else 1
        else:
            # Treat aromatic as single for counting purposes
            val += 1
    if include_hydrogens:
        val += atom.GetTotalNumHs()
    return val


def build_valence_distribution_from_smiles_csv(
    csv_path: str,
    smiles_col: str = 'smiles',
    chunksize: int = 10000,
    max_molecules: int = 100000,
    use_progress: bool = True,
    include_hydrogens: bool = False,
    aromatic_as_double: bool = False,
    kekulize: bool = True,
    count_implicit_hydrogen_atoms: bool = False,
) -> dict[tuple[str, int], Counter]:
    """Scans a .smiles list (one per line) or a CSV of SMILES and returns
    (symbol, formal_charge) -> Counter(valence->count).
    If include_hydrogens=True, valence counts include implicit Hs.
    """
    distribution: dict[tuple[str, int], Counter] = {}
    molecules_processed = 0

    # If a plain .smiles/.smi file is provided, iterate lines directly
    ext = os.path.splitext(csv_path)[1].lower()
    if ext in ['.smiles', '.smi']:
        for smi in iterate_smiles(
            csv_path,
            smiles_col=smiles_col,
            chunksize=chunksize,
            max_molecules=max_molecules,
            use_progress=use_progress,
        ):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Exception:
                    continue
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                ch = atom.GetFormalCharge()
                val = atom_observed_valence(
                    atom,
                    include_hydrogens=include_hydrogens,
                    aromatic_as_double=aromatic_as_double,
                )
                key = (sym, ch)
                if key not in distribution:
                    distribution[key] = Counter()
                distribution[key][val] += 1
                if count_implicit_hydrogen_atoms:
                    num_h = atom.GetTotalNumHs()
                    if num_h > 0:
                        h_key = ('H', 0)
                        if h_key not in distribution:
                            distribution[h_key] = Counter()
                        distribution[h_key][1] += int(num_h)
        return distribution

    # Compute total rows for CSV progress bar (optional)
    try:
        total_rows = min(sum(1 for _ in open(csv_path, 'r')) - 1, max_molecules)
        total_chunks = (total_rows + chunksize - 1) // chunksize
    except Exception:
        total_rows = None
        total_chunks = None

    iterator = pd.read_csv(csv_path, chunksize=chunksize, nrows=max_molecules)
    if use_progress and total_chunks is not None:
        iterator = tqdm(iterator, total=total_chunks)

    for chunk in iterator:
        if molecules_processed >= max_molecules:
            break
        remaining = max_molecules - molecules_processed
        if len(chunk) > remaining:
            chunk = chunk.head(remaining)
        for smi in chunk[smiles_col].dropna().astype(str):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Exception:
                    # If kekulization fails, skip this molecule to avoid mixing definitions
                    continue
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                ch = atom.GetFormalCharge()
                val = atom_observed_valence(atom, include_hydrogens=include_hydrogens, aromatic_as_double=aromatic_as_double)
                key = (sym, ch)
                if key not in distribution:
                    distribution[key] = Counter()
                distribution[key][val] += 1
                if count_implicit_hydrogen_atoms:
                    num_h = atom.GetTotalNumHs()
                    if num_h > 0:
                        h_key = ('H', 0)
                        if h_key not in distribution:
                            distribution[h_key] = Counter()
                        distribution[h_key][1] += int(num_h)
        molecules_processed += len(chunk)
    return distribution


def build_valid_valence_table_from_smiles_csv(csv_path: str, smiles_col: str = 'smiles') -> dict[tuple[str, int], set[int]]:
    """Compatibility helper returning sets of observed valences for each (symbol,charge)."""
    dist = build_valence_distribution_from_smiles_csv(csv_path, smiles_col=smiles_col)
    return {k: set(c.keys()) for k, c in dist.items()}


def build_charge_distribution_from_smiles_csv(
    csv_path: str,
    smiles_col: str = 'smiles',
    chunksize: int = 10000,
    max_molecules: int = 100000,
    use_progress: bool = True,
    kekulize: bool = False,
) -> dict[str, Counter]:
    """Scans a .smiles list (one per line) or a CSV of SMILES and returns a
    distribution of which elements carry +/- charge.

    Returns a dict with keys '+' and '-' mapping to Counter(symbol->count), counting
    one per atom that has positive (>0) or negative (<0) formal charge. The magnitude
    of charge is ignored (i.e., an atom with +2 contributes one to '+').
    """
    distribution: dict[str, Counter] = {
        '+': Counter(),
        '-': Counter(),
    }

    molecules_processed = 0
    ext = os.path.splitext(csv_path)[1].lower()
    if ext in ['.smiles', '.smi']:
        for smi in iterate_smiles(
            csv_path,
            smiles_col=smiles_col,
            chunksize=chunksize,
            max_molecules=max_molecules,
            use_progress=use_progress,
        ):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Exception:
                    continue
            for atom in mol.GetAtoms():
                ch = atom.GetFormalCharge()
                if ch == 0:
                    continue
                sym = atom.GetSymbol()
                if ch > 0:
                    distribution['+'][sym] += 1
                elif ch < 0:
                    distribution['-'][sym] += 1
        return distribution

    try:
        total_rows = min(sum(1 for _ in open(csv_path, 'r')) - 1, max_molecules)
        total_chunks = (total_rows + chunksize - 1) // chunksize
    except Exception:
        total_rows = None
        total_chunks = None

    iterator = pd.read_csv(csv_path, chunksize=chunksize, nrows=max_molecules)
    if use_progress and total_chunks is not None:
        iterator = tqdm(iterator, total=total_chunks)

    for chunk in iterator:
        if molecules_processed >= max_molecules:
            break
        remaining = max_molecules - molecules_processed
        if len(chunk) > remaining:
            chunk = chunk.head(remaining)
        for smi in chunk[smiles_col].dropna().astype(str):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Exception:
                    continue
            for atom in mol.GetAtoms():
                ch = atom.GetFormalCharge()
                if ch == 0:
                    continue
                sym = atom.GetSymbol()
                if ch > 0:
                    distribution['+'][sym] += 1
                elif ch < 0:
                    distribution['-'][sym] += 1
        molecules_processed += len(chunk)

    return distribution


def build_radical_distribution_from_smiles_csv(
    csv_path: str,
    smiles_col: str = 'smiles',
    chunksize: int = 10000,
    max_molecules: int = 100000,
    use_progress: bool = True,
    kekulize: bool = False,
) -> dict[tuple[str, int], Counter]:
    """Escanea un fichero .smiles (uno por línea) o un CSV de SMILES y devuelve
    distribución de radicales por (símbolo, carga).

    Retorna un dict (symbol, formal_charge) -> Counter(radical_electrons -> count).
    radical_electrons es típicamente 0, 1, 2 ... (nos interesa >0).
    """
    distribution: dict[tuple[str, int], Counter] = {}
    molecules_processed = 0
    ext = os.path.splitext(csv_path)[1].lower()
    if ext in ['.smiles', '.smi']:
        for smi in iterate_smiles(
            csv_path,
            smiles_col=smiles_col,
            chunksize=chunksize,
            max_molecules=max_molecules,
            use_progress=use_progress,
        ):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Exception:
                    continue
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                ch = atom.GetFormalCharge()
                rad = atom_num_radical_electrons(atom)
                key = (sym, ch)
                if key not in distribution:
                    distribution[key] = Counter()
                distribution[key][rad] += 1
        return distribution

    try:
        total_rows = min(sum(1 for _ in open(csv_path, 'r')) - 1, max_molecules)
        total_chunks = (total_rows + chunksize - 1) // chunksize
    except Exception:
        total_rows = None
        total_chunks = None

    iterator = pd.read_csv(csv_path, chunksize=chunksize, nrows=max_molecules)
    if use_progress and total_chunks is not None:
        iterator = tqdm(iterator, total=total_chunks)

    for chunk in iterator:
        if molecules_processed >= max_molecules:
            break
        remaining = max_molecules - molecules_processed
        if len(chunk) > remaining:
            chunk = chunk.head(remaining)
        for smi in chunk[smiles_col].dropna().astype(str):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Exception:
                    continue
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                ch = atom.GetFormalCharge()
                rad = atom_num_radical_electrons(atom)
                key = (sym, ch)
                if key not in distribution:
                    distribution[key] = Counter()
                distribution[key][rad] += 1
        molecules_processed += len(chunk)

    return distribution


def validate_graph_valences(G: nx.Graph, valid_valences: dict[tuple[str, int], set[int]]) -> bool:
    """Checks if every node's observed valence is in the valid set for (symbol,charge)."""
    for n, d in G.nodes(data=True):
        sym = d.get('label')
        ch = d.get('formal_charge', 0)
        # Count multiplicity from MultiGraph
        val = 0
        for nbr in G.neighbors(n):
            val += G.number_of_edges(n, nbr)
        allowed = valid_valences.get((sym, ch))
        if not allowed or val not in allowed:
            return False
    return True


def graph_valence_profile(G: nx.Graph) -> Counter:
    """Returns a Counter of (symbol,charge,valence)->count for the graph."""
    profile = Counter()
    for n, d in G.nodes(data=True):
        sym = d.get('label')
        ch = d.get('formal_charge', 0)
        val = 0
        for nbr in G.neighbors(n):
            val += G.number_of_edges(n, nbr)
        profile[(sym, ch, val)] += 1
    return profile

def current_valence_in_graph(G: nx.Graph, node: str) -> int:
    """Sum multiplicities (number of parallel edges) to all neighbors."""
    val = 0
    for nbr in G.neighbors(node):
        val += G.number_of_edges(node, nbr)
    return val


def sample_target_valence(sym: str, ch: int, weights: dict[int, float], rng: Optional[np.random.Generator] = None) -> int:
    """Sample a target valence using provided weights (val->prob)."""
    if rng is None:
        rng = np.random.default_rng()
    vals = list(weights.keys())
    probs = np.array([weights[v] for v in vals], dtype=float)
    s = probs.sum()
    if s <= 0:
        return max(vals)
    probs = probs / s
    return int(rng.choice(vals, p=probs))


def build_target_valence_map(G: nx.Graph, valence_weights: dict[tuple[str, int], dict[int, float]], rng: Optional[np.random.Generator] = None) -> dict[str, int]:
    """For each node, sample a target valence from dataset weights; fallback to max allowed.

    Returns node->target_valence (including hydrogens counted as edges).
    Hydrogens keep target 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    targets: dict[str, int] = {}
    for n, d in G.nodes(data=True):
        sym = d.get('label')
        ch = d.get('formal_charge', 0)
        if sym == 'H':
            targets[n] = 1
            continue
        weights = valence_weights.get((sym, ch))
        if weights:
            targets[n] = sample_target_valence(sym, ch, weights, rng)
        else:
            # Fallback to max permissible valence
            targets[n] = element_max_valence(sym, ch)
    return targets


def improve_graph_valences(
    G: nx.MultiGraph,
    target_valence_map: dict[str, int],
    max_multiplicity: int = 3,
    max_iters: int = 10000,
) -> None:
    """Greedily add heavy-atom bonds to move node valences toward targets.

    - Only adds bonds between non-hydrogen nodes.
    - Respects per-pair maximum multiplicity (triples).
    - Never exceeds element_max_valence for any endpoint.
    """
    rng = np.random.default_rng()
    heavy_nodes = [n for n, d in G.nodes(data=True) if d.get('label') != 'H']

    def allowed_to_add(u: str, v: str) -> bool:
        if G.number_of_edges(u, v) >= max_multiplicity:
            return False
        du = G.nodes[u]
        dv = G.nodes[v]
        cu = current_valence_in_graph(G, u)
        cv = current_valence_in_graph(G, v)
        maxu = element_max_valence(du.get('label'), du.get('formal_charge', 0))
        maxv = element_max_valence(dv.get('label'), dv.get('formal_charge', 0))
        if cu >= maxu or cv >= maxv:
            return False
        if cu >= target_valence_map.get(u, maxu) and cv >= target_valence_map.get(v, maxv):
            return False
        return True

    iters = 0
    while iters < max_iters:
        iters += 1
        # Compute deficits for heavy nodes
        deficits = []
        for n in heavy_nodes:
            cur = current_valence_in_graph(G, n)
            dnode = target_valence_map.get(n, cur) - cur
            # Also cap by max element valence
            maxn = element_max_valence(G.nodes[n].get('label'), G.nodes[n].get('formal_charge', 0))
            if target_valence_map.get(n, cur) > maxn:
                # clamp target
                target_valence_map[n] = maxn
                dnode = maxn - cur
            if dnode > 0:
                deficits.append((n, dnode))
        if len(deficits) < 2:
            break
        # Choose two with largest deficits
        deficits.sort(key=lambda x: -x[1])
        u = deficits[0][0]
        # Pick a partner among top-k
        k = min(8, len(deficits) - 1)
        candidates = [n for n, _ in deficits[1:1+k] if allowed_to_add(u, n)]
        if not candidates:
            # Try random partner
            random_candidates = [n for n, _ in deficits[1:] if allowed_to_add(u, n)]
            if not random_candidates:
                # No allowable additions
                break
            v = rng.choice(random_candidates)
        else:
            v = rng.choice(candidates)
        # Add a bond
        G.add_edge(u, v)



def iterate_smiles(
    smiles_path: str,
    smiles_col: str = 'smiles',
    chunksize: int = 10000,
    max_molecules: int = 10**12,
    use_progress: bool = False,
):
    """Yield SMILES strings from a .smiles/.smi file, one per non-empty line.

    - Ignores blank lines and lines starting with '#'.
    - If lines contain multiple whitespace-separated columns, takes the first token as SMILES.
    - Stops after yielding max_molecules entries.
    """
    del smiles_col, chunksize  # Unused for plain text .smiles files
    yielded = 0
    try:
        total_lines = None
        if use_progress:
            try:
                with open(smiles_path, 'r') as fcount:
                    total_lines = sum(1 for _ in fcount)
            except Exception:
                total_lines = None
        with open(smiles_path, 'r') as f:
            iterator = f
            if use_progress:
                iterator = tqdm(iterator, total=total_lines)
            for raw_line in iterator:
                if yielded >= max_molecules:
                    break
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                smi = line.split()[0]
                if smi:
                    yield smi
                    yielded += 1
    except Exception:
        return

