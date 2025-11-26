from lib_functions.config import *
from lib_functions.libraries import *
from lib_functions.adjacency_utils import connected_double_edge_swap

from collections import Counter
import re


# Default maximum valences for common elements in ENCEL
DEFAULT_MAX_VALENCE = {
    'H': 1,
    'B': 3,
    'C': 4,
    'N': 3,  # neutral typical valence; can be 5 with charges but we keep simple
    'O': 2,
    'F': 1,
    'P': 5,  # allow up to 5 to be permissive
    'S': 6,  # allow up to 6 to be permissive
    'Si': 4, # silicon, typical tetravalent
    'Cl': 1,
    'Br': 1,
    'I': 1,
    'Ca': 2,
    'K': 1,
    'Na': 1,
    'Mg': 2,
}


# Charge-aware overrides for maximum valence (very simplified, chemistry is richer)
# Key: (element_symbol, formal_charge) -> max_valence_for_bonding
CHARGE_MAX_VALENCE: dict[tuple[str, int], int] = {
    ('H', +1): 0,  # proton does not form covalent bonds in this simple model
    ('H', -1): 1,  # hydride can form one bond
    ('C', +1): 3,
    ('C', -1): 3,
    ('N', +1): 4,  # e.g., ammonium
    ('N', -1): 2,
    ('O', +1): 3,  # e.g., oxonium
    ('O', -1): 1,  # e.g., hydroxide
    ('Na', +1): 0, ('K', +1): 0,
    ('Ca', +2): 0, ('Mg', +2): 0,
}


def parse_formula(formula: str, allowed_symbols: list[str] | None = None) -> Counter:
    """Parses a simple chemical formula into a Counter of element -> count.

    If allowed_symbols is provided, tokenization is based on that set (sorted by
    length to prioritize two-letter symbols). Otherwise it falls back to ENCEL.
    Example: 'C2H6O' -> {'C': 2, 'H': 6, 'O': 1}
    """
    # Tokenize with priority for two-letter symbols present in allowed_symbols or ENCEL
    symbols_source = allowed_symbols if allowed_symbols is not None else ENCEL
    encel_sorted = sorted(symbols_source, key=lambda s: -len(s))
    symbol_regex = '|'.join(map(re.escape, encel_sorted))
    pattern = re.compile(rf'({symbol_regex})(\d*)')
    pos = 0
    counts = Counter()
    for match in pattern.finditer(formula):
        sym, num = match.groups()
        if match.start() != pos:
            raise ValueError(f"Unrecognized formula syntax near '{formula[pos:match.start()]}' in {formula}")
        pos = match.end()
        n = int(num) if num else 1
        counts[sym] += n
    if pos != len(formula):
        raise ValueError(f"Unparsed tail in formula: '{formula[pos:]}'")
    return counts


def parse_formula_with_charge(formula: str, allowed_symbols: list[str] | None = None) -> tuple[Counter, int]:
    """Parses a chemical formula and extracts an overall charge if present.

    Supported charge notations (very simple):
    - Trailing global charge: e.g., 'NH4+', 'C2H5O-', 'Ca2+', 'SO4--' (also 'SO4-2')
    - Single-atom ions: 'O-', 'N+' (treated as global charge on the species)

    Returns (element_counts, total_charge).
    """
    # Normalize unicode superscripts if any (not exhaustive)
    formula_str = formula.strip()

    # Robust trailing charge parsing supporting: '+', '-', '+2', '-2', '--', '++', and monatomic 'Ca2+'
    charge = 0
    core = formula_str
    # 1) Digits-then-sign (e.g., 'Ca2+'), only accept if the core is monatomic (to avoid stealing element counts)
    m_ds = re.search(r'(\d+)([+-])$', formula_str)
    if m_ds:
        candidate_core = formula_str[:m_ds.start()].strip()
        try:
            cnts_test = parse_formula(candidate_core, allowed_symbols=allowed_symbols)
            if len(cnts_test) == 1 and sum(cnts_test.values()) == 1:
                magnitude = int(m_ds.group(1))
                sign_char = m_ds.group(2)
                charge = magnitude if sign_char == '+' else -magnitude
                core = candidate_core
            else:
                m_ds = None
        except Exception:
            m_ds = None
    if not m_ds:
        # 2) Repeated signs at end (e.g., '--' or '++')
        m_rep = re.search(r'([+-]+)$', formula_str)
        if m_rep and len(m_rep.group(1)) > 1:
            signs = m_rep.group(1)
            magnitude = len(signs)
            sign_char = signs[-1]
            charge = magnitude if sign_char == '+' else -magnitude
            core = formula_str[:m_rep.start()].strip()
        else:
            # 3) Sign then digits at end (e.g., '-2', '+3')
            m_sd = re.search(r'([+-])(\d+)$', formula_str)
            if m_sd:
                sign_char = m_sd.group(1)
                magnitude = int(m_sd.group(2))
                charge = magnitude if sign_char == '+' else -magnitude
                core = formula_str[:m_sd.start()].strip()
            else:
                # 4) Single trailing sign (e.g., '+', '-')
                m_single = re.search(r'([+-])$', formula_str)
                if m_single:
                    sign_char = m_single.group(1)
                    charge = 1 if sign_char == '+' else -1
                    core = formula_str[:m_single.start()].strip()
                else:
                    core = formula_str

    if not core:
        # Handle cases like just '+' or '-' which are invalid
        raise ValueError(f"Invalid formula: '{formula}'")

    counts = parse_formula(core, allowed_symbols=allowed_symbols)
    return counts, charge


def compute_dbe(counts: Counter) -> int:
    """Computes the double bond equivalents (rings + double bonds) using the formula:
    DBE = C - H/2 - X/2 + N/2 + 1, where X are monovalent halogens (F, Cl, Br, I).
    Elements beyond CHN OX (X halogens) are ignored here.
    """
    c = counts.get('C', 0)
    h = counts.get('H', 0)
    n = counts.get('N', 0)
    x = counts.get('F', 0) + counts.get('Cl', 0) + counts.get('Br', 0) + counts.get('I', 0)
    dbe = c - h / 2 - x / 2 + n / 2 + 1
    return int(round(dbe))


def element_max_valence(sym: str, charge: int = 0) -> int:
    """Returns a permissive max valence possibly adjusted by formal charge."""
    if (sym, charge) in CHARGE_MAX_VALENCE:
        val = CHARGE_MAX_VALENCE[(sym, charge)]
    else:
        val = DEFAULT_MAX_VALENCE.get(sym, 1)
    # Enforce a minimum valence of 1 to avoid zero-valence atoms in generation
    return max(1, val)


def build_gt_from_formula(
    formula: str,
    randomize_swaps: int = 0,
    rng: np.random.Generator | None = None,
    valence_weights: dict[tuple[str, int], dict[int, float]] | None = None,
    charge_symbol_weights: dict[str, dict[str, float]] | None = None,
    max_sampling_retries: int = 20,
    debug_targets: bool = False,
    allow_radicals: bool = False,
    radical_weights: dict[str, float] | None = None,
) -> nx.MultiGraph:
    """Constructs a connected MultiGraph consistent with a molecular formula.

    Steps:
    - Expand heavy atoms and hydrogens
    - Create a connected heavy-atom backbone (random spanning tree)
    - Attach hydrogens weighted by remaining stubs
    - If heavy stubs remain, add multiple bonds up to triple
    - Optionally apply a few DES moves to randomize, preserving connectivity

    Returns a NetworkX MultiGraph with explicit hydrogens and node attributes
    label and formal_charge (set to 0).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Enforce presence of valence_weights: generation always uses provided weights
    if valence_weights is None:
        raise ValueError("valence_weights must be provided and cannot be None")
    if allow_radicals and radical_weights is None:
        raise ValueError("allow_radicals=True requiere radical_weights (p. ej., Data/radical_symbol_weights.json)")

    # Allowed symbols and allowed charges are those present in valence_weights
    allowed_symbols = sorted({sym for (sym, _ch) in valence_weights.keys()}, key=lambda s: -len(s))
    allowed_charges_by_symbol: dict[str, set[int]] = {}
    for (sym_k, ch_k) in valence_weights.keys():
        allowed_charges_by_symbol.setdefault(sym_k, set()).add(int(ch_k))

    counts, total_charge = parse_formula_with_charge(formula, allowed_symbols=allowed_symbols)

    # Helper: max allowed valence from dataset weights (strictly dictionary-driven)
    def max_allowed_valence_from_weights(sym: str, ch: int) -> int:
        w = valence_weights.get((sym, ch))
        if w and len(w) > 0:
            try:
                return int(max(int(k) for k in w.keys()))
            except Exception:
                pass
        # Fallback conservatively to 1 if absent/malformed; upstream logic retries missing combos
        return 1

    # Separate heavy and hydrogens
    num_h = counts.get('H', 0)
    heavy_counts = counts.copy()
    if 'H' in heavy_counts:
        del heavy_counts['H']

    # Basic feasibility: have at least one heavy atom
    if sum(heavy_counts.values()) == 0:
        if num_h == 2:  # H2 special case
            g = nx.MultiGraph()
            g.add_node('H_1', label='H', formal_charge=0)
            g.add_node('H_2', label='H', formal_charge=0)
            g.add_edge('H_1', 'H_2')
            return g
        raise ValueError('Formula must contain at least one heavy atom or be H2')

    # Special cases: single-atom ions like H+, H-, O-, N+
    if sum(heavy_counts.values()) == 0:
        if counts.get('H', 0) == 1 and total_charge in (+1, -1):
            g = nx.MultiGraph()
            ch = total_charge
            g.add_node('H_1', label='H', formal_charge=ch)
            return g
        if counts.get('H', 0) == 2 and total_charge == 0:
            # H2 neutral
            g = nx.MultiGraph()
            g.add_node('H_1', label='H', formal_charge=0)
            g.add_node('H_2', label='H', formal_charge=0)
            g.add_edge('H_1', 'H_2')
            return g
        raise ValueError('Formula must contain at least one heavy atom or be H2')

    # Build node list with labels, charge, and initial stub capacity (max valence)
    nodes = []
    stubs = {}
    charges: dict[str, int] = {}
    # Heavy atoms first
    for sym, cnt in heavy_counts.items():
        for i in range(1, cnt + 1):
            name = f"{sym}_{i}"
            charges[name] = 0
            stubs[name] = max_allowed_valence_from_weights(sym, 0)
            nodes.append((name, { 'label': sym, 'formal_charge': charges[name] }))

    # Then hydrogens
    for i in range(1, num_h + 1):
        name = f"H_{i}"
        charges[name] = 0
        stubs[name] = max_allowed_valence_from_weights('H', 0)
        nodes.append((name, { 'label': 'H', 'formal_charge': charges[name] }))

    # Create empty graph and add nodes
    G = nx.MultiGraph()
    G.add_nodes_from(nodes)

    # Helper: sample heavy with free stubs weighted by remaining
    def sample_heavy_with_stub() -> str:
        heavy = [n for n, d in G.nodes(data=True) if d['label'] != 'H' and stubs[n] > 0]
        if not heavy:
            raise RuntimeError('No heavy atoms with available stubs')
        weights = np.array([stubs[n] for n in heavy], dtype=float)
        weights = weights / weights.sum()
        return rng.choice(heavy, p=weights)

    # Target-based generation (always used when there is at least one heavy atom)
    if len([n for n, d in G.nodes(data=True) if d['label'] != 'H']) > 0:
        heavy_nodes = [n for n, d in G.nodes(data=True) if d['label'] != 'H']
        # Precompute heavy nodes by element symbol
        heavy_nodes_by_symbol: dict[str, list[str]] = {}
        for n in heavy_nodes:
            sym_n = G.nodes[n]['label']
            heavy_nodes_by_symbol.setdefault(sym_n, []).append(n)

        def sample_target(sym: str, ch: int) -> int:
            weights = valence_weights.get((sym, ch))
            # If we do not have weights for the exact (sym,charge), we must retry this attempt
            if not weights:
                raise RuntimeError(f"Missing valence_weights for {(sym, ch)}; retry charges")
            vals = np.array(list(weights.keys()), dtype=int)
            probs = np.array([weights[v] for v in vals], dtype=float)
            s = probs.sum()
            if s <= 0:
                # If all zero, choose uniformly among provided keys
                probs = np.ones_like(vals, dtype=float) / len(vals)
            else:
                probs = probs / s
            chosen = int(rng.choice(vals, p=probs))
            return chosen

        def feasible_targets(targets: dict[str, int]) -> tuple[bool, dict[str, int], dict[str, int]]:
            if debug_targets:
                print("[feasible_targets] Targets:", targets)
            V = sum(targets.values())
            H = num_h
            S_HH = V - H
            n_heavy = len(heavy_nodes)
            if debug_targets:
                print("[feasible_targets] S_HH:", S_HH)
                print("[feasible_targets] n_heavy:", n_heavy)
            if S_HH < 0 or (S_HH % 2) != 0:
                print("Infeasible targets: S_HH < 0 or (S_HH % 2) != 0")
                return False, {}, {}
            if n_heavy > 1 and S_HH < 2 * (n_heavy - 1):
                print("Infeasible targets: n_heavy > 1 and S_HH < 2 * (n_heavy - 1)")
                return False, {}, {}
            # Assign hydrogens H_i using balanced random subject to capacity and D>0 (if n_heavy>1)
            if V == 0 and H > 0:
                print("Infeasible targets: V == 0 and H > 0")
                return False, {}, {}
            # Effective capacity per node to preserve at least one heavy-heavy stub when n_heavy>1
            cap_eff = {}
            for n in heavy_nodes:
                cap = targets[n]
                if n_heavy > 1:
                    cap = max(0, cap - 1)
                cap_eff[n] = cap
            if n_heavy > 1 and sum(cap_eff.values()) < H:
                print("Infeasible targets: insufficient capacity to place H while keeping D>0")
                return False, {}, {}
            base = {n: 0 for n in heavy_nodes}
            rem = H
            # One-by-one random assignment weighted by remaining capacity
            while rem > 0:
                avail = [n for n in heavy_nodes if base[n] < cap_eff[n]]
                if not avail:
                    print("Infeasible targets: ran out of capacity before placing all H")
                    return False, {}, {}
                weights = np.array([cap_eff[n] - base[n] for n in avail], dtype=float)
                s = weights.sum()
                if s <= 0:
                    print("Infeasible targets: zero total capacity")
                    return False, {}, {}
                weights = weights / s
                chosen = rng.choice(avail, p=weights)
                base[chosen] += 1
                rem -= 1
            # Demands heavy-heavy per node
            D = {n: targets[n] - base[n] for n in heavy_nodes}
            # For connectivity (if n_heavy>1) require D[n]>=1 for all
            if n_heavy > 1 and any(D[n] <= 0 for n in heavy_nodes):
                print("Infeasible targets: any(D[n] <= 0 for n in heavy_nodes)")
                return False, {}, {}
            return True, base, D

        # Retry full target-based generation; no legacy fallback
        built = False
        for attempt_idx in range(max_sampling_retries):
            # Reset charges and stubs each attempt; redistribute total charge stochastically
            for n in G.nodes():
                G.nodes[n]['formal_charge'] = 0
                charges[n] = 0
                # Also clear any radicals from previous attempts
                if 'radical_electrons' in G.nodes[n]:
                    try:
                        del G.nodes[n]['radical_electrons']
                    except Exception:
                        G.nodes[n]['radical_electrons'] = 0
            if total_charge != 0:
                # Helper to pick a heavy atom to host a unit of charge using symbol-level weights,
                # but ONLY if the resulting (sym, charge) exists in valence_weights.
                def pick_node_for_sign_constrained(sign: str) -> str | None:
                    sign_delta = -1 if sign == '-' else +1
                    # Build candidate nodes that support the resulting charge according to dataset
                    candidate_nodes: list[str] = []
                    for n in heavy_nodes:
                        sym = G.nodes[n]['label']
                        old_ch = charges.get(n, 0)
                        new_ch = old_ch + sign_delta
                        allowed = allowed_charges_by_symbol.get(sym, set())
                        if new_ch in allowed:
                            candidate_nodes.append(n)
                    if not candidate_nodes:
                        return None
                    # If we have symbol-level weights, translate to node-level by symbol
                    weighted: dict[str, float] = {}
                    if charge_symbol_weights and sign in charge_symbol_weights:
                        weighted = dict(charge_symbol_weights.get(sign, {}))
                    if weighted:
                        weights = []
                        for n in candidate_nodes:
                            sym = G.nodes[n]['label']
                            weights.append(float(weighted.get(sym, 0.0)))
                        arr = np.array(weights, dtype=float)
                        ssum = arr.sum()
                        if ssum > 0:
                            arr = arr / ssum
                            return str(rng.choice(candidate_nodes, p=arr))
                    # Fallback preference ordering by symbol (filtered to candidates)
                    if sign == '+':
                        pref = ['N', 'P', 'S', 'C']
                    else:
                        pref = ['O', 'Cl', 'Br', 'I', 'N', 'S', 'F', 'P', 'C']
                    for sym in pref:
                        pool = [n for n in candidate_nodes if G.nodes[n]['label'] == sym]
                        if pool:
                            return str(rng.choice(pool))
                    # As last resort choose uniformly among candidates
                    return str(rng.choice(candidate_nodes))

                # Distribute charge units; if any step is impossible under constraints, retry attempt
                feasible_charge_assignment = True
                if total_charge < 0:
                    for _step in range(abs(total_charge)):
                        n = pick_node_for_sign_constrained('-')
                        if n is None:
                            feasible_charge_assignment = False
                            break
                        old_ch = charges.get(n, 0)
                        charges[n] = old_ch - 1
                        if debug_targets:
                            lbl = G.nodes[n]['label']
                            print(f"[charge attempt {attempt_idx+1}] -1 -> {n} ({lbl}): {old_ch} -> {charges[n]}")
                else:
                    for _step in range(total_charge):
                        n = pick_node_for_sign_constrained('+')
                        if n is None:
                            feasible_charge_assignment = False
                            break
                        old_ch = charges.get(n, 0)
                        charges[n] = old_ch + 1
                        if debug_targets:
                            lbl = G.nodes[n]['label']
                            print(f"[charge attempt {attempt_idx+1}] +1 -> {n} ({lbl}): {old_ch} -> {charges[n]}")
                if not feasible_charge_assignment:
                    # Retry this attempt with a different charge distribution
                    for u, v in list(G.edges()):
                        G.remove_edge(u, v)
                    continue
                for n, d in G.nodes(data=True):
                    ch = charges.get(n, 0)
                    d['formal_charge'] = ch
                    # Recompute stubs strictly from dataset-driven allowed valences
                    stubs[n] = max_allowed_valence_from_weights(d['label'], ch)
            elif debug_targets:
                print(f"[charge attempt {attempt_idx+1}] neutral")

            try:
                targets = {}
                for n in heavy_nodes:
                    d = G.nodes[n]
                    targets[n] = sample_target(d['label'], d.get('formal_charge', 0))
            except RuntimeError:
                # Missing (sym,charge) in valence_weights: retry attempt (reassign charges)
                for u, v in list(G.edges()):
                    G.remove_edge(u, v)
                continue
            # Optional: reparar paridad S_HH usando exclusivamente radical_weights
            if allow_radicals:
                V_sum = sum(targets.values())
                S_HH_candidate = V_sum - num_h
                if (S_HH_candidate % 2) != 0:
                    adjusted = False
                    # Buscar candidatos ponderados por probabilidad de radical por símbolo y con (t-1) permitido
                    candidates: list[str] = []
                    cand_weights: list[float] = []
                    for n in heavy_nodes:
                        sym = G.nodes[n]['label']
                        chn = G.nodes[n].get('formal_charge', 0)
                        sym_w = float(radical_weights.get(sym, 0.0)) if radical_weights is not None else 0.0
                        if sym_w <= 0.0:
                            continue
                        w = valence_weights.get((sym, chn), {})
                        t = targets[n]
                        if (t - 1) in w and (t - 1) >= 1:
                            candidates.append(n)
                            cand_weights.append(sym_w)
                    if candidates:
                        # Elegir uno ponderado por probabilidad simbólica
                        probs = np.array(cand_weights, dtype=float)
                        s = probs.sum()
                        probs = probs / s if s > 0 else np.ones_like(probs) / len(probs)
                        chosen = rng.choice(candidates, p=probs)
                        targets[chosen] = targets[chosen] - 1
                        # Marcar el átomo elegido como radical (1 electrón desapareado) para visualización
                        try:
                            G.nodes[chosen]['radical_electrons'] = 1
                        except Exception:
                            pass
                        adjusted = True
                        if debug_targets:
                            print(f"[parity_repair] Lowered target of {chosen} guided by radical_weights to fix parity; set radical_electrons=1")
                    # Si no hay candidatos, no forzar ajuste: se reintentará en el siguiente intento
            ok, H_i, D = feasible_targets(targets)
            if not ok:
                continue

            if debug_targets:
                print("[build_gt_from_formula] Sampled target valences (incl. H):")
                print({n: targets[n] for n in heavy_nodes})
                print("[build_gt_from_formula] Planned H assignments per heavy atom:")
                print({n: H_i.get(n, 0) for n in heavy_nodes})
                print("[build_gt_from_formula] Heavy-heavy demand D per heavy atom:")
                print({n: D.get(n, 0) for n in heavy_nodes})

            # 1) Connect heavy atoms to form a tree using D demands
            order = list(heavy_nodes)
            rng.shuffle(order)
            local_D = {k: int(v) for k, v in D.items()}
            local_ok = True
            if len(order) > 1:
                for idx in range(1, len(order)):
                    u = order[idx]
                    prev_candidates = [order[j] for j in range(idx) if local_D[order[j]] > 0]
                    if not prev_candidates:
                        local_ok = False
                        break
                    v = rng.choice(prev_candidates)
                    G.add_edge(u, v)
                    local_D[u] -= 1
                    local_D[v] -= 1
            if not local_ok:
                # Reset graph to nodes only and retry
                for u, v in list(G.edges()):
                    G.remove_edge(u, v)
                continue

            # 2) Attach hydrogens exactly as H_i
            for i in range(1, num_h + 1):
                h = f"H_{i}"
                choices = [n for n in heavy_nodes if H_i.get(n, 0) > 0]
                if not choices:
                    break
                u = rng.choice(choices)
                G.add_edge(u, h)
                H_i[u] -= 1

            # 3) Add remaining heavy-heavy bonds to satisfy local_D
            safety = 0
            def current_multiplicity(u: str, v: str) -> int:
                return G.number_of_edges(u, v)
            while True:
                todo = [n for n in heavy_nodes if local_D.get(n, 0) > 0]
                if len(todo) < 2:
                    break
                u, v = rng.choice(todo, size=2, replace=False)
                if current_multiplicity(u, v) < 3:
                    G.add_edge(u, v)
                    local_D[u] -= 1
                    local_D[v] -= 1
                safety += 1
                if safety > 10000:
                    break

            # Sanity: if some D remains, we failed this attempt; clear and retry
            if any(local_D.get(n, 0) != 0 for n in heavy_nodes):
                for u, v in list(G.edges()):
                    G.remove_edge(u, v)
                continue

            if debug_targets:
                # Compute observed valences including H
                def observed_valence(node: str) -> int:
                    val = 0
                    for nbr in G.neighbors(node):
                        val += G.number_of_edges(node, nbr)
                    return val
                observed = {n: observed_valence(n) for n in heavy_nodes}
                print("[build_gt_from_formula] Observed valences after build:")
                print(observed)
                mism = {n: (targets[n], observed[n]) for n in heavy_nodes if observed[n] != targets[n]}
                if mism:
                    print("[build_gt_from_formula] Mismatches (target, observed):")
                    print(mism)

            # Record how many internal attempts were needed
            try:
                G.graph['tries_used'] = attempt_idx + 1
            except Exception:
                pass
            built = True
            break

        if not built:
            raise RuntimeError(f"Attempt {attempt_idx+1}: Target-based generation failed after retries")

    # Legacy pipeline removed: if target-based build did not succeed, we already raised above

    # Sanity: connectivity of heavy subgraph
    if len(heavy_nodes) > 1:
        if not nx.is_connected(nx.Graph(G).subgraph(heavy_nodes)):
            # As a last resort, connect components among heavy atoms using remaining Hs if any
            comps = list(nx.connected_components(nx.Graph(G).subgraph(heavy_nodes)))
            for i in range(len(comps) - 1):
                u = list(comps[i])[0]
                v = list(comps[i + 1])[0]
                G.add_edge(u, v)

    # 4) Optional randomization via connected DES on simple graph (requires >=4 nodes)
    # Ensure full-graph connectivity (including hydrogens). As a last resort,
    # connect components directly regardless of remaining stubs.
    if not nx.is_connected(nx.Graph(G)):
        comps_full = list(nx.connected_components(nx.Graph(G)))
        for i in range(len(comps_full) - 1):
            u = list(comps_full[i])[0]
            v = list(comps_full[i + 1])[0]
            G.add_edge(u, v)

    if randomize_swaps and len(G) >= 4 and nx.is_connected(G):
        # Use Python's random.Random for connected_double_edge_swap
        rnd = 42
        try:
            connected_double_edge_swap(G, nswap=int(randomize_swaps), seed=rnd)
        except Exception:
            pass

    return G


def formula_to_gt_and_tensor(formula: str, randomize_swaps: int = 0):
    """Helper to build G_T and its padded adjacency tensor like embed_edges_manuel expects.
    Returns (graph, tensor_adj, smiles_str or formula).
    """
    from lib_functions.data_preparation_utils import embed_edges_manuel
    from lib_functions.adjacency_utils import nx_to_rdkit

    G = build_gt_from_formula(formula, randomize_swaps=randomize_swaps)
    tensor_adj, _, _ = embed_edges_manuel(G, list(G.nodes()))
    try:
        mol = nx_to_rdkit(G, False)
        smi = Chem.MolToSmiles(mol)
    except Exception:
        smi = formula
    return G, tensor_adj, smi



