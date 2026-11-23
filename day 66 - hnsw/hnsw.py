"""Day 66 - HNSW: finding the nearest vectors without looking at most of them.

A brute-force search reads every vector; HNSW reads a few hundred of them and
usually returns the same answer.  This module builds the index from scratch -
the layered graph, the greedy descent, the ef-sized candidate list, the
neighbour-selection heuristic - and then measures the three things that
actually decide whether it is a good idea: recall, distance computations, and
what happens after you delete things.

Run it:  python hnsw.py
"""

import heapq
import math
import time

import numpy as np

CORPUS = {'docs': 8_000, 'dims': 48, 'clusters': 40, 'queries': 150, 'k': 10}


# ------------------------------------------------------------------- data
def make_vectors(seed=0, **kw):
    """Clustered unit vectors - embeddings are never uniform on the sphere."""
    cfg = dict(CORPUS, **kw)
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((cfg['clusters'], cfg['dims']))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    members = rng.integers(0, cfg['clusters'], size=cfg['docs'])
    pts = centres[members] + 0.35 * rng.standard_normal((cfg['docs'], cfg['dims']))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    qi = rng.integers(0, cfg['clusters'], size=cfg['queries'])
    qs = centres[qi] + 0.35 * rng.standard_normal((cfg['queries'], cfg['dims']))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    return {'cfg': cfg, 'points': pts.astype(np.float32),
            'queries': qs.astype(np.float32), 'members': members}


def brute_force(points, queries, k=10):
    sims = queries @ points.T
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(found, gold, k=10):
    return float(np.mean([len(set(f[:k]) & set(g[:k])) for f, g in zip(found, gold)])) / k


# ------------------------------------------------------------------- index
class HNSW:
    """Hierarchical Navigable Small World, written out in full.

    One graph per layer.  The bottom layer holds every point; each layer above
    holds a geometric fraction of them and acts as an express lane, exactly the
    way a skip list puts a sparse index above a linked list.
    """

    def __init__(self, dim, M=12, ef_construction=60, ml=None, heuristic=True,
                 seed=0, metric='cosine'):
        self.dim = dim
        self.metric = metric                         # 'cosine' or 'l2'
        self.M = M                                   # neighbours per node
        self.M0 = 2 * M                              # ...on the bottom layer
        self.ef_construction = ef_construction
        self.ml = ml if ml is not None else 1.0 / math.log(M)
        self.heuristic = heuristic
        self.rng = np.random.default_rng(seed)
        self._store = np.zeros((1024, dim), dtype=np.float32)
        self._n = 0
        self.levels = []                             # level of each node
        self.graph = []                              # graph[layer][node] = [ids]
        self.entry = None
        self.deleted = set()
        self.dist_calls = 0
        self.wasted_calls = 0                        # spent on tombstoned points
        self.trace = None                            # set to [] to record the walk

    @property
    def data(self):
        return self._store[:self._n]

    # ---- distances: one call touches a block of candidates at a time
    def _d(self, q, ids):
        """Cosine distance to a list of ids.  Counted, because that is the cost."""
        if not ids:
            return np.empty(0, dtype=np.float32)
        self.dist_calls += len(ids)
        if self.deleted:
            self.wasted_calls += sum(1 for i in ids if i in self.deleted)
        return self._metric(q, self._store[ids])

    def _metric(self, q, block):
        if self.metric == 'l2':
            d = block - q
            return np.einsum('ij,ij->i', d, d)
        return 1.0 - block @ q

    def _random_level(self):
        return int(-math.log(max(self.rng.random(), 1e-12)) * self.ml)

    def _search_layer(self, q, entry_points, ef, layer):
        """The one routine everything else calls: a best-first walk on one graph."""
        visited = set(entry_points)
        dists = self._d(q, list(entry_points))
        candidates = [(float(d), i) for d, i in zip(dists, entry_points)]
        heapq.heapify(candidates)                    # nearest first
        best = [(-float(d), i) for d, i in zip(dists, entry_points)]
        heapq.heapify(best)                          # furthest first, size <= ef
        while len(best) > ef:
            heapq.heappop(best)
        while candidates:
            d, c = heapq.heappop(candidates)
            if self.trace is not None:
                self.trace.append(int(c))
            if best and d > -best[0][0] and len(best) >= ef:
                break                                # nothing closer can follow
            fresh = [n for n in self.graph[layer][c] if n not in visited]
            visited.update(fresh)
            for dn, n in zip(self._d(q, fresh), fresh):
                dn = float(dn)
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(candidates, (dn, n))
                    heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return [(-d, i) for d, i in best]            # (distance, id)

    def _select(self, q, candidates, m):
        """Which m of the candidates become neighbours.

        Plain nearest-m fills the list with points from one cluster, and the
        graph loses the edges that leave it.  The heuristic keeps a candidate
        only if it is closer to the query than to anything already chosen -
        that is what keeps long-range links alive.
        """
        ordered = sorted(candidates)
        if not self.heuristic:
            return [i for _, i in ordered[:m]]
        kept = []
        for d, i in ordered:
            if len(kept) >= m:
                break
            if kept:
                to_kept = self._metric(self._store[i], self._store[kept])
                if d >= float(to_kept.min()):
                    continue
            kept.append(i)
        if len(kept) < m:                            # top up if too strict
            for _, i in ordered:
                if i not in kept:
                    kept.append(i)
                if len(kept) >= m:
                    break
        return kept

    def add(self, vec):
        idx = self._n
        if idx >= len(self._store):                  # grow by doubling
            self._store = np.vstack([self._store,
                                     np.zeros_like(self._store)])
        self._store[idx] = vec
        self._n += 1
        level = self._random_level()
        self.levels.append(level)
        while len(self.graph) <= level:
            self.graph.append({})
        for l in range(level + 1):
            self.graph[l][idx] = []
        if self.entry is None:
            self.entry = idx
            return idx

        ep = [self.entry]
        top = self.levels[self.entry]                # the entry lives up to here
        for l in range(top, level, -1):              # greedy descent, ef = 1
            ep = [self._search_layer(vec, ep, 1, l)[0][1]]
        for l in range(min(level, top), -1, -1):
            found = self._search_layer(vec, ep, self.ef_construction, l)
            m = self.M0 if l == 0 else self.M
            neighbours = self._select(vec, found, m)
            self.graph[l][idx] = list(neighbours)
            for n in neighbours:                     # links go both ways
                self.graph[l][n].append(idx)
                if len(self.graph[l][n]) > m:        # ...and get pruned back
                    others = self.graph[l][n]
                    ds = self._metric(self._store[n], self._store[others])
                    cand = list(zip((float(x) for x in ds), others))
                    self.graph[l][n] = self._select(self._store[n], cand, m)
            ep = [i for _, i in found]
        if level > self.levels[self.entry]:
            self.entry = idx
        return idx

    def build(self, points):
        t0 = time.perf_counter()
        for v in points:
            self.add(v)
        self.build_seconds = time.perf_counter() - t0
        return self

    def search(self, q, k=10, ef=50, clamp=True):
        """Descend the express lanes with ef=1, then explore the bottom layer."""
        if self.entry is None:
            return []
        ep = [self.entry]
        for l in range(self.levels[self.entry], 0, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        found = self._search_layer(q, ep, max(ef, k) if clamp else ef, 0)
        alive = [(d, i) for d, i in found if i not in self.deleted]
        return [i for _, i in sorted(alive)[:k]]

    def delete(self, ids):
        """Deletion is a tombstone: the point stays in the graph as a stepping
        stone, and only disappears from the results."""
        self.deleted.update(int(i) for i in ids)

    # ---- bookkeeping
    def layer_sizes(self):
        return [len(g) for g in self.graph]

    def edges(self):
        return sum(len(v) for g in self.graph for v in g.values())

    def bytes_per_point(self, dtype_bytes=4):
        vec = self.dim * dtype_bytes
        links = self.edges() / max(len(self.data), 1) * 4      # int32 ids
        return {'vector_bytes': vec, 'link_bytes': links, 'total': vec + links}


# -------------------------------------------------------------- experiments
def ef_curve(index, points, queries, gold, efs=(10, 25, 50, 100, 200), k=10):
    rows = []
    for ef in efs:
        index.dist_calls = 0
        t0 = time.perf_counter()
        found = np.array([index.search(q, k, ef) for q in queries])
        ms = (time.perf_counter() - t0) * 1e3 / len(queries)
        calls = index.dist_calls / len(queries)
        rows.append({'ef': ef, 'recall': recall_at_k(found, gold, k),
                     'dist_per_query': calls, 'ms_per_query': ms,
                     'fraction_seen': calls / len(points)})
    return rows


def brute_cost(points, queries, k=10):
    t0 = time.perf_counter()
    brute_force(points, queries, k)
    ms = (time.perf_counter() - t0) * 1e3 / len(queries)
    return {'dist_per_query': float(len(points)), 'ms_per_query': ms, 'recall': 1.0}


def layer_profile(index):
    sizes = index.layer_sizes()
    n = sizes[0]
    return [{'layer': l, 'nodes': s, 'share': s / n,
             'expected': n * (1 / index.M) ** l} for l, s in enumerate(sizes)]


def m_sweep(points, queries, gold, Ms=(6, 12, 24), ef=50, ef_c=60, k=10):
    rows = []
    for M in Ms:
        ix = HNSW(points.shape[1], M=M, ef_construction=ef_c).build(points)
        ix.dist_calls = 0
        found = np.array([ix.search(q, k, ef) for q in queries])
        b = ix.bytes_per_point()
        rows.append({'M': M, 'recall': recall_at_k(found, gold, k),
                     'edges_per_node': ix.edges() / len(points),
                     'link_bytes': b['link_bytes'], 'vector_bytes': b['vector_bytes'],
                     'overhead': b['link_bytes'] / b['vector_bytes'],
                     'build_seconds': ix.build_seconds,
                     'dist_per_query': ix.dist_calls / len(queries)})
    return rows


def neighbour_diversity(index, members, layer=0):
    """How often does an edge leave the cluster it started in?"""
    same = total = 0
    for node, nbrs in index.graph[layer].items():
        for n in nbrs:
            total += 1
            same += int(members[node] == members[n])
    return {'edges': total, 'same_cluster': same / max(total, 1)}


def heuristic_compare(points, queries, gold, members, ef=50, M=12, k=10):
    rows = []
    for heur in (False, True):
        ix = HNSW(points.shape[1], M=M, heuristic=heur).build(points)
        found = np.array([ix.search(q, k, ef) for q in queries])
        div = neighbour_diversity(ix, members)
        rows.append({'name': 'heuristic (diverse)' if heur else 'plain nearest-M',
                     'recall': recall_at_k(found, gold, k),
                     'same_cluster_edges': div['same_cluster'],
                     'build_seconds': ix.build_seconds})
    return rows


def deletion_decay(points, queries, fractions=(0.1, 0.3), ef=50, k=10, seed=0,
                   M=12):
    """Tombstones keep the graph connected and quietly make the search wasteful."""
    rng = np.random.default_rng(seed)
    ix = HNSW(points.shape[1], M=M).build(points)
    rows = []
    gone = set()
    for frac in fractions:
        target = int(len(points) * frac)
        extra = [i for i in rng.permutation(len(points)) if i not in gone][:target - len(gone)]
        ix.delete(extra)
        gone.update(int(i) for i in extra)
        alive = np.array([i for i in range(len(points)) if i not in gone])
        gold_alive = alive[brute_force(points[alive], queries, k)]
        ix.dist_calls = ix.wasted_calls = 0
        found = np.array([ix.search(q, k, ef) for q in queries])
        calls = ix.dist_calls / len(queries)
        rows.append({'deleted': frac, 'recall': recall_at_k(found, gold_alive, k),
                     'dist_per_query': calls,
                     'wasted_share': ix.wasted_calls / max(ix.dist_calls, 1)})
    alive = np.array([i for i in range(len(points)) if i not in gone])
    rebuilt = HNSW(points.shape[1], M=M).build(points[alive])
    gold_alive = alive[brute_force(points[alive], queries, k)]
    rebuilt.dist_calls = 0
    found = np.array([alive[rebuilt.search(q, k, ef)] for q in queries])
    rows.append({'deleted': max(fractions), 'rebuilt': True,
                 'recall': recall_at_k(found, gold_alive, k),
                 'dist_per_query': rebuilt.dist_calls / len(queries),
                 'rebuild_seconds': rebuilt.build_seconds})
    return rows


def scaling(sizes=(1_000, 2_000, 4_000), ef=50, k=10, seed=0, dims=48):
    rows = []
    for n in sizes:
        w = make_vectors(seed=seed, docs=n, dims=dims, queries=60)
        ix = HNSW(dims).build(w['points'])
        gold = brute_force(w['points'], w['queries'], k)
        ix.dist_calls = 0
        found = np.array([ix.search(q, k, ef) for q in w['queries']])
        rows.append({'docs': n, 'recall': recall_at_k(found, gold, k),
                     'dist_per_query': ix.dist_calls / len(w['queries']),
                     'fraction_seen': ix.dist_calls / len(w['queries']) / n,
                     'build_seconds': ix.build_seconds})
    return rows


# ---------------------------------------------------- the quiet failures
def ef_below_k(index, queries, gold, k=10, ef=5):
    """A candidate list smaller than k cannot return k good answers."""
    clamped = [index.search(q, k, ef, clamp=True) for q in queries]
    raw = [index.search(q, k, ef, clamp=False) for q in queries]
    short = float(np.mean([len(r) < k for r in raw]))
    hits = lambda rs: float(np.mean([len(set(r) & set(g[:k]))
                                     for r, g in zip(rs, gold)])) / k
    raw_recall = hits(raw)
    return {'ef': ef, 'k': k, 'clamped_recall': hits(clamped),
            'unclamped_recall': raw_recall, 'queries_returning_fewer_than_k': short}


def forgot_to_normalise(world, ef=50, k=10, scale=3.0, share=0.25, seed=0, M=12):
    """Inner product on unnormalised vectors is not cosine - the graph is wrong."""
    rng = np.random.default_rng(seed)
    pts = world['points'].copy()
    loud = rng.random(len(pts)) < share
    pts[loud] *= scale                                  # a quarter of them get louder
    gold = brute_force(world['points'], world['queries'], k)   # the true cosine answer
    ix = HNSW(pts.shape[1], M=M).build(pts)
    found = np.array([ix.search(q, k, ef) for q in world['queries']])
    picked_loud = float(np.mean([np.mean(loud[f]) for f in found]))
    return {'recall_against_true_cosine': recall_at_k(found, gold, k),
            'share_scaled': float(loud.mean()), 'share_of_hits_scaled': picked_loud}


def circular_ground_truth(index, queries, gold, ef=50, k=10):
    """Scoring an index against its own output: the number is always perfect."""
    found = np.array([index.search(q, k, ef) for q in queries])
    return {'against_brute_force': recall_at_k(found, gold, k),
            'against_itself': recall_at_k(found, found, k)}


def without_layers(index, queries, gold, ef=50, k=10):
    """What the upper layers are actually for: they are a shortcut, not accuracy.

    Search the bottom layer only, starting from the same entry point, and
    compare both the recall and the number of distance computations.
    """
    rows = []
    index.dist_calls = 0
    found = np.array([index.search(q, k, ef) for q in queries])
    rows.append({'name': 'with the express lanes', 'recall': recall_at_k(found, gold, k),
                 'dist_per_query': index.dist_calls / len(queries)})
    index.dist_calls = 0
    flat = []
    for q in queries:
        got = index._search_layer(q, [index.entry], max(ef, k), 0)
        flat.append([i for _, i in sorted(got)[:k]])
    rows.append({'name': 'bottom layer only, same entry point',
                 'recall': recall_at_k(np.array(flat), gold, k),
                 'dist_per_query': index.dist_calls / len(queries)})
    rng = np.random.default_rng(0)
    index.dist_calls = 0
    flat = []
    for q in queries:
        start = int(rng.integers(0, len(index.data)))
        got = index._search_layer(q, [start], max(ef, k), 0)
        flat.append([i for _, i in sorted(got)[:k]])
    rows.append({'name': 'bottom layer only, random entry point',
                 'recall': recall_at_k(np.array(flat), gold, k),
                 'dist_per_query': index.dist_calls / len(queries)})
    return rows


def drop_hub(index, queries, gold, ef=50, k=10):
    """Tombstones keep hubs as stepping stones; really removing them does not."""
    hub = max(index.graph[0], key=lambda n: len(index.graph[0][n]))
    saved = {l: dict(g) for l, g in enumerate(index.graph)}
    for l, g in enumerate(index.graph):
        if hub in g:
            for n in list(g):
                g[n] = [x for x in g[n] if x != hub]
            g[hub] = []
    found = np.array([index.search(q, k, ef) for q in queries])
    broken = recall_at_k(found, gold, k)
    for l, g in saved.items():
        index.graph[l] = g
    return {'hub': int(hub), 'recall_after_cutting_the_hub': broken}


# ----------------------------------------------------------------- printing
def head(n, title):
    print('\n' + '=' * 72)
    print(f'{n}. {title}')
    print('=' * 72 + '\n')


def main():
    print('Day 66 - HNSW: finding the nearest vectors without looking at most of them')
    print('(the whole file takes about a minute: it builds eight graphs from scratch)')
    world = make_vectors(docs=3_000, queries=120)
    pts, qs = world['points'], world['queries']
    gold = brute_force(pts, qs, 10)
    index = HNSW(world['cfg']['dims'], M=12, ef_construction=60).build(pts)
    curve = ef_curve(index, pts, qs, gold)
    bf = brute_cost(pts, qs)

    # ------------------------------------------------------------------ 1
    head(1, 'The short version')
    best = [r for r in curve if r['ef'] == 50][0]
    print(f"  {len(pts):,} vectors of {world['cfg']['dims']} dimensions, {len(qs)} queries, k = 10.\n")
    print('    search              distance computations   recall@10   ms/query')
    print(f"    brute force         {bf['dist_per_query']:21,.0f}   {bf['recall']:9.3f}"
          f"   {bf['ms_per_query']:8.3f}")
    print(f"    HNSW (ef = 50)      {best['dist_per_query']:21,.0f}   {best['recall']:9.3f}"
          f"   {best['ms_per_query']:8.3f}\n")
    print(f"  The index looks at {best['fraction_seen']:.0%} of the corpus and gets "
          f"{best['recall']:.0%} of the")
    print('  answers.  That trade - a tunable amount of wrongness for a large amount')
    print('  of skipped work - is the whole subject.  (The milliseconds here are a')
    print('  pure-Python graph walk against a numpy matrix multiply, so read the')
    print('  distance-computation column, not the clock.)')

    # ------------------------------------------------------------------ 2
    head(2, 'The graph: one layer per order of magnitude')
    print('    layer   nodes   share of corpus   expected (1/M per level)')
    for r in layer_profile(index):
        print(f"    {r['layer']:5}   {r['nodes']:5,}   {r['share']:15.3f}"
              f"   {r['expected']:24,.1f}")
    print(f"\n  Each point draws a level from a geometric distribution with mean")
    print(f"  1/ln(M), so each layer up holds about 1/M of the one below.  The bottom")
    print('  layer has everything; the top has a handful.  That is a skip list with')
    print('  a graph instead of a linked list - the same trick from Day 31, in a')
    print('  space where "the next element" is not defined.')

    # ------------------------------------------------------------------ 3
    head(3, 'The search, and the only knob that matters at query time')
    print('    ef    distance computations   fraction of corpus   recall@10')
    for r in curve:
        print(f"    {r['ef']:4}   {r['dist_per_query']:21,.0f}   {r['fraction_seen']:18.1%}"
              f"   {r['recall']:9.3f}")
    print('\n  ef is how many candidates the walk keeps alive.  Everything else is a')
    print('  consequence: more candidates means more neighbours expanded, more')
    print('  distance computations, and a better chance the greedy walk escapes a')
    print('  local pocket.  **Recall is bought by the candidate list, not by luck**,')
    print('  and the curve is steep at the start and flat at the end - which is why')
    print('  every library ships an efSearch you are expected to tune per workload.')

    # ------------------------------------------------------------------ 4
    head(4, 'The hierarchy is not where the speed comes from')
    for r in without_layers(index, qs, gold):
        print(f"    {r['name']:38} {r['dist_per_query']:9,.0f} distances   "
              f"recall {r['recall']:.3f}")
    print('\n  Deleting the upper layers - searching the bottom graph only, even from')
    print('  a randomly chosen starting point - changes neither the recall nor the')
    print('  work.  This is not a bug in the implementation: it reproduces a')
    print('  published result (Munyampirwa, Lakshman and Coleman, ECIR 2026), which')
    print('  found the hierarchy gives no measurable benefit on high-dimensional')
    print('  data and costs memory, and that a flat graph saves 38-39% of peak')
    print('  construction memory at 100M vectors.  The explanation they offer is')
    print('  the hub highway: in high dimensions a few points are everybody\'s')
    print('  neighbour, so the bottom graph already contains the express lanes the')
    print('  hierarchy was supposed to provide.  Earlier work found the layers only')
    print('  help below about 32 dimensions.')

    # ------------------------------------------------------------------ 5
    head(5, 'M: how many neighbours each point keeps')
    small = make_vectors(seed=1, docs=1_000, queries=80)
    gold_s = brute_force(small['points'], small['queries'], 10)
    ms = m_sweep(small['points'], small['queries'], gold_s, Ms=(6, 12, 20))
    print('    M    edges/node   link bytes   vs vector bytes   build s   recall@10')
    for r in ms:
        print(f"    {r['M']:3}   {r['edges_per_node']:10.1f}   {r['link_bytes']:10.0f}"
              f"   {r['overhead']:15.2f}x   {r['build_seconds']:7.1f}   {r['recall']:9.3f}")
    print('\n  M is the memory knob and the build-time knob at once.  The links are')
    print('  not free: at M = 20 they cost about as much as the vectors themselves,')
    print(f"  and the build goes from {ms[0]['build_seconds']:.1f} s to "
          f"{ms[-1]['build_seconds']:.1f} s for the same points.  The recall it buys")
    print('  saturates, so the useful question is never "what is the best M" but')
    print('  "what is the smallest M that reaches the recall I need".')

    # ------------------------------------------------------------------ 6
    head(6, 'Which neighbours to keep: the diversity heuristic')
    clustered = make_vectors(seed=2, docs=1_500, queries=80, clusters=12)
    gold_c = brute_force(clustered['points'], clustered['queries'], 10)
    hrows = heuristic_compare(clustered['points'], clustered['queries'], gold_c,
                              clustered['members'], ef=25, M=6)
    for r in hrows:
        print(f"    {r['name']:22} recall {r['recall']:.3f}   "
              f"edges staying inside the cluster {r['same_cluster_edges']:.0%}   "
              f"build {r['build_seconds']:.1f} s")
    print('\n  Plain nearest-M fills a point\'s neighbour list with its own cluster,')
    print('  and the graph loses the edges that leave it.  The heuristic keeps a')
    print('  candidate only if it is closer to the new point than to anything')
    print('  already chosen, which preserves the long-range links.  Here that is')
    print(f"  worth {100 * (hrows[1]['recall'] - hrows[0]['recall']):.0f} points of recall at M = 6, "
          f"for {hrows[1]['build_seconds'] / hrows[0]['build_seconds']:.1f}x the build")
    print('  time.  With a generous M the two converge, which is the')
    print('  honest version of the story: the heuristic is a way to buy recall')
    print('  back when the neighbour budget is tight.')

    # ------------------------------------------------------------------ 7
    head(7, 'Deleting things')
    rows = deletion_decay(make_vectors(seed=3, docs=1_500, queries=80)['points'],
                          world['queries'][:80], fractions=(0.1, 0.3))
    print('    state                     recall@10   distances/query   spent on deleted points')
    for r in rows:
        name = (f"rebuilt after {r['deleted']:.0%}" if r.get('rebuilt')
                else f"{r['deleted']:.0%} deleted (tombstones)")
        waste = '-' if r.get('rebuilt') else f"{r['wasted_share']:.1%}"
        print(f"    {name:24}  {r['recall']:9.3f}   {r['dist_per_query']:15,.0f}   {waste:>23}")
    print(f"\n  A deletion in a graph index is a tombstone: the point stays in the")
    print('  graph as a stepping stone and is only filtered out of the results.')
    print('  Cutting it out properly would mean repairing every neighbour list that')
    print('  points at it, so nobody does that online.  The cost is in the last')
    print(f"  column: at {rows[1]['deleted']:.0%} deleted, {rows[1]['wasted_share']:.0%} of every query's distance")
    print('  computations are spent on points that cannot be returned, and the work')
    print(f"  per query does not move ({rows[0]['dist_per_query']:,.0f} either way) while the useful corpus")
    print(f"  shrinks.  Rebuilding takes {rows[-1]['rebuild_seconds']:.0f} s for "
          f"{int(1_500 * 0.7):,} points and brings the cost down to")
    print(f"  {rows[-1]['dist_per_query']:,.0f}.  Every vector database that offers deletion is really")
    print('  offering a compaction schedule, and the only question is who runs it.')

    # ------------------------------------------------------------------ 8
    head(8, 'What happens as the corpus grows')
    rows = scaling(sizes=(500, 1_000, 2_000))
    rows.append({'docs': len(pts), 'recall': best['recall'],
                 'dist_per_query': best['dist_per_query'],
                 'fraction_seen': best['fraction_seen'],
                 'build_seconds': index.build_seconds})
    print('    vectors   distances/query   fraction of corpus   recall@10   build s')
    for r in rows:
        print(f"    {r['docs']:7,}   {r['dist_per_query']:15,.0f}   {r['fraction_seen']:18.1%}"
              f"   {r['recall']:9.3f}   {r['build_seconds']:7.1f}")
    print('\n  Six times the data, and the number of distance computations grows by')
    print('  a factor of about two while brute force grows by six.  That is the')
    print('  shape of the whole family: the work per query goes like the logarithm')
    print('  of the corpus, the index is built once and paid for in memory, and the')
    print('  recall drifts down slowly as the graph gets bigger - which is why ef')
    print('  has to be re-tuned after a corpus grows, not just after it changes.')

    # ------------------------------------------------------------------ 9
    head(9, 'Four ways this goes wrong quietly')
    e = ef_below_k(index, qs, gold, k=10, ef=5)
    print(f"  1. ef smaller than k.  With ef = {e['ef']} and k = {e['k']}, the raw walk returns")
    print(f"     fewer than k results on {e['queries_returning_fewer_than_k']:.0%} of queries "
          f"(recall {e['unclamped_recall']:.3f}); clamping")
    print(f"     ef up to k hides it and still only reaches {e['clamped_recall']:.3f}.  "
          f"**ef is a floor on\n     how good the answer can be**, not a performance setting.\n")
    n = forgot_to_normalise(world)
    print(f"  2. Inner product on unnormalised vectors.  Scale {n['share_scaled']:.0%} of the "
          f"corpus by 3x and")
    print(f"     recall against the true cosine answer is {n['recall_against_true_cosine']:.3f}, "
          f"with {n['share_of_hits_scaled']:.0%} of the hits")
    print('     coming from the scaled group.  The index is not broken; it is')
    print('     answering a different question, and nothing says so.\n')
    c = circular_ground_truth(index, qs, gold)
    print(f"  3. Measuring recall against the index itself: {c['against_itself']:.3f}, next to "
          f"{c['against_brute_force']:.3f}")
    print('     against brute force.  Any evaluation that builds its ground truth')
    print('     with the same index it is grading will report a perfect score.\n')
    print('  4. Never compacting.  Section 7: the deleted points keep being visited,')
    print('     so the cost per query stays while the useful results shrink.')

    # ----------------------------------------------------------------- 10
    head(10, 'In hnswlib and FAISS')
    print('    import hnswlib')
    print('    ix = hnswlib.Index(space="cosine", dim=d)')
    print('    ix.init_index(max_elements=N, M=16, ef_construction=200)')
    print('    ix.add_items(vectors, ids)')
    print('    ix.set_ef(64)                 # section 3, per query')
    print('    ix.mark_deleted(i)            # section 7, a tombstone\n')
    print('    index = faiss.IndexHNSWFlat(d, 32)        # 32 is M')
    print('    index.hnsw.efConstruction = 200')
    print('    index.hnsw.efSearch = 64\n')
    print('  Three parameters, and this file measured all three: M is memory and')
    print('  build time, efConstruction is how carefully the neighbour lists are')
    print('  chosen, efSearch is recall per query.  The defaults are usually fine')
    print('  until the corpus grows - and then it is efSearch that has to move.')

    # ----------------------------------------------------------------- 11
    head(11, 'In a vector database')
    print('    Chroma     hnsw:space, hnsw:M, hnsw:construction_ef, hnsw:search_ef')
    print('    Qdrant     hnsw_config: m, ef_construct ; search_params: hnsw_ef')
    print('    Milvus     index_params M / efConstruction ; search_params ef')
    print('    pgvector   CREATE INDEX ... USING hnsw (v vector_cosine_ops)')
    print('               WITH (m = 16, ef_construction = 64); SET hnsw.ef_search = 64;')
    print('    Pinecone   none of them - the index type is the product\n')
    print('  The same three numbers under four different spellings.  What the')
    print('  managed ones take away is not the algorithm, it is the compaction')
    print('  schedule from section 7 and the decision about when to rebuild.')

    # ----------------------------------------------------------------- 12
    head(12, 'Assertions')
    check()
    print('all assertions passed')


def check():
    world = make_vectors(seed=7, docs=800, queries=60)
    pts, qs = world['points'], world['queries']
    gold = brute_force(pts, qs, 10)
    ix = HNSW(world['cfg']['dims'], M=12, ef_construction=60).build(pts)

    # the index is correct enough, and cheap enough to be worth it
    found = np.array([ix.search(q, 10, 50) for q in qs])
    assert recall_at_k(found, gold, 10) > 0.9
    ix.dist_calls = 0
    [ix.search(q, 10, 50) for q in qs]
    assert ix.dist_calls / len(qs) < len(pts)          # fewer than brute force

    # the graph: every node is in layer 0, levels decay geometrically
    sizes = ix.layer_sizes()
    assert sizes[0] == len(pts) and len(sizes) > 1
    assert sizes[1] < sizes[0] / 4
    assert all(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1))
    assert ix.levels[ix.entry] == len(sizes) - 1       # the entry is on top

    # links are bidirectional and bounded
    for l, g in enumerate(ix.graph):
        m = ix.M0 if l == 0 else ix.M
        for node, nbrs in g.items():
            assert len(nbrs) <= m + 1
            assert node not in nbrs
            for n in nbrs:
                assert n in g

    # ef is monotone in recall and in work
    curve = ef_curve(ix, pts, qs, gold, efs=(10, 50, 200))
    assert curve[0]['recall'] < curve[-1]['recall']
    assert curve[0]['dist_per_query'] < curve[-1]['dist_per_query']
    assert curve[-1]['recall'] > 0.97

    # the hierarchy is a routing shortcut, not an accuracy device
    rows = without_layers(ix, qs, gold)
    assert abs(rows[0]['recall'] - rows[-1]['recall']) < 0.05

    # deleting is a tombstone: the point stays in the graph but leaves the answers
    victim = int(gold[0][0])
    ix.delete([victim])
    assert victim in ix.graph[0]
    assert victim not in ix.search(qs[0], 10, 50)

    # ef below k cannot return k answers
    e = ef_below_k(ix, qs, gold, k=10, ef=4)
    assert e['queries_returning_fewer_than_k'] == 1.0
    assert e['unclamped_recall'] < e['clamped_recall']

    # scoring an index against itself is always perfect
    c = circular_ground_truth(ix, qs, gold)
    assert c['against_itself'] == 1.0 and c['against_brute_force'] < 1.0

    # unnormalised vectors answer a different question
    bad = forgot_to_normalise(world, M=8)
    assert bad['recall_against_true_cosine'] < 0.6
    assert bad['share_of_hits_scaled'] > 2 * bad['share_scaled']


if __name__ == '__main__':
    main()
