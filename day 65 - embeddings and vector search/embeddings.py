"""Day 65 - embeddings and vector search.

An embedding turns a document into a direction in space, and retrieval turns a
question into "whose direction is closest".  This module builds a corpus whose
true meaning is known, trains an embedding on it with nothing but a term count
matrix and an SVD, and then measures what every practical decision costs:
normalising or not, how many dimensions to keep, how few bits per number,
one vector per document or several, and when to pay for a reranker.

Run it:  python embeddings.py
"""

import time

import numpy as np

# ------------------------------------------------------------ the corpus
CORPUS = {'topics': 24, 'vocab': 800, 'docs': 20_000, 'queries': 200,
          'targets': 10, 'target_noise': 0.15, 'query_len': (24, 48),
          'topic_alpha': 0.03, 'doc_len': 200, 'train_docs': 4_000, 'dims': 256}


def make_world(seed=0, **kw):
    """A corpus whose *meaning* we know, and whose answers are planted.

    Each topic is a distribution over the vocabulary.  Most documents are
    background: one to three random topics.  For every query we also plant
    `targets` documents drawn from that query's own mixture (with noise and
    a random length), so "the right answer" is a known set of ids instead of
    a judgement call - which is what lets us score an embedding at all.
    """
    cfg = dict(CORPUS, **kw)
    rng = np.random.default_rng(seed)
    T, V = cfg['topics'], cfg['vocab']
    qn, tgt = cfg['queries'], cfg['targets']
    n_bg = cfg['docs'] - qn * tgt

    topic_word = rng.dirichlet(np.full(V, cfg['topic_alpha']), size=T)   # (T, V)

    def single(t):
        m = np.zeros(T)
        m[t] = 1.0
        return m

    def pair(a, b, w):
        m = np.zeros(T)
        m[a], m[b] = w, 1 - w
        return m

    # background: one topic each.  queries: a *pair* of topics that no
    # background document uses, so the planted answers are the only
    # documents that match both halves of the question.
    bg_topics = rng.integers(0, T, size=n_bg)
    mixes = [single(int(t)) for t in bg_topics]
    qpairs = [tuple(rng.choice(T, size=2, replace=False)) for _ in range(qn)]
    qmix = np.array([pair(a, b, 0.6) for a, b in qpairs])
    gold = []
    for qi, (a, b) in enumerate(qpairs):
        ids = []
        for _ in range(tgt):
            w = 0.6 + rng.uniform(-cfg['target_noise'], cfg['target_noise'])
            ids.append(len(mixes))
            mixes.append(pair(a, b, float(np.clip(w, 0.25, 0.85))))
        gold.append(ids)
    mix = np.array(mixes)
    order = rng.permutation(len(mix))                   # planted ids are not at the end
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    mix = mix[order]
    gold = np.array([[inverse[i] for i in ids] for ids in gold])

    lengths = np.clip(rng.lognormal(np.log(cfg['doc_len']), 0.8, size=len(mix)),
                      20, 2000).astype(int)
    counts = np.zeros((len(mix), V), dtype=np.float32)
    probs = mix @ topic_word
    for i in range(len(mix)):
        counts[i] = rng.multinomial(lengths[i], probs[i])

    qlen = rng.integers(cfg['query_len'][0], cfg['query_len'][1], size=qn)
    qcounts = np.zeros((qn, V), dtype=np.float32)
    qprobs = qmix @ topic_word
    for i in range(qn):
        qcounts[i] = rng.multinomial(qlen[i], qprobs[i])

    return {'cfg': cfg, 'counts': counts, 'lengths': lengths, 'mix': mix,
            'qcounts': qcounts, 'qmix': qmix, 'qlen': qlen, 'gold': gold,
            'topic_word': topic_word}


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def truth(world):
    """The answer key: the planted documents, by construction."""
    return world['gold']


# ------------------------------------------------- the embedding, from scratch
def tfidf(counts, idf=None):
    tf = np.log1p(counts)
    if idf is None:
        df = (counts > 0).sum(axis=0) + 1
        idf = np.log(counts.shape[0] / df)
    return tf * idf, idf


def train_embedding(world, dims=None, seed=0):
    """Twenty lines: term counts -> tf-idf -> SVD -> dense vectors.

    The singular vectors come out ordered by how much of the corpus they
    explain, so the first d columns are already the best d-dimensional
    version of the same embedding - the property Matryoshka training puts
    into a neural encoder on purpose.
    """
    cfg = world['cfg']
    dims = dims or cfg['dims']
    rng = np.random.default_rng(seed)
    train = rng.choice(cfg['docs'], size=cfg['train_docs'], replace=False)
    X, idf = tfidf(world['counts'][train])
    _, s, vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    basis = vt[:dims].T                                            # (V, dims)
    doc_tfidf, _ = tfidf(world['counts'], idf)
    query_tfidf, _ = tfidf(world['qcounts'], idf)
    return {'basis': basis, 'idf': idf, 's': s[:dims],
            'docs': (doc_tfidf @ basis).astype(np.float32),
            'queries': (query_tfidf @ basis).astype(np.float32)}


# ------------------------------------------------------------------ scoring
def topk(scores, k=10):
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(found, gold, k=10):
    hits = [len(set(f[:k]) & set(g[:k])) for f, g in zip(found, gold)]
    return float(np.mean(hits)) / k


def search(qvec, dvec, k=10, normalise=True):
    q = unit(qvec) if normalise else qvec
    d = unit(dvec) if normalise else dvec
    return topk(q @ d.T, k)


# ------------------------------------------------------- normalise or not
def length_bias(world, emb, k=10):
    """Dot product rewards长 documents; cosine asks only about direction."""
    gold = truth(world)
    norms = np.linalg.norm(emb['docs'], axis=1)
    corr = float(np.corrcoef(norms, world['lengths'])[0, 1])
    out = {}
    for name, norm in (('cosine (normalised)', True), ('raw dot product', False)):
        found = search(emb['queries'], emb['docs'], k, norm)
        picked = world['lengths'][found].mean()
        out[name] = {'recall': recall_at_k(found, gold, k),
                     'mean_length_of_hits': float(picked)}
    return {'norm_vs_length_corr': corr, 'median_length': float(np.median(world['lengths'])),
            'rows': out}


# --------------------------------------------------------- how many dimensions
def dimension_curve(world, emb, dims=(256, 128, 64, 32, 16, 8, 4), k=10):
    gold = truth(world)
    rows = []
    for d in dims:
        found = search(emb['queries'][:, :d], emb['docs'][:, :d], k)
        rows.append({'dims': d, 'recall': recall_at_k(found, gold, k),
                     'bytes_per_doc': d * 4,
                     'gb_per_million': d * 4 * 1e6 / 1024 ** 3})
    return rows


# ------------------------------------------------------------- fewer bits
def calibrate(vecs):
    """What a real library stores alongside the index: per-dimension ranges.

    Quantization needs to know the range of each dimension *over the corpus*;
    scaling each vector by its own maximum throws away exactly the comparison
    the search is about.
    """
    v = unit(vecs)
    return {'scale': float(np.abs(v).max()), 'mean': v.mean(axis=0)}


def quantize_int8(vecs, cal):
    """One scale for the whole corpus, so a dot product survives the rounding.

    Per-dimension offsets look tempting and quietly break the comparison: the
    score stops being proportional to the original dot product.
    """
    v = unit(vecs) / cal['scale'] * 127.0
    return np.clip(np.round(v), -127, 127).astype(np.int8), cal['scale']


def quantize_binary(vecs, cal):
    """One bit per dimension: is this dimension above the corpus average?"""
    return (unit(vecs) - cal['mean']) > 0


def search_binary(qbits, dbits, k=10):
    """Hamming distance is a dot product on +-1, so numpy can do it directly."""
    q = qbits.astype(np.int8) * 2 - 1
    d = dbits.astype(np.int8) * 2 - 1
    return topk((q @ d.T).astype(np.float32), k)


def rescore(cand, qvec, dvec, k=10):
    """Fetch the full-precision vectors of a shortlist and re-rank them."""
    q = unit(qvec)
    d = unit(dvec)
    out = np.zeros((len(cand), k), dtype=int)
    for i, ids in enumerate(cand):
        s = d[ids] @ q[i]
        out[i] = ids[np.argsort(-s)[:k]]
    return out


def quantization_table(world, emb, k=10, oversample=4):
    gold = truth(world)
    rows = []
    base = search(emb['queries'], emb['docs'], k)
    dims = emb['docs'].shape[1]
    rows.append({'name': 'float32', 'bytes_per_doc': dims * 4,
                 'recall': recall_at_k(base, gold, k), 'rescored': False})
    cal = calibrate(emb['docs'])
    qi, _ = quantize_int8(emb['queries'], cal)
    di, _ = quantize_int8(emb['docs'], cal)
    found = topk((qi.astype(np.float32) @ di.astype(np.float32).T), k)
    rows.append({'name': 'int8', 'bytes_per_doc': dims,
                 'recall': recall_at_k(found, gold, k), 'rescored': False})
    qb, db = quantize_binary(emb['queries'], cal), quantize_binary(emb['docs'], cal)
    fb = search_binary(qb, db, k)
    rows.append({'name': 'binary (1 bit)', 'bytes_per_doc': dims / 8,
                 'recall': recall_at_k(fb, gold, k), 'rescored': False})
    wide = search_binary(qb, db, k * oversample)
    fixed = rescore(wide, emb['queries'], emb['docs'], k)
    rows.append({'name': f'binary + rescore top {k * oversample}',
                 'bytes_per_doc': dims / 8 + dims * 4 * k * oversample / world['cfg']['docs'],
                 'recall': recall_at_k(fixed, gold, k), 'rescored': True})
    return rows


# ------------------------------------------- the expensive scorer, and when
def query_likelihood(world, qi, doc_ids, mu=2000.0):
    """An interaction scorer: how likely is this query under this document?

    A bi-encoder compares two summaries.  This one looks at the document's
    own words for every query term, which is why it is better and why it
    cannot be precomputed - the cost is paid per (query, document) pair.
    """
    counts = world['counts']
    coll = counts.sum(axis=0)
    coll = coll / coll.sum()
    q = world['qcounts'][qi]
    terms = np.nonzero(q)[0]
    d = counts[doc_ids][:, terms]
    lens = counts[doc_ids].sum(axis=1, keepdims=True)
    probs = (d + mu * coll[terms]) / (lens + mu)
    return (np.log(probs) * q[terms]).sum(axis=1)


def two_stage(world, emb, k=10, shortlist=(10, 50, 100, 500), full=False):
    """Shortlist with the cheap vector, re-rank the shortlist with the dear one."""
    gold = truth(world)
    rows = []
    base = search(emb['queries'], emb['docs'], k)
    t0 = time.perf_counter()
    search(emb['queries'], emb['docs'], k)
    vec_ms = (time.perf_counter() - t0) * 1e3 / len(gold)
    rows.append({'stage': 'bi-encoder only', 'shortlist': k,
                 'recall': recall_at_k(base, gold, k), 'ms_per_query': vec_ms})
    for n in shortlist:
        cand = search(emb['queries'], emb['docs'], max(n, k))
        t0 = time.perf_counter()
        out = []
        for i in range(len(gold)):
            s = query_likelihood(world, i, cand[i])
            out.append(cand[i][np.argsort(-s)[:k]])
        ms = (time.perf_counter() - t0) * 1e3 / len(gold)
        rows.append({'stage': f'+ rerank top {n}', 'shortlist': n,
                     'recall': recall_at_k(np.array(out), gold, k),
                     'ms_per_query': vec_ms + ms})
    if full:
        allids = np.arange(world['cfg']['docs'])
        t0 = time.perf_counter()
        out = []
        for i in range(len(gold)):
            s = query_likelihood(world, i, allids)
            out.append(allids[np.argsort(-s)[:k]])
        ms = (time.perf_counter() - t0) * 1e3 / len(gold)
        rows.append({'stage': 'rerank everything', 'shortlist': world['cfg']['docs'],
                     'recall': recall_at_k(np.array(out), gold, k), 'ms_per_query': ms})
    return rows


def fusion_table(world, emb, k=10, pool=200, weights=(0.8, 0.2), rrf_k=60):
    """Two scores are not automatically better than one."""
    gold = truth(world)
    cand = search(emb['queries'], emb['docs'], pool)
    q, d = unit(emb['queries']), unit(emb['docs'])
    rows = [{'name': 'vector score only', 'recall': recall_at_k(cand[:, :k], gold, k)}]
    rrf_out, mix_out, lex_out = [], [], []
    for i in range(len(gold)):
        ids = cand[i]
        sem = d[ids] @ q[i]
        lex = query_likelihood(world, i, ids)
        r_sem = np.argsort(np.argsort(-sem))
        r_lex = np.argsort(np.argsort(-lex))
        rrf = 1 / (rrf_k + r_sem) + 1 / (rrf_k + r_lex)
        rrf_out.append(ids[np.argsort(-rrf)[:k]])
        z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
        mix_out.append(ids[np.argsort(-(weights[0] * z(sem) + weights[1] * z(lex)))[:k]])
        lex_out.append(ids[np.argsort(-lex)[:k]])
    rows.append({'name': 'lexical score only (on the same shortlist)',
                 'recall': recall_at_k(np.array(lex_out), gold, k)})
    rows.append({'name': f'reciprocal rank fusion (k={rrf_k})',
                 'recall': recall_at_k(np.array(rrf_out), gold, k)})
    rows.append({'name': f'weighted {weights[0]:.0%} / {weights[1]:.0%}',
                 'recall': recall_at_k(np.array(mix_out), gold, k)})
    return rows


# ------------------------------------------- one vector per document is a bet
LONG = {'topics': 64, 'vocab': 800, 'docs': 4_000, 'queries': 32, 'targets': 10,
        'sections': 5, 'section_len': 200, 'topic_alpha': 0.03, 'query_len': (24, 48),
        'sibling_overlap': 0.5, 'dims': 128}


def make_long_world(seed=1, **kw):
    """Documents that are three unrelated sections glued together.

    This is the shape a single vector cannot represent: the average of three
    directions points at none of them.
    """
    cfg = dict(LONG, **kw)
    rng = np.random.default_rng(seed)
    T, V = cfg['topics'], cfg['vocab']
    topic_word = rng.dirichlet(np.full(V, cfg['topic_alpha']), size=T)
    qtopics = rng.choice(T, size=cfg['queries'], replace=False)
    hot = set(int(t) for t in qtopics)
    cold = [t for t in range(T) if t not in hot]
    # background sections use *siblings* of the query topics: half the same
    # vocabulary, so a whole background document looks vaguely on-topic while
    # a planted document is only on-topic in one of its sections.
    a = cfg['sibling_overlap']
    for i, t in enumerate(qtopics):
        sib = cold[i % len(cold)]
        topic_word[sib] = a * topic_word[t] + (1 - a) * topic_word[sib]
        topic_word[sib] /= topic_word[sib].sum()

    sections, gold = [], []
    n_bg = cfg['docs'] - cfg['queries'] * cfg['targets']
    for _ in range(n_bg):
        sections.append(list(rng.choice(cold, size=cfg['sections'], replace=False)))
    for qi, t in enumerate(qtopics):
        ids = []
        for _ in range(cfg['targets']):
            rest = list(rng.choice(cold, size=cfg['sections'] - 1, replace=False))
            sec = [int(t)] + rest
            rng.shuffle(sec)
            ids.append(len(sections))
            sections.append(sec)
        gold.append(ids)
    order = rng.permutation(len(sections))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    sections = [sections[i] for i in order]
    gold = np.array([[inv[i] for i in ids] for ids in gold])

    S = cfg['sections']
    sec_counts = np.zeros((len(sections), S, V), dtype=np.float32)
    for i, secs in enumerate(sections):
        for j, t in enumerate(secs):
            n = int(np.clip(rng.lognormal(np.log(cfg['section_len']), 0.4), 40, 800))
            sec_counts[i, j] = rng.multinomial(n, topic_word[t])
    qcounts = np.zeros((cfg['queries'], V), dtype=np.float32)
    for qi, t in enumerate(qtopics):
        qcounts[qi] = rng.multinomial(int(rng.integers(*cfg['query_len'])), topic_word[t])
    return {'cfg': cfg, 'sec_counts': sec_counts, 'counts': sec_counts.sum(axis=1),
            'qcounts': qcounts, 'gold': gold, 'topics': sections}


def chunking_table(long_world, k=10, seed=0):
    cfg = long_world['cfg']
    gold = long_world['gold']
    rng = np.random.default_rng(seed)
    train = rng.choice(cfg['docs'], size=min(2000, cfg['docs']), replace=False)
    X, idf = tfidf(long_world['counts'][train])
    _, _, vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    basis = vt[:cfg['dims']].T
    qv, _ = tfidf(long_world['qcounts'], idf)
    qv = unit(qv @ basis)

    whole, _ = tfidf(long_world['counts'], idf)
    whole = unit(whole @ basis)
    doc_found = topk(qv @ whole.T, k)

    S = cfg['sections']
    flat = long_world['sec_counts'].reshape(-1, cfg['vocab'])
    cv, _ = tfidf(flat, idf)
    cv = unit(cv @ basis)
    owner = np.repeat(np.arange(cfg['docs']), S)
    scores = qv @ cv.T                                   # (queries, docs * S)
    best = np.full((len(gold), cfg['docs']), -np.inf, dtype=np.float32)
    for j in range(S):
        best = np.maximum(best, scores[:, j::S])
    chunk_found = topk(best, k)
    return [{'name': 'one vector for the whole document',
             'recall': recall_at_k(doc_found, gold, k),
             'vectors_per_doc': 1, 'bytes_per_doc': cfg['dims'] * 4},
            {'name': f'one vector per section ({S}), score = best section',
             'recall': recall_at_k(chunk_found, gold, k),
             'vectors_per_doc': S, 'bytes_per_doc': cfg['dims'] * 4 * S}]


# ------------------------------------------------------- brute force scaling
def scan_cost(dims=256, sizes=(10_000, 100_000, 400_000), seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    q = unit(rng.standard_normal((1, dims)).astype(dtype))
    rows = []
    for n in sizes:
        d = unit(rng.standard_normal((n, dims)).astype(dtype))
        topk(q @ d.T, 10)                                   # warm up
        t0 = time.perf_counter()
        for _ in range(2):
            topk(q @ d.T, 10)
        ms = (time.perf_counter() - t0) * 1e3 / 2
        rows.append({'docs': n, 'ms_per_query': ms,
                     'gb': d.nbytes / 1024 ** 3,
                     'queries_per_second': 1e3 / ms})
        del d
    return rows


# --------------------------------------------- the two sides are not the same
def asymmetry_table(world, emb, k=10):
    """A query is short and a document is long; only one side needs care."""
    gold = truth(world)
    q, d = emb['queries'], emb['docs']
    rows = [{'name': 'both sides normalised (cosine)',
             'recall': recall_at_k(topk(unit(q) @ unit(d).T, k), gold, k)},
            {'name': 'query left unnormalised',
             'recall': recall_at_k(topk(q @ unit(d).T, k), gold, k)},
            {'name': 'documents left unnormalised',
             'recall': recall_at_k(topk(unit(q) @ d.T, k), gold, k)}]
    raw_q = np.log1p(world['qcounts']) @ emb['basis']          # no idf on the query
    rows.append({'name': 'query encoded without the idf weights',
                 'recall': recall_at_k(topk(unit(raw_q) @ unit(d).T, k), gold, k)})
    wrong, _ = tfidf(world['qcounts'])                         # idf fit on queries
    rows.append({'name': 'query encoded with an idf of its own',
                 'recall': recall_at_k(topk(unit(wrong @ emb['basis']) @ unit(d).T, k),
                                       gold, k)})
    twice, _ = tfidf(world['qcounts'], emb['idf'] ** 2)        # idf applied twice
    rows.append({'name': 'query with the idf applied twice',
                 'recall': recall_at_k(topk(unit(twice @ emb['basis']) @ unit(d).T, k),
                                       gold, k)})
    return rows


def duplicates(world, emb, k=10, copies=3, seed=0):
    """Near-duplicates do not split the top-k; they eat it."""
    rng = np.random.default_rng(seed)
    gold = truth(world)
    docs = emb['docs'].copy()
    victim = int(gold[0][0])
    noise = rng.standard_normal((copies, docs.shape[1])).astype(np.float32) * 1e-3
    dupes = docs[victim] + noise
    docs = np.vstack([docs, dupes])
    found = search(emb['queries'], docs, k)
    dup_ids = set(range(len(emb['docs']), len(docs)))
    taken = float(np.mean([len(set(f) & dup_ids) for f in found[:1]]))
    return {'copies': copies, 'slots_taken_for_that_query': taken,
            'recall_overall': recall_at_k(found, gold, k)}


# ----------------------------------------------------------------- printing
def head(n, title):
    print('\n' + '=' * 72)
    print(f'{n}. {title}')
    print('=' * 72 + '\n')


def main():
    print('Day 65 - embeddings and vector search')
    print('An embedding is a direction; retrieval is "whose direction is closest".')
    print('Everything below measures what each practical decision costs.\n')
    world = make_world()
    emb = train_embedding(world)
    gold = truth(world)
    cfg = world['cfg']

    # ------------------------------------------------------------------ 1
    head(1, 'The short version')
    base = recall_at_k(search(emb['queries'], emb['docs'], 10), gold)
    oracle = recall_at_k(topk(unit(world['qmix']) @ unit(world['mix']).T, 10), gold)
    tf, idf = tfidf(world['counts'])
    qtf, _ = tfidf(world['qcounts'], idf)
    lexical = recall_at_k(topk(unit(qtf) @ unit(tf).T, 10), gold)
    dims = dimension_curve(world, emb)
    best = max(dims, key=lambda r: r['recall'])
    print(f"  {cfg['docs']:,} documents, {cfg['queries']} queries, and for every query")
    print(f"  {cfg['targets']} planted answers, so recall@10 is a fact and not a taste test.\n")
    print(f'    term matching (tf-idf cosine)        recall@10 {lexical:.3f}')
    print(f"    our embedding ({cfg['dims']} dims, 20 lines)  recall@10 {base:.3f}")
    print(f'    the answer key itself                recall@10 {oracle:.3f}  (the ceiling)\n')
    print('  Three numbers from the rest of the file, to set expectations:')
    print(f"    normalising the document vectors is worth "
          f"{base:.3f} against "
          f"{recall_at_k(search(emb['queries'], emb['docs'], 10, False), gold):.3f}")
    print(f"    cutting {cfg['dims']} dimensions down to {best['dims']} gives "
          f"{best['recall']:.3f} at 1/{cfg['dims'] // best['dims']} of the memory")
    print('    int8 costs nothing, one bit per dimension costs most of the recall')

    # ------------------------------------------------------------------ 2
    head(2, 'A vector is a direction, so the length has to go')
    lb = length_bias(world, emb)
    print(f"  Document vector length correlates {lb['norm_vs_length_corr']:.2f} with how many")
    print(f"  words the document has.  The median document is {lb['median_length']:.0f} words.\n")
    print('    scoring                 recall@10   mean length of the hits')
    for name, r in lb['rows'].items():
        print(f"    {name:22} {r['recall']:8.3f}   {r['mean_length_of_hits']:10.0f} words")
    print('\n  A raw dot product does not ask "is this about the same thing", it asks')
    print('  "is this about the same thing *and* is it long".  The long documents win')
    print('  every query, which is why cosine - the dot product of unit vectors - is')
    print('  the default everywhere, and why every library has a normalize flag that')
    print('  people forget exactly once.')

    # ------------------------------------------------------------------ 3
    head(3, 'Where the vectors come from: 20 lines, no neural network')
    print('    counts = how often each word appears in each document')
    print('    X      = log1p(counts) * idf              # down-weight common words')
    print('    U,S,Vt = svd(X - X.mean(0))               # the directions that matter')
    print('    vec    = X @ Vt[:dims].T                  # project into dims')
    print()
    print(f'    tf-idf cosine over {cfg["vocab"]} words : recall@10 {lexical:.3f}')
    print(f'    the same, projected to {cfg["dims"]} dims : recall@10 {base:.3f}')
    print(f'    the answer key                      : recall@10 {oracle:.3f}\n')
    print('  The projection is not compression for its own sake: it is what lets a')
    print('  query match a document that uses different words for the same topic.')
    print('  A trained encoder replaces the SVD with a contrastive objective, but the')
    print('  shape of the object - one dense direction per document - is identical,')
    print('  and so is everything below.')

    # ------------------------------------------------------------------ 4
    head(4, 'The two sides are not symmetric')
    for r in asymmetry_table(world, emb):
        print(f"    {r['name']:42} recall@10 {r['recall']:.3f}")
    rows = asymmetry_table(world, emb)
    spread = max(r['recall'] for r in rows[3:]) - min(r['recall'] for r in rows[3:])
    print('\n  The first fact is exact: **normalising the query cannot change the')
    print('  ranking**, because scaling one side scales every score by the same number.')
    print('  Only the document side matters, which is the whole of section 2.\n')
    print('  The last three are the interesting ones.  Encoding the query differently')
    print('  from the documents - no idf, its own idf, the idf twice - moves the answer')
    print(f'  by up to {spread:.3f} recall **in both directions**, with no error anywhere and')
    print('  no signal that anything changed.  In this corpus the damage is small; what')
    print('  makes it dangerous in practice is that instruction-tuned embedding models')
    print('  are trained with a query prefix and a document prefix, so the two encoders')
    print('  really are different functions - and swapping them is exactly this kind of')
    print('  silent move, with nothing in the logs to say so.')

    # ------------------------------------------------------------------ 5
    head(5, 'How many dimensions do you actually need')
    print('    dims   bytes/doc   GB per million   recall@10')
    for r in dims:
        print(f"    {r['dims']:4}   {r['bytes_per_doc']:9,}   {r['gb_per_million']:14.2f}"
              f"   {r['recall']:9.3f}")
    print(f"\n  The peak is at {best['dims']} dimensions, not at {cfg['dims']}: the tail directions of")
    print('  an SVD are noise, and asking a noisy direction to vote costs recall.  A')
    print('  trained Matryoshka encoder is built so the curve is flat and then falls,')
    print('  instead of rising like this one, but the practical conclusion is the same')
    print('  in both cases: **the dimension count is a hyperparameter, not a virtue**,')
    print('  and the first thing to try on a vector bill is truncation.')

    # ------------------------------------------------------------------ 6
    head(6, 'How few bits per number')
    for label, e in (('full width', emb),
                     (f"truncated to {best['dims']}",
                      {'queries': emb['queries'][:, :best['dims']],
                       'docs': emb['docs'][:, :best['dims']]})):
        print(f'  {label}:')
        print('    representation                bytes/doc   recall@10')
        for r in quantization_table(world, e):
            print(f"    {r['name']:28} {r['bytes_per_doc']:9,.1f}   {r['recall']:9.3f}")
        print()
    print('  int8 is free: one scale for the whole corpus, the dot product survives the')
    print('  rounding, four times less memory.  One bit per dimension is not free here,')
    print('  because this embedding puts most of its information in the first few')
    print('  dimensions and a sign bit treats all of them as equals.  Rescoring the')
    print('  shortlist with the full vectors buys a lot of it back, which is the')
    print('  standard recipe: search in the cheap representation, decide in the dear one.')

    # ------------------------------------------------------------------ 7
    head(7, 'A second score is not automatically better')
    fus = fusion_table(world, emb)
    for r in fus:
        print(f"    {r['name']:44} recall@10 {r['recall']:.3f}")
    only, lex, rrf, mix = (r['recall'] for r in fus)
    print('\n  The lexical score here is genuinely weaker - these queries ask for two')
    print('  topics at once and word overlap cannot see that.  Fusing it in by rank')
    print('  (reciprocal rank fusion, which treats both lists as equally trustworthy)')
    print(f'  costs {only - rrf:.3f} recall; weighting it down to 20% is a wash ({mix - only:+.3f}).')
    print('  The rule this leaves: a second signal helps when it is right where the')
    print('  first is wrong, and the weights are a measurement, not a default.')

    # ------------------------------------------------------------------ 8
    head(8, 'One vector per document is a bet about the document')
    lw = make_long_world()
    rows = chunking_table(lw)
    print(f"  {lw['cfg']['docs']:,} documents, each {lw['cfg']['sections']} sections about different things.")
    print(f"  The answer sits in one section of {lw['cfg']['sections']}; the other documents are")
    print('  vaguely on-topic all the way through.\n')
    print('    representation                                       vectors/doc  recall@10')
    for r in rows:
        print(f"    {r['name']:52} {r['vectors_per_doc']:6}   {r['recall']:9.3f}")
    print(f"\n  {rows[1]['recall'] / max(rows[0]['recall'], 1e-9):.1f}x the recall for "
          f"{rows[1]['vectors_per_doc']}x the vectors.  The average of five directions")
    print('  points at none of them, so a single vector for a long mixed document is')
    print('  a claim that the document is about one thing.  Chunking is how everybody')
    print('  fixes this; late-interaction models (one vector per token or per span,')
    print('  scored by the best match) are the same idea taken to its limit.')

    # ------------------------------------------------------------------ 9
    head(9, 'What brute force costs')
    print('    documents    memory      ms per query   queries/s')
    for r in scan_cost():
        print(f"    {r['docs']:9,}   {r['gb']:6.2f} GB   {r['ms_per_query']:12.2f}"
              f"   {r['queries_per_second']:9.0f}")
    print('\n  A full scan is a single matrix multiply, so it is fast enough to be the')
    print('  right answer for a long time - at ten thousand documents it is a quarter')
    print('  of a millisecond.  It is also perfectly linear, which is the problem: the')
    print('  same code at ten million documents is seconds per query, and no amount of')
    print('  vectorisation fixes a linear scan.  That is the gap approximate nearest')
    print('  neighbour indexes exist to close, by refusing to look at most of the data.')

    # ----------------------------------------------------------------- 10
    head(10, 'Five ways this goes wrong quietly')
    lbr = lb['rows']['raw dot product']
    print(f"  1. Forgetting to normalise.  Recall {lb['rows']['cosine (normalised)']['recall']:.3f} -> "
          f"{lbr['recall']:.3f}, and the hits are")
    print(f"     {lbr['mean_length_of_hits']:.0f} words long against a median of {lb['median_length']:.0f}: "
          f"the index now ranks by length.\n")
    print('  2. Quantizing per vector instead of per corpus.  A scale computed from')
    print('     each vector on its own is a different unit for every document, so the')
    print('     scores stop being comparable - the thing the whole index is for.\n')
    d = duplicates(world, emb)
    print(f"  3. Near-duplicates.  Adding {d['copies']} copies of one answer costs that query")
    print(f"     {d['slots_taken_for_that_query']:.0f} of its 10 slots; the copies do not split the vote, they")
    print('     take the seats.  Deduplicate before indexing, not after retrieving.\n')
    asym = asymmetry_table(world, emb)
    worst = min(asym[3:], key=lambda r: r['recall'])
    print('  4. Encoding queries and documents differently by accident:')
    print(f"     {worst['name']} gives {worst['recall']:.3f} against {base:.3f}, and the")
    print('     variants that happen to help are just as silent as the ones that hurt.\n')
    print('  5. Reporting cosine similarity as if it meant relevance.  It measures')
    print('     "about the same things", and the planted answers here prove the gap:')
    print(f"     even the answer key only reaches {oracle:.3f}, because documents about the")
    print('     right topic are not the same thing as documents that answer the question.')

    # ----------------------------------------------------------------- 11
    head(11, 'In sentence-transformers')
    print('    model = SentenceTransformer("...", truncate_dim=256)   # Matryoshka')
    print('    doc_vecs = model.encode(docs, normalize_embeddings=True,')
    print('                            prompt_name="document", batch_size=256)')
    print('    q_vecs   = model.encode(queries, normalize_embeddings=True,')
    print('                            prompt_name="query")')
    print('    from sentence_transformers.quantization import quantize_embeddings')
    print('    small = quantize_embeddings(doc_vecs, precision="int8",')
    print('                                calibration_embeddings=doc_vecs[:50_000])\n')
    print('  Three of the arguments on that page are the three sections above:')
    print('  normalize_embeddings is section 2, truncate_dim is section 5, and')
    print('  calibration_embeddings is the per-corpus scale from section 6.  The')
    print('  prompt names are section 4 - the model was trained with a query prefix')
    print('  and a document prefix, and swapping them is silent.')

    # ----------------------------------------------------------------- 12
    head(12, 'In a vector store')
    print('    index = faiss.IndexFlatIP(dims)        # exactly section 9')
    print('    faiss.normalize_L2(doc_vecs)           # ...so inner product = cosine')
    print('    index.add(doc_vecs)')
    print('    scores, ids = index.search(q_vecs, 10)\n')
    print('  Every managed vector database is this plus three decisions you now have')
    print('  numbers for: how many dimensions to store, how many bits per number, and')
    print('  whether one vector per document is enough.  The index type - which is how')
    print('  you avoid the linear scan - is the only part this day did not build.')

    # ----------------------------------------------------------------- 13
    head(13, 'Assertions')
    check()
    print('all assertions passed')


def check():
    world = make_world(seed=2, docs=8_000, queries=100)
    emb = train_embedding(world)
    gold = truth(world)

    # the planted answers are findable, and the answer key is the ceiling
    base = recall_at_k(search(emb['queries'], emb['docs'], 10), gold)
    oracle = recall_at_k(topk(unit(world['qmix']) @ unit(world['mix']).T, 10), gold)
    assert 0.3 < base < oracle <= 1.0
    assert gold.shape == (100, world['cfg']['targets'])

    # normalising the query cannot change the ranking; normalising documents can
    a = topk(unit(emb['queries']) @ unit(emb['docs']).T, 10)
    b = topk(emb['queries'] @ unit(emb['docs']).T, 10)
    assert (a == b).all()
    assert recall_at_k(search(emb['queries'], emb['docs'], 10, False), gold) < base / 2

    # vector length tracks document length
    lb = length_bias(world, emb)
    assert lb['norm_vs_length_corr'] > 0.8
    assert lb['rows']['raw dot product']['mean_length_of_hits'] > 3 * lb['median_length']

    # truncation: the tail of an SVD is not free information
    curve = dimension_curve(world, emb, dims=(256, 64, 32, 8))
    best = max(curve, key=lambda r: r['recall'])
    assert best['dims'] < 256
    assert curve[-1]['recall'] < best['recall']          # too few also hurts

    # int8 with a corpus-wide scale is free; one bit per dimension is not
    rows = {r['name']: r for r in quantization_table(world, emb)}
    assert abs(rows['int8']['recall'] - rows['float32']['recall']) < 0.02
    assert rows['int8']['bytes_per_doc'] == rows['float32']['bytes_per_doc'] / 4
    assert rows['binary (1 bit)']['recall'] < rows['float32']['recall'] / 2
    rescored = [r for r in rows.values() if r['rescored']][0]
    assert rescored['recall'] > rows['binary (1 bit)']['recall']

    # fusing a weaker signal by rank costs recall
    fus = {r['name']: r['recall'] for r in fusion_table(world, emb, pool=100)}
    assert fus['lexical score only (on the same shortlist)'] < fus['vector score only']
    assert fus['reciprocal rank fusion (k=60)'] < fus['vector score only']

    # one vector per document loses the section that matters
    rows = chunking_table(make_long_world(seed=3))
    assert rows[1]['recall'] > 3 * rows[0]['recall']
    assert rows[1]['bytes_per_doc'] == rows[0]['bytes_per_doc'] * rows[1]['vectors_per_doc']

    # the scan is linear in the number of documents
    sc = scan_cost(sizes=(20_000, 200_000))
    assert sc[1]['ms_per_query'] > 4 * sc[0]['ms_per_query']


if __name__ == '__main__':
    main()
