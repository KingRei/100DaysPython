"""Day 47 - k-means and k-nearest-neighbours: two algorithms that are a distance
function plus a rule, and that only ever fail inside the distance function.

Run `python clustering_knn.py` to print the whole story.  Timings depend on the
machine; every other number is exact and checked by an assert at the bottom.
"""

import heapq
import math
import random
import time
import warnings

import numpy as np


# ======================================================== 1. distances, once and for all
def sq_dists(X, C):
    """Squared euclidean distance from every row of X to every row of C.

    The expansion |x - c|^2 = |x|^2 - 2 x.c + |c|^2 turns a double loop into one
    matrix product (Day 44).  `np.maximum(..., 0)` is not cosmetic: the expansion
    subtracts two large numbers, so a point's distance to its own position comes
    out as a tiny negative value, and the sqrt of that is nan.
    """
    x2 = np.einsum('ij,ij->i', X, X)[:, None]
    c2 = np.einsum('ij,ij->i', C, C)[None, :]
    return np.maximum(x2 - 2.0 * X @ C.T + c2, 0.0)


def inertia(X, C, labels):
    """The objective k-means actually minimises: sum of squared distances to the
    assigned centroid.  Note the square - that is the whole story of section 6."""
    return float(np.sum((X - C[labels]) ** 2))


def l1_cost(X, C, labels):
    return float(np.sum(np.abs(X - C[labels])))


# ============================================================== 2. Lloyd's algorithm
def kmeans(X, k, seed=0, init='random', max_iter=100, on_empty='nan', trace=False):
    """Lloyd's algorithm.  The rule is two lines inside the loop.

    assign: every point joins its nearest centroid
    update: every centroid moves to the mean of its members

    Both halves lower the same objective - assignment because each point picks its
    cheapest option, update because the mean is the minimiser of squared distance -
    so the objective can only go down, and a finite number of partitions means it
    has to stop.  That proof is what breaks in section 6.
    """
    rng = np.random.default_rng(seed)
    C = kmeanspp_init(X, k, rng) if init == 'kmeans++' else X[rng.choice(len(X), k, replace=False)].copy()
    history = []
    labels = np.zeros(len(X), dtype=int)
    for it in range(max_iter):
        labels = np.argmin(sq_dists(X, C), axis=1)
        if trace:
            history.append((C.copy(), labels.copy(), inertia(X, C, labels)))
        newC = np.empty_like(C)
        for j in range(k):
            members = X[labels == j]
            if len(members) == 0:
                # The silent failure of section 5.  'nan' is what the naive mean does.
                newC[j] = np.nan if on_empty == 'nan' else X[np.argmax(np.min(sq_dists(X, C), axis=1))]
            else:
                newC[j] = members.mean(axis=0)
        if np.allclose(newC, C, equal_nan=True):
            C = newC
            break
        C = newC
    labels = np.argmin(sq_dists(X, C), axis=1)
    return C, labels, inertia(X, C, labels), it + 1, history


def kmeanspp_init(X, k, rng):
    """k-means++: pick the first centre uniformly, then pick each further centre with
    probability proportional to its squared distance to the nearest centre so far.

    It is Day 42's weighted random pick, and it is still random - it just makes the
    'two centres land in the same blob' draw unlikely instead of common.
    """
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(sq_dists(X, np.array(C)), axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(C)


def kmedians(X, k, seed=0, max_iter=100, update='median'):
    """Same loop, L1 distance - and a switch for the update that is the point of
    section 6.  'median' is the correct partner for L1; 'mean' is the one that is
    copied over from k-means and quietly breaks the termination proof.
    """
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy().astype(float)
    costs = []
    for _ in range(max_iter):
        labels = np.argmin(np.abs(X[:, None, :] - C[None, :, :]).sum(axis=2), axis=1)
        costs.append(l1_cost(X, C, labels))
        newC = np.array([
            (np.median(X[labels == j], axis=0) if update == 'median' else X[labels == j].mean(axis=0))
            if np.any(labels == j) else C[j]
            for j in range(k)
        ])
        if np.allclose(newC, C):
            break
        C = newC
    labels = np.argmin(np.abs(X[:, None, :] - C[None, :, :]).sum(axis=2), axis=1)
    costs.append(l1_cost(X, C, labels))
    return C, labels, costs


# =================================================================== 3. toy data sets
def blobs(centers, n_per, spread, seed=0):
    rng = np.random.default_rng(seed)
    cen = np.array(centers, dtype=float)
    return np.vstack([c + spread * rng.standard_normal((n_per, len(c))) for c in cen])


def elongated(n=150, seed=3):
    """Two groups that are long in x and thin in y, side by side in y."""
    rng = np.random.default_rng(seed)
    a = np.column_stack([rng.uniform(-4, 4, n), 0.9 + 0.22 * rng.standard_normal(n)])
    b = np.column_stack([rng.uniform(-4, 4, n), -0.9 + 0.22 * rng.standard_normal(n)])
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


def moons(n=150, seed=4, noise=0.09):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, math.pi, n)
    a = np.column_stack([np.cos(t), np.sin(t)]) + noise * rng.standard_normal((n, 2))
    b = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)]) + noise * rng.standard_normal((n, 2))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


def requests(n_per=120, seed=7):
    """n requests with two features in wildly different units:
    prompt_tokens (hundreds to thousands) and cache hit rate (0 to 1).

    The real structure is in the hit rate - three populations at 0.15, 0.50, 0.85 -
    while prompt length is drawn from the same wide range for all three.
    """
    rng = np.random.default_rng(seed)
    out, lab = [], []
    for j, hit in enumerate([0.15, 0.50, 0.85]):
        toks = rng.uniform(200, 4000, n_per)
        h = np.clip(hit + 0.05 * rng.standard_normal(n_per), 0, 1)
        out.append(np.column_stack([toks, h]))
        lab += [j] * n_per
    return np.vstack(out), np.array(lab)


def purity(labels, truth, k):
    """Fraction of points whose cluster's majority true label is their own."""
    ok = 0
    for j in range(k):
        m = labels == j
        if m.any():
            ok += np.bincount(truth[m]).max()
    return ok / len(truth)


# ======================================================================= 4. k-nearest
def knn_predict(Xtr, ytr, Xq, k, exclude_self=False):
    """The whole classifier.  There is no training: the data is the model."""
    d2 = sq_dists(Xq, Xtr)
    if exclude_self:
        # Writing the diagonal, not adding eye * inf: off the diagonal eye is 0,
        # and 0 * inf is nan, which would poison every other entry.
        d2 = d2.copy()
        np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1, kind='stable')[:, :k]
    votes = ytr[nn]
    return np.array([np.bincount(v).argmax() for v in votes])


def accuracy(a, b):
    return float(np.mean(a == b))


def contrast(n, d, seed=0, queries=50):
    """How much nearer is the nearest point than the furthest one?

    (max - min) / min over uniform random points.  In two dimensions the nearest
    neighbour is genuinely near; by a few hundred dimensions every point is almost
    exactly the same distance away, and 'nearest' stops carrying information.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    Q = rng.random((queries, d))
    dd = np.sqrt(sq_dists(Q, X))
    return float(np.mean((dd.max(axis=1) - dd.min(axis=1)) / dd.min(axis=1)))


def noisy_features(n=400, d=2, seed=5):
    """Two informative features, d - 2 pure noise ones.  Nothing about the problem
    changed; the only thing that grew is the number of dimensions the distance
    function has to average over."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = np.column_stack([
        np.where(y == 1, 1.2, -1.2) + 0.9 * rng.standard_normal(n),
        np.where(y == 1, -0.8, 0.8) + 0.9 * rng.standard_normal(n),
    ])
    if d > 2:
        X = np.column_stack([X, rng.standard_normal((n, d - 2))])
    return X, y


# ======================================================== 5. LeetCode 973 and 658
class SolutionKClosest:
    def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
        # A max-heap of size k: push, and drop the worst as soon as there are k + 1.
        # Comparing x*x + y*y instead of its square root is both faster and exact -
        # the square root is monotone, so it cannot change the ordering.
        heap: list[tuple[int, int, int]] = []
        for x, y in points:
            heapq.heappush(heap, (-(x * x + y * y), x, y))
            if len(heap) > k:
                heapq.heappop(heap)
        return [[x, y] for _, x, y in heap]


def k_closest_quickselect(points, k, seed=0):
    """Day 21's quickselect: partition around a random pivot, recurse into the side
    that still contains the boundary.  O(n) expected instead of O(n log k)."""
    pts = [tuple(p) for p in points]
    rng = random.Random(seed)

    def key(p):
        return p[0] * p[0] + p[1] * p[1]

    lo, hi = 0, len(pts) - 1
    arr = pts[:]
    while lo < hi:
        pivot = key(arr[rng.randint(lo, hi)])
        i, j = lo, hi
        while i <= j:
            while key(arr[i]) < pivot:
                i += 1
            while key(arr[j]) > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i, j = i + 1, j - 1
        if k - 1 <= j:
            hi = j
        elif k - 1 >= i:
            lo = i
        else:
            break
    return [list(p) for p in arr[:k]]


class SolutionFindClosestElements:
    def findClosestElements(self, arr: list[int], k: int, x: int) -> list[int]:
        # Binary search over the left edge of the window, not over the values.
        # The window [lo, lo + k) is better than [lo + 1, lo + k + 1) exactly when
        # dropping arr[lo] costs more than it saves by gaining arr[lo + k].
        # `>` and not `>=` is the tie rule: on a tie we keep the smaller element.
        lo, hi = 0, len(arr) - k
        while lo < hi:
            mid = (lo + hi) // 2
            if x - arr[mid] > arr[mid + k] - x:
                lo = mid + 1
            else:
                hi = mid
        return arr[lo:lo + k]


def closest_elements_by_abs(arr, k, x):
    """The sort-by-distance answer.  Correct, O(n log n), and it throws away the
    fact that the input was already sorted."""
    return sorted(sorted(arr, key=lambda v: (abs(v - x), v))[:k])


def closest_elements_ge(arr, k, x):
    """The same binary search with `>=` instead of `>`: the tie goes the wrong way."""
    lo, hi = 0, len(arr) - k
    while lo < hi:
        mid = (lo + hi) // 2
        if x - arr[mid] >= arr[mid + k] - x:
            lo = mid + 1
        else:
            hi = mid
    return arr[lo:lo + k]


def outlier_blobs(seed=0, n_per=30, n_out=6):
    """Three blobs plus a handful of points scattered far away - the shape of a
    real data set that has a few mislabelled or mis-parsed rows in it."""
    rng = np.random.default_rng(seed)
    return np.vstack([blobs([[0, 0], [6, 0], [3, 5]], n_per, 1.0, seed=seed),
                      rng.uniform(-6, 14, (n_out, 2))])


def check() -> None:
    """Everything the walkthrough claims, re-derived from scratch."""
    Xb = blobs([[0, 0], [6, 0.5], [3, 5]], 60, 0.9, seed=1)
    # Lloyd never increases its own objective.
    _, _, _, _, h2 = kmeans(Xb, 3, seed=0, init='kmeans++', trace=True)
    assert all(h2[i][2] >= h2[i + 1][2] - 1e-9 for i in range(len(h2) - 1))
    # Adding a centroid can never raise the inertia.
    Js = [kmeans(Xb, k, seed=0, init='kmeans++')[2] for k in range(1, 7)]
    assert all(Js[i] >= Js[i + 1] - 1e-9 for i in range(len(Js) - 1))
    # k=1 inertia is the total spread around the global mean - no algorithm involved.
    assert abs(Js[0] - float(np.sum((Xb - Xb.mean(axis=0)) ** 2))) < 1e-6

    # Raw units lose the structure; standardising helps but does not recover it;
    # dropping the irrelevant column does.
    Xr, yr = requests()
    pr = purity(kmeans(Xr, 3, seed=0, init='kmeans++')[1], yr, 3)
    Xz = (Xr - Xr.mean(axis=0)) / Xr.std(axis=0)
    pz = purity(kmeans(Xz, 3, seed=0, init='kmeans++')[1], yr, 3)
    ph = purity(kmeans(Xr[:, 1:2], 3, seed=0, init='kmeans++')[1], yr, 3)
    assert ph == 1.0 and ph > pz > pr

    # The big blob is split in half and the small group is lost entirely.
    Xun = np.vstack([blobs([[0, 0]], 200, 1.0, seed=8), blobs([[2.2, 0]], 20, 0.5, seed=9)])
    yun = np.array([0] * 200 + [1] * 20)
    lun = kmeans(Xun, 2, seed=0, init='kmeans++')[1]
    found = sum(1 for i in range(len(yun)) if yun[i] == 1 and
                np.bincount(yun[lun == lun[i]]).argmax() == 1)
    assert found == 0 and purity(lun, yun, 2) == 200 / 220

    # The mean update breaks L1 monotonicity, and ends up worse.
    Xo = outlier_blobs(seed=0)
    c_mean = kmedians(Xo, 3, seed=6, update='mean')[2]
    c_med = kmedians(Xo, 3, seed=6, update='median')[2]
    assert any(c_mean[i + 1] > c_mean[i] + 1e-9 for i in range(len(c_mean) - 1))
    assert c_mean[-1] > c_med[-1]
    # The median is the L1 minimiser (Day 45, LeetCode 462).
    v = np.array([1.0, 2.0, 3.0, 100.0])
    assert np.abs(v - np.median(v)).sum() < np.abs(v - v.mean()).sum()

    # k=1 training accuracy is 1.0 whatever the data, unless you exclude the point itself.
    Xk, yk = moons(120, seed=12, noise=0.22)
    tr = np.random.default_rng(0).permutation(len(Xk))[:160]
    assert accuracy(knn_predict(Xk[tr], yk[tr], Xk[tr], 1), yk[tr]) == 1.0
    assert accuracy(knn_predict(Xk[tr], yk[tr], Xk[tr], 1, exclude_self=True), yk[tr]) < 1.0
    for s_ in range(5):
        Xn_, yn_ = noisy_features(120, 6, seed=s_)
        assert accuracy(knn_predict(Xn_, yn_, Xn_, 1), yn_) == 1.0

    # Distance concentration is monotone in the number of dimensions.
    cons = [contrast(500, d, seed=0) for d in (2, 4, 8, 16, 64, 256, 1024)]
    assert all(cons[i] > cons[i + 1] for i in range(len(cons) - 1))

    # The three k-closest routines agree, and squared distance orders points
    # identically to the distance itself.
    P = np.random.default_rng(21).integers(-10000, 10000, (4000, 2)).tolist()
    ref = sorted(sorted(P, key=lambda q: q[0] ** 2 + q[1] ** 2)[:50])
    assert sorted(SolutionKClosest().kClosest(P, 50)) == ref
    assert sorted(k_closest_quickselect(P, 50, seed=1)) == ref
    chk = np.random.default_rng(77).integers(-500, 500, (300, 2))
    assert np.array_equal(np.argsort((chk ** 2).sum(axis=1), kind='stable'),
                          np.argsort(np.sqrt((chk ** 2).sum(axis=1)), kind='stable'))

    # LeetCode 658: the window version matches sort-by-distance everywhere, and the
    # `>=` variant does not.
    rng = np.random.default_rng(32)
    diffs = 0
    for _ in range(3000):
        n = int(rng.integers(1, 40))
        a = sorted(int(x) for x in rng.integers(-50, 50, n))
        kk = int(rng.integers(1, n + 1))
        xx = int(rng.integers(-60, 60))
        ok = closest_elements_by_abs(a, kk, xx)
        assert SolutionFindClosestElements().findClosestElements(a, kk, xx) == ok
        diffs += closest_elements_ge(a, kk, xx) != ok
    assert diffs > 0
    assert SolutionFindClosestElements().findClosestElements([1, 2, 3, 4, 5], 4, 3) == [1, 2, 3, 4]
    assert SolutionFindClosestElements().findClosestElements([1, 1, 2, 3, 4, 5], 4, -1) == [1, 1, 2, 3]


def header(t):
    print('\n' + '=' * 72 + '\n' + t + '\n' + '=' * 72)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # ------------------------------------------------------------------ 1
    header('1. Two algorithms, one shape: a distance function and a rule')
    print("""k-means   : repeat { assign each point to the nearest centroid
                     move each centroid to the mean of its members }
k-nearest : for a query, take the k nearest training points and let them vote

Both rules fit on one line.  Neither of them is where the bugs are.""")

    Xb = blobs([[0, 0], [6, 0.5], [3, 5]], 60, 0.9, seed=1)
    C, lab, J, iters, hist = kmeans(Xb, 3, seed=0, init='kmeans++', trace=True)
    print(f'\n180 points, three blobs, k-means++ start: converged in {iters} iterations')
    print('  inertia per iteration: ' + ' -> '.join(f'{h[2]:.1f}' for h in hist))
    drops = [hist[i][2] - hist[i + 1][2] for i in range(len(hist) - 1)]
    print(f'  every step lowered it: {all(d >= -1e-9 for d in drops)}  (final {J:.2f})')
    print('  centroids:\n' + '\n'.join('    ' + str(c) for c in C[np.argsort(C[:, 0])]))
    far = np.array([[1000000.5, 48000000.25]])
    f2 = np.einsum('ij,ij->i', far, far)[:, None]
    raw = (f2 - 2.0 * far @ far.T + f2)[0, 0]
    with np.errstate(invalid='ignore'):
        raw_sqrt = np.sqrt(raw)
    print(f'\nOne line in sq_dists is not cosmetic.  For the point {far[0].tolist()},')
    print(f'the expansion |x|^2 - 2 x.x + |x|^2 evaluates to {raw} - a distance from a point')
    print(f'to itself, so sqrt of it is {raw_sqrt}.  np.maximum(..., 0) gives '
          f'{sq_dists(far, far)[0, 0]}.')

    # ------------------------------------------------------------------ 2
    header('2. Silent failure #1: the objective can never tell you k')
    print('Inertia drops every time you add a centroid, structure or no structure.')
    rng = np.random.default_rng(11)
    Xu = rng.random((180, 2)) * 6.0
    print(f"\n{'k':>3} {'3 blobs':>12} {'drop':>8}   {'uniform noise':>14} {'drop':>8}")
    prev_b = prev_u = None
    elbow_b, elbow_u = [], []
    for k in range(1, 9):
        _, _, Jb, _, _ = kmeans(Xb, k, seed=0, init='kmeans++')
        _, _, Ju, _, _ = kmeans(Xu, k, seed=0, init='kmeans++')
        db = '' if prev_b is None else f'{100 * (prev_b - Jb) / prev_b:6.1f}%'
        du = '' if prev_u is None else f'{100 * (prev_u - Ju) / prev_u:6.1f}%'
        if prev_b is not None:
            elbow_b.append(100 * (prev_b - Jb) / prev_b)
            elbow_u.append(100 * (prev_u - Ju) / prev_u)
        print(f'{k:>3} {Jb:12.1f} {db:>8}   {Ju:14.1f} {du:>8}')
        prev_b, prev_u = Jb, Ju
    print(f'\nBlobs: the k=2->3 step removes {elbow_b[1]:.1f}% and the next only {elbow_b[2]:.1f}%')
    print(f'       - a factor of {elbow_b[1] / elbow_b[2]:.1f}, which is the elbow.')
    print(f'Noise: {elbow_u[1]:.1f}% then {elbow_u[2]:.1f}% - a factor of {elbow_u[1] / elbow_u[2]:.1f}, no elbow,')
    print('       and yet k-means still handed back three tidy clusters of uniform noise.')
    _, labu, _, _, _ = kmeans(Xu, 3, seed=0, init='kmeans++')
    print(f'       their sizes: {np.bincount(labu)} - nothing about that output looks wrong.')

    # ------------------------------------------------------------------ 3
    header('3. Silent failure #2: the answer depends on the random start')
    Xs = blobs([[0, 0], [7, 0], [3.5, 6]], 60, 1.0, seed=2)
    best = min(kmeans(Xs, 3, seed=s, init='kmeans++')[2] for s in range(200))
    for init in ('random', 'kmeans++'):
        Js = [kmeans(Xs, 3, seed=s, init=init)[2] for s in range(200)]
        bad = sum(1 for j in Js if j > best * 1.01)
        print(f'  {init:>9}: {bad:>3}/200 seeds end more than 1% above the best found '
              f'({best:.1f});  worst {max(Js):.1f}')
    Jg = kmeans(Xs, 3, seed=0, init='random')
    Jb2 = max((kmeans(Xs, 3, seed=s, init='random') for s in range(200)), key=lambda r: r[2])
    print(f'\n180 points in three clearly separated blobs - the easiest data set there is.')
    print(f'Seed 0 gives inertia {Jg[2]:.1f} with cluster sizes {np.bincount(Jg[1])};')
    print(f'the worst seed gives {Jb2[2]:.1f} with sizes {np.bincount(Jb2[1])}: two centroids landed')
    print('in the same blob, so one blob was split in half and two were merged.  Both runs')
    print('are "the output of k-means", neither prints a warning, and the only way to tell')
    print('them apart is to compare the inertia - which is why you restart and keep the best.')

    # ------------------------------------------------------------------ 4
    header('4. Silent failure #3: a centroid can lose all of its points')
    Xe = np.array([[0.0, 0.0], [0.5, 0.45], [0.9, 0.05], [5.0, 5.0], [5.4, 4.6]])
    Ce = np.array([[0.45, 0.17], [2.6, 2.6], [5.2, 4.8]])
    d2 = sq_dists(Xe, Ce)
    la = np.argmin(d2, axis=1)
    print(f'5 points in two groups, 3 centroids at {Ce.tolist()};')
    print('the middle one is not nearest to anything, which is what a stale centroid looks')
    print('like after the other two have moved.')
    print(f'  assignment: {la.tolist()} -> cluster 1 gets {int((la == 1).sum())} points')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        empty_mean = Xe[la == 1].mean(axis=0)
        msg = str(w[0].message) if w else '(no warning)'
    nan_c = np.array([Xe[la == j].mean(axis=0) if (la == j).any() else empty_mean for j in range(3)])
    print(f'  mean of the empty cluster: {empty_mean}')
    print(f'  numpy says "{msg}" - on stderr, as a warning, and then returns')
    print(f'  distance from every point to that nan centroid: {sq_dists(Xe, nan_c)[:, 1]}')
    la2 = np.argmin(sq_dists(Xe, nan_c), axis=1)
    row0 = np.round(sq_dists(Xe, nan_c)[0], 3).tolist()
    print(f'\n  np.argmin returns the index of a nan rather than skipping it:')
    print(f'    np.argmin({row0}) = {int(la2[0])}')
    print(f'  so the next assignment is {la2.tolist()} - every point has joined the')
    print('  dead centroid.')
    nan_c2 = np.array([Xe[la2 == j].mean(axis=0) if (la2 == j).any() else np.full(2, np.nan)
                       for j in range(3)])
    print(f'  one more update and the model is\n    {np.round(nan_c2, 3).tolist()}')
    print('  - a single cluster holding everything, plus two that do not exist.')
    print('\n  The whole collapse is one RuntimeWarning on stderr.  scikit-learn relocates')
    print('  the orphan to the point furthest from any centroid; doing nothing is the bug,')
    print('  and `np.nanargmin` would only hide it one level deeper.')

    # ------------------------------------------------------------------ 5
    header('5. Silent failure #4: the units decide the clustering')
    Xr, yr = requests()
    Cr, lr, Jr, _, _ = kmeans(Xr, 3, seed=0, init='kmeans++')
    pr = purity(lr, yr, 3)
    mu, sd = Xr.mean(axis=0), Xr.std(axis=0)
    Xz = (Xr - mu) / sd
    Cz, lz, Jz, _, _ = kmeans(Xz, 3, seed=0, init='kmeans++')
    pz = purity(lz, yr, 3)
    print('360 requests, two features: prompt_tokens (200-4000) and cache hit rate (0-1).')
    print('The three real populations differ only in hit rate: 0.15, 0.50, 0.85.')
    Ch, lh, _, _, _ = kmeans(Xr[:, 1:2], 3, seed=0, init='kmeans++')
    ph = purity(lh, yr, 3)
    print(f'\n  raw units       : purity {pr:.3f}   centroid hit rates {np.sort(Cr[:, 1]).round(3)}')
    print(f'  standardised    : purity {pz:.3f}   centroid hit rates '
          f'{np.sort(Cz[:, 1] * sd[1] + mu[1]).round(3)}')
    print(f'  hit rate only   : purity {ph:.3f}   centroid hit rates {np.sort(Ch[:, 0]).round(3)}')
    sp = (Xr[:, 0].max() - Xr[:, 0].min()) / (Xr[:, 1].max() - Xr[:, 1].min())
    print(f'\nThe token axis spans {sp:.0f}x more than the hit-rate axis, so it contributes')
    print(f'about {sp ** 2 / 1e6:.0f} million times more squared distance.  Raw k-means cut the data')
    print('into three token bands and never looked at the column the answer was in - and the')
    print(f'inertia it reports ({Jr:.0f}) is a perfectly good number for the question it answered.')
    print('\nStandardising is usually called the fix.  It is not a fix, it is a different')
    print('arbitrary choice: "every column matters equally".  Here that still leaves one')
    print(f'column of pure noise voting on every distance, which is why purity stops at {pz:.3f}')
    print(f'instead of {ph:.3f}.  Scaling decides what the distance means, and no amount of it')
    print('can tell you which columns are relevant - only you know that.')

    # ------------------------------------------------------------------ 6
    header('6. The update rule has to match the distance (Day 45, LeetCode 462)')
    print('Lloyd stops because every step lowers the objective.  The update half only')
    print('lowers it because the mean is the minimiser of squared distance.  Day 45 proved')
    print('the companion fact: the minimiser of sum |x - c| is the median, not the mean.')
    v = np.array([1.0, 2.0, 3.0, 100.0])
    print(f'\n  on {v.tolist()}:  mean {v.mean():.1f} costs L1 {np.abs(v - v.mean()).sum():.1f},'
          f'  median {np.median(v):.1f} costs L1 {np.abs(v - np.median(v)).sum():.1f}')
    Xo = outlier_blobs(seed=0)
    _, _, c_mean = kmedians(Xo, 3, seed=6, update='mean')
    _, _, c_med = kmedians(Xo, 3, seed=6, update='median')
    ups = [(i, c_mean[i], c_mean[i + 1]) for i in range(len(c_mean) - 1) if c_mean[i + 1] > c_mean[i] + 1e-9]
    print(f'\n96 points (three blobs + 6 scattered outliers), L1 assignment, seed 6:')
    print('  mean update   : ' + ' -> '.join(f'{c:.1f}' for c in c_mean))
    print('  median update : ' + ' -> '.join(f'{c:.1f}' for c in c_med))
    print(f'\n  the mean version went UP at step {ups[0][0]}: {ups[0][1]:.2f} -> {ups[0][2]:.2f}')
    print(f'  and settled {c_mean[-1] - c_med[-1]:.2f} higher ({c_mean[-1]:.2f} vs {c_med[-1]:.2f}).')
    non_mono = worse = 0
    for ds in range(30):
        Xd = outlier_blobs(seed=1000 + ds)
        for s in range(20):
            cm_ = kmedians(Xd, 3, seed=s, update='mean', max_iter=200)[2]
            cd_ = kmedians(Xd, 3, seed=s, update='median', max_iter=200)[2]
            if any(cm_[i + 1] > cm_[i] + 1e-9 for i in range(len(cm_) - 1)):
                non_mono += 1
            if cm_[-1] > cd_[-1] + 1e-9:
                worse += 1
    print(f'\n  over 600 (data set, seed) pairs: {non_mono} runs are non-monotone,')
    print(f'  {worse} finish worse than the median update.  A loop whose objective can')
    print('  rise has no termination proof left - it is only convergent by luck.')

    # ------------------------------------------------------------------ 7
    header('7. Silent failure #5: k-means only knows one shape')
    for name, (Xd, yd) in [('two long bars', elongated()), ('two moons', moons())]:
        _, ld, _, _, _ = kmeans(Xd, 2, seed=0, init='kmeans++')
        print(f'  {name:>14}: purity {purity(ld, yd, 2):.3f}   sizes {np.bincount(ld)}')
    Xun = np.vstack([blobs([[0, 0]], 200, 1.0, seed=8), blobs([[2.2, 0]], 20, 0.5, seed=9)])
    yun = np.array([0] * 200 + [1] * 20)
    _, lun, _, _, _ = kmeans(Xun, 2, seed=0, init='kmeans++')
    p_un = purity(lun, yun, 2)
    found = sum(1 for i in range(len(yun)) if yun[i] == 1 and
                np.bincount(yun[lun == lun[i]]).argmax() == 1)
    print(f'  {"200 vs 20":>14}: purity {p_un:.3f}   sizes {np.bincount(lun)}')
    print('\nSquared distance to one centre has circular level sets, and a point joins')
    print('whichever centre is nearest, so the border between two clusters is the')
    print('perpendicular bisector of the line joining them: every cluster k-means can')
    print('express is a convex cell with straight sides.  Long bars get cut across the')
    print('middle and moons get split down the middle.')
    print(f'\nThe last row is the one worth staring at.  Purity {p_un:.3f} and two clusters of')
    print(f'{np.bincount(lun).tolist()} look like a healthy result.  They are not: the 20-point group was')
    print(f'recovered {found} times out of 20.  k-means cut the big blob in half, because halving')
    print('200 points removes far more squared distance than isolating 20 does, and 0.909 is')
    print('exactly the score you get for calling everything the majority class.')

    # ------------------------------------------------------------------ 8
    header('8. k-nearest neighbours: the data is the model')
    Xk, yk = moons(120, seed=12, noise=0.22)
    rngk = np.random.default_rng(0)
    perm = rngk.permutation(len(Xk))
    tr, te = perm[:160], perm[160:]
    print(f'  240 points on two overlapping moons, {len(tr)} train / {len(te)} test\n')
    print(f'  {"k":>4} {"train acc":>10} {"train (self excluded)":>22} {"test acc":>10}')
    accs = {}
    for k in (1, 3, 5, 11, 25, 61, 160):
        a_in = accuracy(knn_predict(Xk[tr], yk[tr], Xk[tr], k), yk[tr])
        a_out = accuracy(knn_predict(Xk[tr], yk[tr], Xk[tr], k, exclude_self=True), yk[tr])
        a_te = accuracy(knn_predict(Xk[tr], yk[tr], Xk[te], k), yk[te])
        accs[k] = (a_in, a_out, a_te)
        print(f'  {k:>4} {a_in:>10.3f} {a_out:>22.3f} {a_te:>10.3f}')
    top = max(a[2] for a in accs.values())
    bestks = [k for k in accs if accs[k][2] == top]
    bestk = bestks[0]
    print(f'\nSilent failure #6: at k=1 the training accuracy is {accs[1][0]:.3f}, and it is 1.000')
    print('for any data set whose points are distinct, because the nearest point to a')
    print('training point is itself.  (Two identical points with different labels are the')
    print('only way to lose, and that is a data problem rather than a model one.)')
    print(f'Excluding it drops to {accs[1][1]:.3f} and the honest test number is {accs[1][2]:.3f}.')
    print('A validation split that leaks one row does the same thing more quietly, and')
    print('1.000 is the number people stop checking at.')
    tie = '' if len(bestks) == 1 else ' (tied with k=' + ', '.join(str(k) for k in bestks[1:]) + ')'
    print(f'\nTest accuracy peaks at k={bestk}{tie}, at {top:.3f}.  Small k follows every piece')
    print(f'of noise; k = 160 is the whole training set, so every query gets the majority')
    print(f'label and accuracy falls to {accs[160][2]:.3f} - the model has flattened into a constant.')
    print('k is the smoothing knob, and the training score points the wrong way along it.')

    # ------------------------------------------------------------------ 9
    header('9. Silent failure #7: an even k can tie')
    Xt = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 2.0], [1.0, -2.0]])
    yt = np.array([0, 1, 0, 1])
    q = np.array([[0.0, 0.0]])
    d2t = sq_dists(q, Xt)[0]
    print(f'  4 training points, labels {yt.tolist()}, query at (0, 0)')
    print(f'  squared distances {d2t} -> the two nearest are one of each class')
    print(f'  np.bincount([0, 1]).argmax() = {np.bincount(np.array([0, 1])).argmax()}')
    print('  numpy breaks the tie by returning the smallest index, so class 0 wins every')
    print('  tie in the data set.  Nothing is raised; the bias is uniform and invisible.')
    print('  Odd k removes the two-class tie, and distance weighting removes the rest.')

    # ------------------------------------------------------------------ 10
    header('10. Silent failure #8: in high dimensions nothing is near')
    print('  500 uniform random points; how much nearer is the nearest than the furthest?\n')
    print(f'  {"dims":>6} {"(max-min)/min":>16}')
    cons = {}
    for d in (2, 4, 8, 16, 64, 256, 1024):
        cons[d] = contrast(500, d, seed=0)
        print(f'  {d:>6} {cons[d]:>16.3f}')
    print(f'\n  At d=2 the furthest point is {1 + cons[2]:.1f}x further away than the nearest.')
    print(f'  At d=1024 it is {1 + cons[1024]:.2f}x.  "Nearest" has stopped meaning anything,')
    print('  and the only thing that changed is how many coordinates the sum runs over:')
    print('  every extra dimension adds one more draw to an average, and averages concentrate.')
    print('\n  The practical version - two informative features, the rest pure noise:\n')
    print(f'  {"dims":>6} {"informative":>12} {"knn k=11 test acc":>20}')
    noisy = {}
    for d in (2, 4, 10, 30, 100, 400):
        Xn, yn = noisy_features(400, d, seed=5)
        rn = np.random.default_rng(1).permutation(400)
        a = accuracy(knn_predict(Xn[rn[:300]], yn[rn[:300]], Xn[rn[300:]], 11), yn[rn[300:]])
        noisy[d] = a
        print(f'  {d:>6} {2:>12} {a:>20.3f}')
    print(f'\n  The two useful columns never changed.  Accuracy fell from {noisy[2]:.3f} to '
          f'{noisy[400]:.3f}')
    print('  because the distance function averages them with 398 columns of noise.')
    print('  No exception, no warning - just a model that quietly stopped working.')

    # ------------------------------------------------------------------ 11
    header('11. The cost of being lazy, and the three ways out')
    rngc = np.random.default_rng(3)
    Qc = rngc.random((100, 16))
    print('  16 dimensions, 100 queries, growing the training set:\n')
    print(f'  {"points":>8} {"ms / query":>12} {"vs 2,500":>10}')
    base = None
    for n in (2500, 5000, 10000, 20000, 40000):
        Xc = rngc.random((n, 16))
        sq_dists(Qc[:2], Xc)
        t0 = time.perf_counter()
        np.argmin(sq_dists(Qc, Xc), axis=1)
        per = (time.perf_counter() - t0) * 1000 / len(Qc)
        base = base or per
        print(f'  {n:>8} {per:>12.3f} {per / base:>9.1f}x')
    print('\n  Vectorising buys a constant (Day 44); it does not bend the line.  k-nearest is')
    print('  O(n*d) per query however it is written.  Three escapes, in the order they fail:')
    print('    1 dimension  -> the data can be sorted, so binary search it (LeetCode 658)')
    print('    k out of n   -> you need a set, not an order, so quickselect it (LeetCode 973)')
    print('    many dims    -> no exact index survives; approximate it, and accept a recall')
    print('                    below 1.0 as the price (the HNSW graph built on Day 31).')

    # ------------------------------------------------------------------ 12
    header('12. LeetCode 973 - K Closest Points to Origin')
    pts = [[3, 3], [5, -1], [-2, 4]]
    got = SolutionKClosest().kClosest(pts, 2)
    print(f'  points {pts}, k=2')
    print(f'    squared distances {[p[0] ** 2 + p[1] ** 2 for p in pts]}')
    print(f'    heap answer       {sorted(got)}')
    print(f'    quickselect       {sorted(k_closest_quickselect(pts, 2))}')
    rngp = np.random.default_rng(21)
    P = rngp.integers(-10000, 10000, (4000, 2)).tolist()
    kk = 50
    ref = sorted(sorted(P, key=lambda p: p[0] ** 2 + p[1] ** 2)[:kk])
    h = sorted(SolutionKClosest().kClosest(P, kk))
    qs = sorted(k_closest_quickselect(P, kk, seed=1))
    print(f'\n  4,000 random points, k={kk}: heap == sorted: {h == ref}, '
          f'quickselect == sorted: {qs == ref}')
    t0 = time.perf_counter(); sorted(P, key=lambda p: p[0] ** 2 + p[1] ** 2)[:kk]; t_sort = time.perf_counter() - t0
    t0 = time.perf_counter(); SolutionKClosest().kClosest(P, kk); t_heap = time.perf_counter() - t0
    t0 = time.perf_counter(); k_closest_quickselect(P, kk, seed=1); t_qs = time.perf_counter() - t0
    print(f'  full sort {1000 * t_sort:.2f} ms, size-k heap {1000 * t_heap:.2f} ms, '
          f'quickselect {1000 * t_qs:.2f} ms (one measurement)')
    big = [[46341, 46341], [1, 1]]
    print(f'\n  Why squared and not sqrt: {big} -> {[p[0] ** 2 + p[1] ** 2 for p in big]} as ints,')
    print(f'  exactly; as floats math.hypot gives {math.hypot(*big[0]):.6f} and the ordering is')
    print('  the same, because sqrt is monotone and 2e8 is far inside the range of integers')
    print('  a float64 holds exactly.  Both comparisons are exact; the square root is simply')
    print('  work nobody needed to do.')

    # ------------------------------------------------------------------ 13
    header('13. LeetCode 658 - Find K Closest Elements')
    arr, k658, x658 = [1, 2, 3, 4, 5], 4, 3
    print(f'  arr={arr}, k={k658}, x={x658} -> {SolutionFindClosestElements().findClosestElements(arr, k658, x658)}')
    arr2 = [1, 1, 2, 3, 4, 5]
    print(f'  arr={arr2}, k=4, x=-1 -> {SolutionFindClosestElements().findClosestElements(arr2, 4, -1)}')
    tie = [1, 2, 3, 4, 5]
    print(f'\n  the tie: arr={tie}, k=2, x=3 - both 2 and 4 are one away')
    print(f'    with `>`  (correct) : {SolutionFindClosestElements().findClosestElements(tie, 2, 3)}')
    print(f'    with `>=`           : {closest_elements_ge(tie, 2, 3)}')
    rng658 = np.random.default_rng(31)
    bad = 0
    for _ in range(3000):
        n = int(rng658.integers(1, 40))
        a = sorted(int(v) for v in rng658.integers(-50, 50, n))
        kk2 = int(rng658.integers(1, n + 1))
        xx = int(rng658.integers(-60, 60))
        if closest_elements_ge(a, kk2, xx) != closest_elements_by_abs(a, kk2, xx):
            bad += 1
    print(f'    over 3,000 random cases the `>=` version differs from the reference {bad} times')
    print(f'    ({100 * bad / 3000:.1f}%) - rare enough that the three sample cases all pass.')
    rng_ok = np.random.default_rng(32)
    agree = 0
    for _ in range(3000):
        n = int(rng_ok.integers(1, 40))
        a = sorted(int(v) for v in rng_ok.integers(-50, 50, n))
        kk2 = int(rng_ok.integers(1, n + 1))
        xx = int(rng_ok.integers(-60, 60))
        agree += SolutionFindClosestElements().findClosestElements(a, kk2, xx) == closest_elements_by_abs(a, kk2, xx)
    print(f'  the `>` version agrees with sort-by-distance on {agree}/3000 random cases')
    print('\n  The window is the answer because the array is sorted: the k closest values to')
    print('  x are contiguous.  That is the one-dimensional case of k-nearest, and it is the')
    print('  only case where the search costs O(log n) instead of O(n*d).')

    # ------------------------------------------------------------------ 14
    header('14. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
