"""Day 49 - PCA, SVD and t-SNE, in pure NumPy.

Every dimensionality reduction answers a question you did not ask.  Run
`python dimred.py` to print the whole story; every number is exact and checked by
an assert at the bottom.
"""

import math

import numpy as np


# ================================================================ 1. PCA
def center(X):
    mu = X.mean(axis=0)
    return X - mu, mu


class PCA:
    """Principal components, computed from the SVD of the centred data.

    The covariance route (eigendecompose Xc^T Xc / (n - 1) on the centred data) gives
    the same answer in exact arithmetic and a worse one in floating point - section 6
    measures how much worse.
    """

    def __init__(self, k=None, center_data=True):
        self.k = k
        self.center_data = center_data

    def fit(self, X):
        self.mean_ = X.mean(axis=0) if self.center_data else np.zeros(X.shape[1])
        Xc = X - self.mean_
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.singular_values_ = s
        self.components_ = Vt                       # rows are the directions
        # the variance along component j, with the same 1/(n-1) convention as np.var(ddof=1)
        self.explained_variance_ = s ** 2 / (len(X) - 1)
        tot = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / tot if tot > 0 else self.explained_variance_
        return self

    def transform(self, X, k=None):
        k = k or self.k or self.components_.shape[0]
        return (X - self.mean_) @ self.components_[:k].T

    def inverse_transform(self, Z):
        return Z @ self.components_[:Z.shape[1]] + self.mean_


def pca_via_covariance(X, center_data=True):
    """The textbook route: build the covariance matrix and eigendecompose it.

    `np.linalg.eigh` returns eigenvalues in ascending order, so everything has to be
    reversed - a step that is easy to forget and that produces a perfectly ordinary
    looking answer pointing at the least important direction.
    """
    Xc = X - X.mean(axis=0) if center_data else X
    C = Xc.T @ Xc / (len(X) - 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order].T


def reconstruction_error(X, p, k):
    """Mean squared error of rebuilding X from its first k components."""
    return float(np.mean((X - p.inverse_transform(p.transform(X, k))) ** 2))


# ================================================================ 2. t-SNE
def pairwise_sq_dists(X):
    """|a - b|^2 for every pair.  There are n(n-1)/2 of them, which is the whole
    reason t-SNE is slow and the whole point of LeetCode 719 below."""
    x2 = np.einsum('ij,ij->i', X, X)
    return np.maximum(x2[:, None] - 2.0 * X @ X.T + x2[None, :], 0.0)


def conditional_P(D, perplexity, tol=1e-5, max_iter=60):
    """Row i gets its own gaussian width, chosen by binary search so that the
    perplexity of row i's neighbour distribution hits the number you asked for, to
    within `tol` in nats (the search stops at 1e-5, i.e. perplexity 30 comes out
    somewhere in 29.9997-30.0003, not at exactly 30).

    Perplexity is 2^entropy: "how many neighbours does this point effectively have".
    Every row therefore gets a different sigma, which is how t-SNE copes with regions
    of different density - and also why the output says nothing about density.
    """
    n = len(D)
    P = np.zeros((n, n))
    target = math.log(perplexity)
    betas = np.zeros(n)
    for i in range(n):
        lo, hi = -np.inf, np.inf
        beta = 1.0                                  # beta = 1 / (2 sigma^2)
        Di = np.delete(D[i], i)
        for _ in range(max_iter):
            Pi = np.exp(-Di * beta)
            s = Pi.sum()
            if s <= 0:
                H, Pi = 0.0, np.zeros_like(Di)
            else:
                H = math.log(s) + beta * float(np.sum(Di * Pi)) / s
                Pi = Pi / s
            if abs(H - target) < tol:
                break
            if H > target:                          # too many neighbours: narrow it
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        betas[i] = beta
        P[i] = np.insert(Pi, i, 0.0)
    return P, betas


def joint_P(X, perplexity):
    P, betas = conditional_P(pairwise_sq_dists(X), perplexity)
    P = (P + P.T) / (2 * len(X))                    # symmetrise, then normalise
    # the floor keeps log(P / Q) finite; renormalise so P is still a distribution
    P = np.maximum(P, 1e-12)
    return P / P.sum(), betas


def tsne(X, perplexity=30.0, dim=2, n_iter=600, lr=200.0, seed=0,
         early_exaggeration=12.0, exaggerate_until=100, momentum=(0.5, 0.8)):
    """Minimise KL(P || Q) by gradient descent on the embedding itself.

    The embedding IS the parameter vector - there is no function being fitted, which
    is why t-SNE has no `transform` for new points and why two runs differ.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    P, _ = joint_P(X, perplexity)
    Y = rng.standard_normal((n, dim)) * 1e-4
    dY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    losses = []
    for it in range(n_iter):
        Pe = P * early_exaggeration if it < exaggerate_until else P
        num = 1.0 / (1.0 + pairwise_sq_dists(Y))    # the Student-t kernel
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        PQ = Pe - Q
        # dC/dy_i = 4 sum_j (p_ij - q_ij) (y_i - y_j) / (1 + |y_i - y_j|^2)
        W = PQ * num
        grad = 4.0 * (Y * W.sum(axis=1)[:, None] - W @ Y)
        mom = momentum[0] if it < 250 else momentum[1]
        gains = np.where(np.sign(grad) != np.sign(dY), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        dY = mom * dY - lr * gains * grad
        Y = Y + dY
        Y = Y - Y.mean(axis=0)
        losses.append(float(np.sum(P * np.log(P / Q))))
    return Y, losses


def kl_divergence(P, Q):
    return float(np.sum(P * np.log(P / Q)))


def neighbour_recall(X, Y, k=10):
    """Of the k nearest neighbours a point has in the original space, how many are
    still among its k nearest in the embedding?  This is the thing t-SNE optimises,
    and the only thing the picture is entitled to claim."""
    def knn(A):
        D = pairwise_sq_dists(A)
        np.fill_diagonal(D, np.inf)
        return np.argsort(D, axis=1)[:, :k]
    a, b = knn(X), knn(Y)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(X))]))


# ================================================================ 3. reading the picture
def kmeans_small(X, k, seed=0, restarts=8, iters=60):
    """A small Lloyd loop with restarts, only used to ask how clustered a picture
    looks.  Day 47 has the real one, and all of its caveats."""
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        C = X[rng.choice(len(X), k, replace=False)].copy()
        lab = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(axis=2), axis=1)
            new = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else C[j]
                            for j in range(k)])
            if np.allclose(new, C):
                break
            C = new
        inertia = float(np.sum((X - C[lab]) ** 2))
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def silhouette(X, labels):
    """Mean over points of (b - a) / max(a, b): how much closer a point is to its own
    group than to the nearest other one.  Near 0 means the grouping is arbitrary."""
    D = np.sqrt(pairwise_sq_dists(X))
    ks = np.unique(labels)
    out = []
    for i in range(len(X)):
        own = labels[i] == labels
        own[i] = False
        if not own.any():
            continue
        a = D[i, own].mean()
        b = min(D[i, labels == c].mean() for c in ks if c != labels[i])
        out.append((b - a) / max(a, b))
    return float(np.mean(out))


# ================================================================ 4. data sets
def anisotropic(n=300, seed=0):
    """A stretched cloud sitting a long way from the origin.  The offset is bigger
    than the spread, which is the only thing section 3 needs."""
    rng = np.random.default_rng(seed)
    A = np.array([[2.0, 0.5], [0.5, 0.6]])
    return rng.standard_normal((n, 2)) @ A + np.array([8.0, 5.0])


def hidden_signal(n=400, seed=1, sep=0.8, noise_scale=6.0, angle=30.0):
    """Two classes separated along one direction, with a much larger amount of
    variance along a direction that carries no information at all.  Then the whole
    thing is rotated, so the useful direction is not an axis."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal = np.where(y == 1, sep, -sep) + 0.25 * rng.standard_normal(n)
    noise = noise_scale * rng.standard_normal(n)
    X = np.column_stack([noise, signal])
    t = math.radians(angle)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return X @ R.T, y


def requests(n=300, seed=3):
    """Day 47's two columns again: prompt_tokens in the hundreds, a hit rate in [0, 1]."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(200, 4000, n), rng.uniform(0, 1, n)])


def ill_conditioned(n=200, eps=1e-8, seed=0):
    """Singular values proportional to 1, eps, eps/2 in a random orthogonal basis, so
    the right answer is available to compare against.  The basis is scaled by
    sqrt(n - 1) so the covariance route's eigenvalues come out near 1 - that scale is
    why the smallest singular value below reads 7.04e-04 and not 5e-05 for eps=1e-4."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    base = np.linalg.qr(rng.standard_normal((n, 3)))[0] * math.sqrt(n - 1)
    return base @ np.diag([1.0, eps, eps / 2]) @ Q.T


def blobs(spreads, centers, n_per=60, seed=1):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, (c, s) in enumerate(zip(centers, spreads)):
        X.append(np.array(c, dtype=float) + s * rng.standard_normal((n_per, len(c))))
        y += [i] * n_per
    return np.vstack(X), np.array(y)


def pure_noise(n=180, d=10, seed=7):
    return np.random.default_rng(seed).standard_normal((n, d))


def best_threshold_accuracy(v, y):
    """The best accuracy any single cut on one axis can reach.  A cheap, honest way
    to ask whether a projection kept the thing you cared about."""
    order = np.argsort(v)
    ys = y[order]
    n = len(ys)
    ones = np.concatenate([[0], np.cumsum(ys)])
    best = 0.0
    for i in range(n + 1):
        left_zeros = i - ones[i]
        right_ones = ones[n] - ones[i]
        best = max(best, (left_zeros + right_ones) / n, (ones[i] + (n - i) - right_ones) / n)
    return float(best)


def spread(X):
    return float(np.sqrt(((X - X.mean(axis=0)) ** 2).sum(axis=1).mean()))


def centre_distance(X, a, b):
    return float(np.linalg.norm(X[a].mean(axis=0) - X[b].mean(axis=0)))


def pairwise_corr(A, B):
    """Correlation between the two sets of pairwise distances - how much two pictures
    of the same points agree."""
    iu = np.triu_indices(len(A), 1)
    return float(np.corrcoef(np.sqrt(pairwise_sq_dists(A))[iu],
                             np.sqrt(pairwise_sq_dists(B))[iu])[0, 1])


# ================================================================ 5. LeetCode
class SolutionKthPairDistance:
    def smallestDistancePair(self, nums: list[int], k: int) -> int:
        # There are n(n-1)/2 distances and up to 5e7 of them, so they are never
        # built.  Binary search the ANSWER instead: count(m) = how many pairs are
        # at most m apart is non-decreasing in m, so the smallest m with
        # count(m) >= k is the k-th smallest distance.
        nums.sort()
        n = len(nums)

        def count(m: int) -> int:
            total = j = 0
            for i in range(n):
                while nums[i] - nums[j] > m:
                    j += 1
                total += i - j          # every index in [j, i) pairs with i
            return total

        lo, hi = 0, nums[-1] - nums[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if count(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo


def kth_pair_distance_lo1(nums, k):
    """The same search started at lo = 1 instead of lo = 0 - "a distance is at least
    1" - which is true of every pair of DIFFERENT values and false the moment the
    array repeats one."""
    nums = sorted(nums)
    n = len(nums)

    def count(m):
        total = j = 0
        for i in range(n):
            while nums[i] - nums[j] > m:
                j += 1
            total += i - j
        return total

    lo, hi = 1, nums[-1] - nums[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return max(lo, 0)


def kth_pair_distance_brute(nums, k):
    ds = sorted(abs(a - b) for i, a in enumerate(nums) for b in nums[i + 1:])
    return ds[k - 1]


class SolutionEqualPairs:
    def equalPairs(self, grid: list[list[int]]) -> int:
        # A row and a column are equal when they are the same sequence of numbers.
        # A tuple keeps the element boundaries; gluing the digits into one string
        # throws them away, so (1, 2, 34) and (1, 23, 4) collide.
        from collections import Counter
        rows = Counter(tuple(r) for r in grid)
        return sum(rows[col] for col in zip(*grid))


def equal_pairs_joined(grid):
    """The same count with `''.join(map(str, row))` as the key: [1, 23] and [12, 3]
    become the same string, so distinct rows are merged and the count comes out high."""
    from collections import Counter
    rows = Counter(''.join(map(str, r)) for r in grid)
    return sum(rows[''.join(map(str, c))] for c in zip(*grid))


def equal_pairs_brute(grid):
    n = len(grid)
    return sum(1 for r in range(n) for c in range(n)
               if all(grid[r][i] == grid[i][c] for i in range(n)))


def check() -> None:
    """Everything the walkthrough claims, re-derived from scratch."""
    rng = np.random.default_rng(0)

    # The SVD route and the covariance route agree, up to the sign of each component.
    X = anisotropic(300, seed=0)
    p = PCA().fit(X)
    w, V = pca_via_covariance(X)
    assert np.allclose(p.explained_variance_, w, atol=1e-9)
    for j in range(2):
        assert abs(abs(float(p.components_[j] @ V[j])) - 1.0) < 1e-9
    # explained variance really is the variance of the projection
    Z = p.transform(X)
    assert np.allclose(Z.var(axis=0, ddof=1), p.explained_variance_, atol=1e-9)
    # and the components are orthonormal
    assert np.allclose(p.components_ @ p.components_.T, np.eye(2), atol=1e-9)

    # Without centring the first component chases the mean, and reports a HIGHER
    # explained-variance ratio for doing it.
    pu = PCA(center_data=False).fit(X)
    mean_dir = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
    ang = lambda a, b: math.degrees(math.acos(min(1.0, abs(float(a @ b)))))
    assert ang(pu.components_[0], mean_dir) < 1.0
    assert ang(pu.components_[0], p.components_[0]) > 10.0
    assert pu.explained_variance_ratio_[0] > p.explained_variance_ratio_[0]

    # Units pick the axes.
    R = requests(300, seed=3)
    pr = PCA().fit(R)
    assert pr.explained_variance_ratio_[0] > 0.999
    assert abs(abs(pr.components_[0][0]) - 1.0) < 1e-6

    # A component can hold almost all of the variance and none of the signal.
    H, hy = hidden_signal(400, seed=1)
    ph = PCA().fit(H)
    Z2 = ph.transform(H, 2)
    a1 = best_threshold_accuracy(Z2[:, 0], hy)
    a2 = best_threshold_accuracy(Z2[:, 1], hy)
    assert ph.explained_variance_ratio_[0] > 0.9 and a1 < 0.65 < 0.95 < a2

    # Squaring the condition number destroys the small singular values.
    Xi = ill_conditioned(200, eps=1e-8, seed=0)
    sv = np.linalg.svd(Xi - Xi.mean(axis=0), full_matrices=False)[1]
    wi, _ = pca_via_covariance(Xi)
    assert sv[-1] > 1e-9 and np.sqrt(max(wi[-1], 0.0) * 199) == 0.0
    assert (pca_via_covariance(ill_conditioned(200, eps=1e-9, seed=0))[0] < 0).any()

    # Eckart-Young, exactly.
    A = rng.standard_normal((60, 20)) @ rng.standard_normal((20, 20))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    for k in (1, 5, 10, 20):
        Ak = (U[:, :k] * s[:k]) @ Vt[:k]
        assert abs(float(np.linalg.norm(A - Ak)) - math.sqrt((s[k:] ** 2).sum())) < 1e-8

    # P is a symmetric distribution that sums to 1, and every row hits the perplexity
    # it was asked for.
    Xb, _ = blobs([1.0, 1.0, 1.0], [[0, 0, 0, 0, 0], [15, 0, 0, 0, 0], [0, 40, 0, 0, 0]],
                  60, seed=1)
    Pm, betas = joint_P(Xb, 30.0)
    assert abs(Pm.sum() - 1.0) < 1e-9 and np.abs(Pm - Pm.T).max() < 1e-15
    Dm = pairwise_sq_dists(Xb)
    Pc, _ = conditional_P(Dm, 30.0)
    for i in (0, 37, 90, 179):
        row = Pc[i][Pc[i] > 0]
        assert abs(2.0 ** float(-np.sum(row * np.log2(row))) - 30.0) < 0.1

    # The gradient really does decrease KL, and the map keeps neighbours.
    Yb, loss = tsne(Xb, perplexity=30.0, n_iter=600, lr=50.0, seed=0)
    assert loss[-1] < loss[0] / 4
    assert neighbour_recall(Xb, Yb, 10) > 0.5

    # Noise comes out looking clustered, and the clusters move when the seed does.
    Xn = pure_noise(180, 10, seed=7)
    s_raw = silhouette(Xn, kmeans_small(Xn, 3, seed=0))
    Yn, _ = tsne(Xn, perplexity=30.0, n_iter=600, lr=50.0, seed=0)
    assert silhouette(Yn, kmeans_small(Yn, 3, seed=0)) > 3 * s_raw
    runs_n = [tsne(Xn, perplexity=30.0, n_iter=600, lr=50.0, seed=sd)[0] for sd in range(2)]
    runs_b = [tsne(Xb, perplexity=30.0, n_iter=600, lr=50.0, seed=sd)[0] for sd in range(2)]
    assert pairwise_corr(*runs_n) < 0.6 < 0.95 < pairwise_corr(*runs_b)

    # Cluster width and between-cluster distance do not survive the trip.
    Xs, ys = blobs([0.5, 5.0, 0.5], [[0, 0, 0, 0, 0], [20, 0, 0, 0, 0], [0, 60, 0, 0, 0]],
                   60, seed=1)
    Ys_, _ = tsne(Xs, perplexity=30.0, n_iter=600, lr=50.0, seed=0)
    m = [ys == i for i in range(3)]
    o_sp = [spread(Xs[mi]) for mi in m]
    e_sp = [spread(Ys_[mi]) for mi in m]
    assert max(o_sp) / min(o_sp) > 10 and max(e_sp) / min(e_sp) < 1.5
    Xf, yf = blobs([1.0, 1.0, 1.0], [[0, 0, 0, 0, 0], [15, 0, 0, 0, 0], [0, 400, 0, 0, 0]],
                   60, seed=1)
    Yf, _ = tsne(Xf, perplexity=30.0, n_iter=600, lr=50.0, seed=0)
    mf = [yf == i for i in range(3)]
    assert centre_distance(Xf, mf[0], mf[2]) / centre_distance(Xf, mf[0], mf[1]) > 20
    assert centre_distance(Yf, mf[0], mf[2]) / centre_distance(Yf, mf[0], mf[1]) < 1.5

    # LeetCode 719: correct against brute force, and the lo = 1 slip is not.
    r719 = np.random.default_rng(3)
    diffs = 0
    for _ in range(2000):
        n = int(r719.integers(2, 9))
        a = [int(v) for v in r719.integers(0, 15, n)]
        k = int(r719.integers(1, n * (n - 1) // 2 + 1))
        want = kth_pair_distance_brute(a, k)
        assert SolutionKthPairDistance().smallestDistancePair(a[:], k) == want
        diffs += kth_pair_distance_lo1(a[:], k) != want
    assert diffs > 0
    assert SolutionKthPairDistance().smallestDistancePair([1, 3, 1], 1) == 0
    assert SolutionKthPairDistance().smallestDistancePair([1, 1, 1], 2) == 0
    assert SolutionKthPairDistance().smallestDistancePair([1, 6, 1], 3) == 5

    # LeetCode 2352: correct against brute force, and the joined-string key is not.
    r2352 = np.random.default_rng(4)
    for _ in range(1500):
        n = int(r2352.integers(1, 5))
        g = [[int(v) for v in r2352.integers(1, 60, n)] for _ in range(n)]
        assert SolutionEqualPairs().equalPairs(g) == equal_pairs_brute(g)
    assert SolutionEqualPairs().equalPairs([[3, 2, 1], [1, 7, 6], [2, 7, 7]]) == 1
    assert SolutionEqualPairs().equalPairs([[3, 1, 2, 2], [1, 4, 4, 5], [2, 4, 2, 2],
                                            [2, 4, 2, 2]]) == 3
    trap = [[1, 2, 34], [23, 5, 6], [4, 7, 8]]
    assert SolutionEqualPairs().equalPairs(trap) == 0 == equal_pairs_brute(trap)
    assert equal_pairs_joined(trap) == 1


def header(t):
    print('\n' + '=' * 72 + '\n' + t + '\n' + '=' * 72)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # ------------------------------------------------------------------ 1
    header('1. Two methods, two questions, and neither of them is yours')
    print("""Dimensionality reduction takes d columns and hands back 2, and the d - 2 that
went away leave no trace.  What comes out is a picture, and a picture is extremely
easy to over-read.

    PCA   answers: which directions carry the most variance?
    t-SNE answers: which points are each other's neighbours?

Neither of those is "which directions matter" or "how far apart are these groups",
and those are the two things people read off the plot.  Everything below is a
measurement of the gap between what was optimised and what was read.""")

    # ------------------------------------------------------------------ 2
    header('2. PCA is the SVD of the centred data')
    X = anisotropic(300, seed=0)
    p = PCA().fit(X)
    w, V = pca_via_covariance(X)
    print('  300 points, 2 columns, a stretched cloud.')
    print(f'  singular values            {p.singular_values_}')
    print(f'  explained variance         {p.explained_variance_}')
    print(f'  eigenvalues of the covariance {w}')
    print(f'  the two routes agree to    {np.abs(p.explained_variance_ - w).max():.3e}')
    print(f'\n  PC1 from the SVD        {p.components_[0]}')
    print(f'  PC1 from eigh           {V[0]}')
    print(f'  dot product             {p.components_[0] @ V[0]:+.6f}')
    print('\n  The dot product is +1 here, but only because both routes happened to')
    print('  pick the same sign.  An eigenvector with a simple eigenvalue is defined')
    print('  only up to its sign: v and -v span the same line and both are correct.')
    print('  (When two eigenvalues are equal it is worse - any rotation inside their')
    print('  plane is equally valid.)  Two libraries, two versions of one library, or')
    print('  the same code on another machine can hand you a mirrored plot, and nothing')
    print('  warns you.  Code that compares components has to compare them up to sign,')
    print('  |u . v|, which is what check() does; anything a human will look at needs a')
    print('  pinned convention instead - e.g. force the largest-magnitude entry of each')
    print('  component positive before plotting.')

    # ------------------------------------------------------------------ 3
    header('3. Silent failure #1: forgetting to subtract the mean')
    pc = PCA(center_data=True).fit(X)
    pu = PCA(center_data=False).fit(X)
    mean_dir = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
    ang = lambda a, b: math.degrees(math.acos(min(1.0, abs(float(a @ b)))))
    print(f'  the cloud sits at {np.round(X.mean(axis=0), 3)}, and its spread is '
          f'{spread(X - X.mean(axis=0)):.3f}')
    print(f'\n  centred   PC1 {pc.components_[0]}   explained variance ratio {pc.explained_variance_ratio_}')
    print(f'  uncentred PC1 {pu.components_[0]}   explained variance ratio {pu.explained_variance_ratio_}')
    print(f'\n  angle between the uncentred PC1 and the direction of the mean: {ang(pu.components_[0], mean_dir):.3f} deg')
    print(f'  angle between the uncentred PC1 and the real PC1              : {ang(pu.components_[0], pc.components_[0]):.3f} deg')
    print(f'\n  Without centring, "variance" is measured about the origin instead of about')
    print('  the data, so the first component points at where the data IS rather than at')
    print('  how it is shaped.  What makes it dangerous is the score it reports:')
    print(f'  the broken version claims {pu.explained_variance_ratio_[0]:.4f} of the variance in one component')
    mse_c = reconstruction_error(X, pc, 1)
    mse_u = reconstruction_error(X, pu, 1)
    print(f'  against the correct {pc.explained_variance_ratio_[0]:.4f}.  It looks better - the one number anyone')
    print('  actually reports goes UP when you make this mistake, because its denominator,')
    print('  the total "variance" measured about the origin, was inflated by the distance')
    print('  to the mean along with the numerator.')
    print(f'\n  Not every number is fooled, which is the useful part.  Rebuilding the data')
    print(f'  from one component costs {mse_c:.6f} centred and {mse_u:.6f} uncentred, so')
    print(f'  reconstruction error is {mse_u / mse_c:.1f}x worse and does catch it.  It is simply not the')
    print('  number anyone puts on the slide.')

    # ------------------------------------------------------------------ 4
    header('4. Silent failure #2: the units pick the axes')
    R = requests(300, seed=3)
    pr = PCA().fit(R)
    Rz = (R - R.mean(axis=0)) / R.std(axis=0)
    pz = PCA().fit(Rz)
    print('  Day 47 and Day 48 both ran into this; PCA is where it is most visible,')
    print('  because the answer IS a direction.  Two columns: prompt_tokens in the')
    print('  hundreds, a cache hit rate in [0, 1], both pure noise and independent.\n')
    print(f'  raw units    : PC1 [{pr.components_[0][0]:.8f} {pr.components_[0][1]:.8f}]'
          f'  explained variance ratio [{pr.explained_variance_ratio_[0]:.8f} '
          f'{pr.explained_variance_ratio_[1]:.8f}]')
    print(f'  standardised : PC1 [{pz.components_[0][0]:.8f} {pz.components_[0][1]:.8f}]'
          f'  explained variance ratio [{pz.explained_variance_ratio_[0]:.8f} '
          f'{pz.explained_variance_ratio_[1]:.8f}]')
    print(f'\n  In raw units PC1 is the token axis to within {math.degrees(math.acos(min(1.0, abs(pr.components_[0][0])))):.4f} degrees, and PCA')
    print(f'  reports that one component explains {100 * pr.explained_variance_ratio_[0]:.6f}% of the variance of two')
    print('  independent noise columns.  That number is true, and it is a fact about')
    print('  the units rather than about the data.')
    print('  Standardising makes the two columns equally loud, which is a different')
    print('  arbitrary choice, not a neutral one - but at least it is one you made.')

    # ------------------------------------------------------------------ 5
    header('5. Silent failure #3: explained variance is not usefulness')
    H, hy = hidden_signal(400, seed=1)
    ph = PCA().fit(H)
    print('  400 points, two classes.  The classes are separated along one direction;')
    print('  another direction carries far more variance and no information at all,')
    print('  and the whole thing is rotated so that neither one is an axis.\n')
    Z2 = ph.transform(H, 2)
    per_comp = [best_threshold_accuracy(Z2[:, j], hy) for j in range(2)]
    print(f'  {"component":>10} {"variance explained":>19} {"best single cut":>17}')
    for j in range(2):
        print(f'  {"PC" + str(j + 1):>10} {100 * ph.explained_variance_ratio_[j]:>18.2f}% '
              f'{per_comp[j]:>17.4f}')
    print(f'\n  PC1 holds {100 * ph.explained_variance_ratio_[0]:.2f}% of the variance and separates the classes at '
          f'{per_comp[0]:.4f},')
    print(f'  a coin flip.  PC2 holds {100 * ph.explained_variance_ratio_[1]:.2f}% and separates them at {per_comp[1]:.4f}.')
    print(f'  The ratio of the two variances is {ph.explained_variance_[0] / ph.explained_variance_[1]:.1f} to 1, and the ordering is')
    print('  exactly backwards for the only question anyone was going to ask.')
    print('\n  "Keep 95% of the variance" discards PC2 here.  Note that PCA did find the')
    print('  right pair of directions - the rotation is correct and the signal is clean')
    print('  along one of its axes - it only sorted them by the wrong key.  It was asked')
    print('  which direction is widest and it answered correctly; nobody asked it which')
    print('  direction was useful, and it never sees the labels, so it cannot know.')

    # ------------------------------------------------------------------ 6
    header('6. Why the SVD and not the covariance matrix')
    print('  Both routes are the same mathematics.  Forming X^T X squares the condition')
    print('  number, so a matrix you could still work with becomes one you cannot.\n')
    print(f'  {"eps":>13} {"cond(X)":>10} {"cond(cov)":>11} {"from SVD":>12} {"from eigh":>12} {"rel. error":>11}')
    cond_rows = []
    for eps in (1e-4, 1e-6, 1e-8, 1e-9):
        Xi = ill_conditioned(200, eps=eps, seed=0)
        Xc = Xi - Xi.mean(axis=0)
        sv = np.linalg.svd(Xc, full_matrices=False)[1]
        wi, _ = pca_via_covariance(Xi)
        sv_eig = np.sqrt(np.maximum(wi, 0) * (len(Xi) - 1))
        rel = abs(sv_eig[-1] - sv[-1]) / sv[-1]
        cond_rows.append((eps, rel, bool((wi < 0).any())))
        # cond(cov) is measured from the covariance matrix itself, not squared by hand
        cov_cond = float(np.linalg.cond(Xc.T @ Xc / (len(Xi) - 1)))
        print(f'  {eps:>13.0e} {sv[0] / sv[-1]:>10.1e} {cov_cond:>11.1e} '
              f'{sv[-1]:>12.4e} {sv_eig[-1]:>12.4e} {rel:>11.3e}')
    neg = [e for e, _, n_ in cond_rows if n_]
    print(f'\n  At 1e-8 the covariance route returns exactly 0 for a singular value that is')
    print('  really 7.04e-08: a direction with real variance in it has been rounded out')
    print('  of existence.  The arithmetic reason is one line - float64 carries about')
    print('  16 digits, cond(cov) = cond(X)^2, so once cond(X) passes about 1e8 the small')
    print('  singular values are below the noise floor of the squared problem.')
    print('\n  The cond(cov) column is measured from the covariance matrix rather than')
    print('  squared by hand, which is why the last row reads 6.5e+17 instead of the')
    print('  predicted 4.0e+18: by then the matrix is too degenerate for its own')
    print('  condition number to survive.  The measurement is a victim of what it measures.')
    if neg:
        wi, _ = pca_via_covariance(ill_conditioned(200, eps=neg[0], seed=0))
        print(f'\n  There is a second symptom: eigh returns {wi.min():.3e} for a covariance matrix,')
        print('  which is a negative variance.  np.sqrt of it is nan, so code that takes')
        print('  standard deviations from eigenvalues has to clamp at zero - and clamping')
        print('  is exactly what hides the problem.')

    # ------------------------------------------------------------------ 7
    header('7. Truncated SVD: the best rank-k approximation there is')
    rng = np.random.default_rng(5)
    A = rng.standard_normal((120, 40)) @ rng.standard_normal((40, 40))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print('  A random 120 x 40 matrix.  Eckart-Young says that truncating the SVD at k')
    print('  is the best rank-k approximation in Frobenius norm, and that the error is')
    print('  exactly the square root of the sum of the discarded singular values squared.\n')
    print(f'  {"k":>4} {"Frobenius error":>16} {"sqrt(sum dropped^2)":>21} {"|A|_F^2 kept":>14}')
    for k in (1, 5, 10, 20, 40):
        Ak = (U[:, :k] * s[:k]) @ Vt[:k]
        err = float(np.linalg.norm(A - Ak))
        pred = float(math.sqrt((s[k:] ** 2).sum()))
        print(f'  {k:>4} {err:>16.6f} {pred:>21.6f} {(s[:k] ** 2).sum() / (s ** 2).sum():>14.4f}')
    Ac = A - A.mean(axis=0)
    sc = np.linalg.svd(Ac, full_matrices=False)[1]
    print(f'\n  The last column is the share of the squared Frobenius norm kept, and on')
    print('  CENTRED data that is exactly PCA\'s explained-variance ratio - same ratio,')
    print('  one step before the statistics.  A is not centred here, so the two differ:')
    print(f'  at k=10 the raw share is {(s[:10] ** 2).sum() / (s ** 2).sum():.4f} and the centred one is '
          f'{(sc[:10] ** 2).sum() / (sc ** 2).sum():.4f}.')
    print('  That identity is all "explained variance" ever means, and it is also why')
    print('  the phrase is so easy to misuse: it is a statement about reconstructing the')
    print('  matrix, and nothing at all about reconstructing the thing you wanted.')

    # ------------------------------------------------------------------ 8
    header('8. t-SNE: keep the neighbours, let go of everything else')
    print("""P: in the original space, turn distances into neighbour probabilities with a
   gaussian.  Every row gets its OWN width, found by binary search so that the
   perplexity of that row hits the number you asked for.  Perplexity is 2^entropy -
   "how many neighbours does this point effectively have".

Q: in the 2-D map, use a Student-t with one degree of freedom instead.  Its tail is
   heavy, so a pair that is moderately far apart in the original space is allowed to
   be very far apart on the page.  That is what stops everything piling into the
   middle when d is large.

Then minimise KL(P || Q) by moving the points.  The embedding IS the parameter
vector: there is no function being fitted, which is why there is no transform() for
new points and why two runs are two different pictures.""")
    Xb, yb = blobs([1.0, 1.0, 1.0], [[0, 0, 0, 0, 0], [15, 0, 0, 0, 0], [0, 40, 0, 0, 0]],
                   60, seed=1)
    P, betas = joint_P(Xb, 30.0)
    sig = 1.0 / np.sqrt(2.0 * betas)
    print(f'\n  180 points, 5 columns, three blobs.  Perplexity 30 gives each row its own')
    print(f'  gaussian width: sigma runs from {sig.min():.4f} to {sig.max():.4f} across the 180 rows.')
    print(f'  P sums to {P.sum():.6f} and is symmetric to {np.abs(P - P.T).max():.3e}.')
    Pc, _ = conditional_P(pairwise_sq_dists(Xb), 30.0)
    H = -(Pc * np.log(np.maximum(Pc, 1e-300))).sum(axis=1)
    per_hit = np.exp(H)
    print(f'  The binary search stops at tol 1e-5, so the achieved perplexity runs')
    print(f'  {per_hit.min():.4f} to {per_hit.max():.4f} - near enough, but not exactly 30.')
    dd = 3.0
    g, t = math.exp(-dd * dd / 2.0), 1.0 / (1.0 + dd * dd)
    print(f'\n  Why the Student-t: at a distance of {dd:.0f} the gaussian gives {g:.5f} and the')
    print(f'  Student-t gives {t:.5f}, {t / g:.1f} times more, so a moderately far pair is')
    print('  allowed to sit far apart on the page instead of being dragged inwards.')
    Yb, loss = tsne(Xb, perplexity=30.0, n_iter=600, lr=50.0, seed=0)
    print(f'\n  600 iterations: KL {loss[0]:.4f} -> {loss[-1]:.4f}')
    print(f'  neighbours kept (10 nearest, original vs map): {neighbour_recall(Xb, Yb, 10):.4f}')
    print('  That recall number is the only claim the picture is entitled to make.')

    # ------------------------------------------------------------------ 9
    header('9. Silent failure #4: t-SNE draws clusters in pure noise')
    Xn = pure_noise(180, 10, seed=7)
    lab_raw = kmeans_small(Xn, 3, seed=0)
    s_raw = silhouette(Xn, lab_raw)
    print('  180 points of 10-dimensional gaussian noise.  There is nothing in it.\n')
    print(f'  {"perplexity":>11} {"silhouette of the raw data":>28} {"silhouette of the picture":>27}')
    sils = {}
    for per in (5, 15, 30, 50):
        Yn, _ = tsne(Xn, perplexity=float(per), n_iter=600, lr=50.0, seed=0)
        s_emb = silhouette(Yn, kmeans_small(Yn, 3, seed=0))
        sils[per] = s_emb
        print(f'  {per:>11} {s_raw:>28.4f} {s_emb:>27.4f}')
    print('\n  Silhouette near 0 means "this grouping is arbitrary".  The data scores')
    rr = sorted(v / s_raw for v in sils.values())
    print(f'  {s_raw:.4f}; the four pictures of it score {rr[0]:.2f} to {rr[-1]:.2f} times that.  This is Day 47\'s')
    print('  k-means on uniform noise, one layer further along: the method is built to')
    print('  produce groups, so it produces groups.\n')
    print('  The test that catches it costs one more run:')
    for nm, data in [('three real blobs', Xb), ('pure 10-D noise', Xn)]:
        Ys = [tsne(data, perplexity=30.0, n_iter=600, lr=50.0, seed=sd)[0] for sd in range(3)]
        cs = [pairwise_corr(Ys[a], Ys[b]) for a, b in ((0, 1), (0, 2), (1, 2))]
        print(f'    {nm:>17}: pairwise-distance correlation between three runs '
              f'{np.round(cs, 4).tolist()}, mean {np.mean(cs):.4f}')
    print('\n  Real structure survives a different random seed.  Invented structure does')
    print('  not: the same noise gives three maps that barely agree with each other.')

    # ------------------------------------------------------------------ 10
    header('10. Silent failure #5: sizes and distances on the page mean nothing')
    Xs, ys = blobs([0.5, 5.0, 0.5], [[0, 0, 0, 0, 0], [20, 0, 0, 0, 0], [0, 60, 0, 0, 0]],
                   60, seed=1)
    Ys_, _ = tsne(Xs, perplexity=30.0, n_iter=600, lr=50.0, seed=0)
    m = [ys == i for i in range(3)]
    o_sp = [spread(Xs[mi]) for mi in m]
    e_sp = [spread(Ys_[mi]) for mi in m]
    print('  Three blobs, the middle one ten times wider than the other two.\n')
    print(f'  spreads in the data : {np.round(o_sp, 3).tolist()}   widest / narrowest = {max(o_sp) / min(o_sp):.2f}')
    print(f'  spreads in the map  : {np.round(e_sp, 3).tolist()}   widest / narrowest = {max(e_sp) / min(e_sp):.2f}')
    print('\n  A cluster ten times wider comes out the same size as the others, because')
    print('  every row chose its own sigma to reach the same perplexity.  That per-row')
    print('  width is how t-SNE copes with regions of different density, and it is')
    print('  exactly why the output cannot report density.\n')
    print(f'  {"true distance ratio":>21} {"ratio on the page":>20}')
    ratios = []
    for far in (30, 60, 120, 400):
        Xf, yf = blobs([1.0, 1.0, 1.0], [[0, 0, 0, 0, 0], [15, 0, 0, 0, 0], [0, far, 0, 0, 0]],
                       60, seed=1)
        Yf, _ = tsne(Xf, perplexity=30.0, n_iter=600, lr=50.0, seed=0)
        mf = [yf == i for i in range(3)]
        ro = centre_distance(Xf, mf[0], mf[2]) / centre_distance(Xf, mf[0], mf[1])
        re = centre_distance(Yf, mf[0], mf[2]) / centre_distance(Yf, mf[0], mf[1])
        ratios.append((ro, re))
        print(f'  {ro:>21.2f} {re:>20.2f}')
    print('\n  One group is nearly 27 times further away than another and the map puts')
    print('  them at the same distance.  There is no bug here: KL(P || Q) contains no')
    print('  term that asks about the gap between two clusters, so nothing in the')
    print('  optimisation was ever trying to get that right.')

    # ------------------------------------------------------------------ 11
    header('11. Silent failure #6: the loss gets better while the map gets worse')
    print('  Same three blobs, one knob changed.\n')
    print(f'  {"perplexity":>11} {"final KL":>10} {"neighbours kept":>17}')
    per_rows = []
    for per in (2, 5, 30, 80, 150):
        Yp, lp = tsne(Xb, perplexity=float(per), n_iter=600, lr=50.0, seed=0)
        per_rows.append((per, lp[-1], neighbour_recall(Xb, Yp, 10)))
        print(f'  {per:>11} {lp[-1]:>10.4f} {per_rows[-1][2]:>17.4f}')
    best = max(per_rows, key=lambda r: r[2])
    lowest = min(per_rows, key=lambda r: r[1])
    print(f'\n  The lowest KL in the table is {lowest[1]:.4f}, at perplexity {lowest[0]}.  That looks like a')
    print(f'  perfect fit, and it keeps {lowest[2]:.4f} of the neighbours against {best[2]:.4f} at')
    print(f'  perplexity {best[0]}.  The trap is that P is rebuilt from scratch for every')
    print('  perplexity, so each row of this table is a different objective function.')
    print('  KL values are not comparable across perplexities, and the one number the')
    print('  optimiser prints is therefore useless for choosing the one knob that matters.')
    print('\n  Two more properties with no warning attached: there is no transform(), so a')
    print('  new point means re-running everything from the start, and that re-run is')
    print(f'  {len(Xb)} * {len(Xb)} pairs every iteration.  Which is the next section.')

    # ------------------------------------------------------------------ 12
    header('12. LeetCode 719 - Find K-th Smallest Pair Distance')
    print('  Both halves of this day are built on pairwise distances, and there are')
    print('  n(n-1)/2 of them.  At n = 10,000 that is 49,995,000 - the number you are')
    print('  not allowed to materialise.  This problem is that constraint, isolated.\n')
    sol = SolutionKthPairDistance()
    print(f'  nums = [1, 3, 1], k = 1 -> {sol.smallestDistancePair([1, 3, 1], 1)}   '
          f'(the distances are 0, 2, 2)')
    print(f'  nums = [1, 1, 1], k = 2 -> {sol.smallestDistancePair([1, 1, 1], 2)}')
    print(f'  nums = [1, 6, 1], k = 3 -> {sol.smallestDistancePair([1, 6, 1], 3)}')
    rng719 = np.random.default_rng(5)
    bad_lo = 0
    first = None
    for _ in range(5000):
        n = int(rng719.integers(2, 9))
        a = [int(v) for v in rng719.integers(0, 15, n)]
        k = int(rng719.integers(1, n * (n - 1) // 2 + 1))
        want = kth_pair_distance_brute(a, k)
        if kth_pair_distance_lo1(a[:], k) != want:
            bad_lo += 1
            if first is None:
                first = (sorted(a), k, kth_pair_distance_lo1(a[:], k), want)
    print(f'\n  Starting the binary search at lo = 1 instead of lo = 0 - "distances are at')
    print(f'  least 1" - differs from the truth on {bad_lo} of 5,000 random cases ({100 * bad_lo / 5000:.1f}%).')
    print(f'  It only breaks when the answer is 0, which needs a repeated value:')
    print(f'    nums = {first[0]}, k = {first[1]} -> {first[2]} instead of {first[3]}')
    print('  A handful of hand-written tests with distinct values will never see it.')

    # ------------------------------------------------------------------ 13
    header('13. LeetCode 2352 - Equal Row and Column Pairs')
    sp2 = SolutionEqualPairs()
    g1 = [[3, 2, 1], [1, 7, 6], [2, 7, 7]]
    g2 = [[3, 1, 2, 2], [1, 4, 4, 5], [2, 4, 2, 2], [2, 4, 2, 2]]
    print(f'  {g1} -> {sp2.equalPairs(g1)}')
    print(f'  {g2} -> {sp2.equalPairs(g2)}')
    print('\n  A row and a column are equal when they are the same sequence of numbers,')
    print('  so the key has to be the sequence.  A tuple keeps the element boundaries;')
    print('  gluing the digits into one string throws them away.')
    g3 = [[1, 2, 34], [23, 5, 6], [4, 7, 8]]
    print(f'\n  {g3}')
    print(f'    row 0 is {tuple(g3[0])} and column 0 is {tuple(r[0] for r in g3)} - different sequences')
    print(f"    joined into a string both become '{''.join(map(str, g3[0]))}'")
    print(f'    correct answer {sp2.equalPairs(g3)}, brute force {equal_pairs_brute(g3)}, '
          f'joined-string key {equal_pairs_joined(g3)}')
    from collections import Counter
    pairs = [(a, b) for a in range(1, 100) for b in range(1, 100)]
    keys = Counter(''.join(map(str, s)) for s in pairs)
    shared = [k for k, v in keys.items() if v > 1]
    assert all(keys[k] == 2 for k in shared)   # every clash here is an exact pair
    clash = len(shared)
    nonuniq = sum(keys[k] for k in shared)
    rng2352 = np.random.default_rng(11)
    hits = 0
    for _ in range(200000):
        a = tuple(int(v) for v in rng2352.integers(1, 100000, 3))
        b = tuple(int(v) for v in rng2352.integers(1, 100000, 3))
        hits += a != b and ''.join(map(str, a)) == ''.join(map(str, b))
    print(f'\n  How big is the bug?  Of the {len(pairs)} two-element sequences with values in')
    print(f'  1..99, only {len(keys)} distinct joined keys exist.  {clash} keys are shared,')
    print(f'  each by exactly two sequences, so {nonuniq} of the {len(pairs)} sequences')
    print(f'  ({100 * nonuniq / len(pairs):.1f}%) are not uniquely keyed.  How likely is a random test to')
    print(f'  find it?  Two independently drawn 3-element sequences with values up to')
    print(f'  100,000 collided {hits} times in 200,000 draws.  Fifteen percent of the input')
    print('  space, and essentially zero chance of stumbling into it - which is the exact')
    print('  shape of a bug that reaches production.')

    # ------------------------------------------------------------------ 14
    header('14. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
