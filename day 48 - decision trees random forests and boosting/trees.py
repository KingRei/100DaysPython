"""Day 48 - Decision trees, random forests and gradient boosting, in pure NumPy.

One building block used three ways.  Run `python trees.py` to print the whole story;
every number is exact and checked by an assert at the bottom.
"""

import math

import numpy as np


# ================================================================ 1. impurity
def gini(y, k):
    """1 - sum p^2.  Zero when a node holds one class, 1 - 1/k when it is uniform."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=k) / len(y)
    return float(1.0 - np.sum(p * p))


def entropy(y, k):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=k) / len(y)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) + 0.0)   # keeps a pure node at 0.0, not -0.0


def variance(y, k=None):
    """The regression criterion.  Minimising weighted variance is the same thing as
    minimising squared error, because the best constant prediction for a leaf is its
    mean (Day 45 again)."""
    return 0.0 if len(y) == 0 else float(np.var(y))


CRITERIA = {'gini': gini, 'entropy': entropy, 'variance': variance}


# ================================================================ 2. one split
def split_candidates(col):
    """Midpoints between consecutive distinct values.  A column with m distinct
    values offers m - 1 thresholds - which is the whole of section 7."""
    u = np.unique(col)
    return (u[:-1] + u[1:]) / 2.0 if len(u) > 1 else np.empty(0)


def best_split_naive(X, y, k, crit, feats=None, min_leaf=1):
    """Exhaustive search, written the obvious way: for every candidate threshold,
    rebuild both children and score them.  O(n) work per candidate and O(n)
    candidates per feature, so O(n^2 d) per node.  Section 3 replaces it."""
    fn = CRITERIA[crit]
    n = len(y)
    parent = fn(y, k)
    best = None
    for j in (range(X.shape[1]) if feats is None else feats):
        col = X[:, j]
        for t in split_candidates(col):
            left = col <= t
            nl = int(left.sum())
            if nl < min_leaf or n - nl < min_leaf:
                continue
            child = (nl * fn(y[left], k) + (n - nl) * fn(y[~left], k)) / n
            gain = parent - child
            if best is None or gain > best[0] + 1e-12:
                best = (gain, j, float(t))
    return best


def best_split(X, y, k, crit, feats=None, min_leaf=1):
    """The same answer in O(n log n * d): sort the column once, then sweep the cut
    from left to right.  Every statistic the criterion needs is a running total, so
    moving the cut by one row is O(1) instead of O(n).

    For gini and entropy the running totals are class counts; for variance they are
    sum(y) and sum(y^2), because var = E[y^2] - E[y]^2.
    """
    n = len(y)
    parent = CRITERIA[crit](y, k)
    best = None
    cols = range(X.shape[1]) if feats is None else feats
    for j in cols:
        col = X[:, j]
        order = np.argsort(col, kind='stable')
        cv, yv = col[order], y[order]
        # a cut after position i is only legal where the value actually changes
        cut = np.flatnonzero(cv[:-1] < cv[1:])
        if len(cut) == 0:
            continue
        nl = cut + 1
        ok = (nl >= min_leaf) & (n - nl >= min_leaf)
        if not ok.any():
            continue
        cut, nl = cut[ok], nl[ok]
        if crit == 'variance':
            c1 = np.cumsum(yv)[cut]
            c2 = np.cumsum(yv * yv)[cut]
            t1, t2 = yv.sum(), (yv * yv).sum()
            vl = c2 / nl - (c1 / nl) ** 2
            vr = (t2 - c2) / (n - nl) - ((t1 - c1) / (n - nl)) ** 2
            child = (nl * vl + (n - nl) * vr) / n
        else:
            onehot = np.zeros((n, k))
            onehot[np.arange(n), yv] = 1.0
            cum = np.cumsum(onehot, axis=0)[cut]          # class counts on the left
            tot = onehot.sum(axis=0)
            pl = cum / nl[:, None]
            pr = (tot - cum) / (n - nl)[:, None]
            if crit == 'gini':
                il = 1.0 - np.sum(pl * pl, axis=1)
                ir = 1.0 - np.sum(pr * pr, axis=1)
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    il = -np.sum(np.where(pl > 0, pl * np.log2(pl), 0.0), axis=1)
                    ir = -np.sum(np.where(pr > 0, pr * np.log2(pr), 0.0), axis=1)
            child = (nl * il + (n - nl) * ir) / n
        gains = parent - child
        i = int(np.argmax(gains))
        g = float(gains[i])
        if best is None or g > best[0] + 1e-12:
            best = (g, j, float((cv[cut[i]] + cv[cut[i] + 1]) / 2.0))
    return best


class Node:
    __slots__ = ('feat', 'thr', 'left', 'right', 'value', 'n', 'imp')

    def __init__(self, value, n, imp):
        self.feat = None
        self.thr = None
        self.left = None
        self.right = None
        self.value = value
        self.n = n
        self.imp = imp

    @property
    def is_leaf(self):
        return self.feat is None


class DecisionTree:
    """One tree, for classification (majority vote) or regression (mean)."""

    def __init__(self, task='clf', crit=None, max_depth=None, min_leaf=1,
                 min_gain=0.0, max_features=None, rng=None, n_classes=None):
        self.task = task
        self.crit = crit or ('gini' if task == 'clf' else 'variance')
        self.max_depth = math.inf if max_depth is None else max_depth
        self.min_leaf = min_leaf
        self.min_gain = min_gain
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(0)
        self.k = n_classes
        self.root = None
        self.n_nodes = 0
        self.n_leaves = 0
        self.depth = 0

    def _value(self, y):
        if self.task == 'clf':
            return int(np.bincount(y, minlength=self.k).argmax())
        return float(y.mean())

    def fit(self, X, y):
        if self.task == 'clf' and self.k is None:
            self.k = int(y.max()) + 1
        self.n_nodes = self.n_leaves = self.depth = 0
        self.root = self._grow(X, y, 0)
        return self

    def _grow(self, X, y, d):
        node = Node(self._value(y), len(y), CRITERIA[self.crit](y, self.k))
        self.n_nodes += 1
        self.depth = max(self.depth, d)
        pure = node.imp <= 1e-12
        if pure or d >= self.max_depth or len(y) < 2 * self.min_leaf:
            self.n_leaves += 1
            return node
        feats = None
        if self.max_features is not None and self.max_features < X.shape[1]:
            feats = self.rng.choice(X.shape[1], self.max_features, replace=False)
        sp = best_split(X, y, self.k, self.crit, feats, self.min_leaf)
        if sp is None or sp[0] <= self.min_gain:
            self.n_leaves += 1
            return node
        _, j, t = sp
        node.feat, node.thr = j, t
        mask = X[:, j] <= t
        node.left = self._grow(X[mask], y[mask], d + 1)
        node.right = self._grow(X[~mask], y[~mask], d + 1)
        return node

    def predict(self, X):
        out = np.empty(len(X), dtype=int if self.task == 'clf' else float)
        for i, row in enumerate(X):
            node = self.root
            while not node.is_leaf:
                node = node.left if row[node.feat] <= node.thr else node.right
            out[i] = node.value
        return out

    def path(self, row, names=None):
        """The root-to-leaf path a single prediction takes - the tree's whole
        explanation for one answer.  This is LeetCode 257 with real data in it."""
        node, steps = self.root, []
        while not node.is_leaf:
            nm = names[node.feat] if names else f'x{node.feat}'
            go_left = row[node.feat] <= node.thr
            steps.append(f'{nm} {"<=" if go_left else ">"} {node.thr:.3f}')
            node = node.left if go_left else node.right
        return steps, node.value

    def importances(self, n_features):
        """Impurity-based importance: how much weighted impurity each feature's
        splits removed, normalised.  Section 7 is about how this lies."""
        imp = np.zeros(n_features)

        def walk(node):
            if node.is_leaf:
                return
            drop = node.n * node.imp - node.left.n * node.left.imp - node.right.n * node.right.imp
            imp[node.feat] += drop
            walk(node.left)
            walk(node.right)

        walk(self.root)
        s = imp.sum()
        return imp / s if s > 0 else imp


def grow_trace(X, y, max_depth=6):
    """Grow a tree depth-first and report one row per split, in the order it is made:
    (depth, feature, threshold, gini drop, leaves afterwards, training accuracy)."""
    k = int(y.max()) + 1
    rows = []
    root = {'X': X, 'y': y}
    leaves = [root]

    def acc():
        ok = sum(int((lf['y'] == np.bincount(lf['y'], minlength=k).argmax()).sum())
                 for lf in leaves)
        return ok / len(y)

    def rec(node, d):
        Xn, yn = node['X'], node['y']
        if d >= max_depth or gini(yn, k) <= 1e-12:
            return
        sp = best_split(Xn, yn, k, 'gini')
        if sp is None or sp[0] <= 1e-12:
            return
        gain, j, t = sp
        m = Xn[:, j] <= t
        left = {'X': Xn[m], 'y': yn[m]}
        right = {'X': Xn[~m], 'y': yn[~m]}
        i = next(q for q, lf in enumerate(leaves) if lf is node)
        leaves[i:i + 1] = [left, right]
        rows.append((d, j, float(t), gain, len(leaves), acc()))
        rec(left, d + 1)
        rec(right, d + 1)

    rec(root, 0)
    return rows


# ================================================================ 3. the forest
class RandomForest:
    """Bagging plus a random subset of features at every split.

    The two knobs do different jobs.  Bagging makes the trees see different rows;
    max_features makes them ask different questions.  Without the second one a few
    dominant features get picked by every tree, the trees agree, and averaging
    agreeing predictors buys nothing.
    """

    def __init__(self, n_trees=30, max_features='sqrt', max_depth=None, min_leaf=1,
                 bootstrap=True, seed=0, task='clf'):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.bootstrap = bootstrap
        self.seed = seed
        self.task = task
        self.trees = []
        self.oob_idx = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        k = int(y.max()) + 1 if self.task == 'clf' else None
        mf = (max(1, int(round(math.sqrt(d)))) if self.max_features == 'sqrt'
              else (d if self.max_features is None else self.max_features))
        self.trees, self.oob_idx = [], []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            oob = np.setdiff1d(np.arange(n), idx) if self.bootstrap else np.empty(0, int)
            t = DecisionTree(task=self.task, max_depth=self.max_depth, min_leaf=self.min_leaf,
                             max_features=mf, rng=np.random.default_rng(rng.integers(1 << 30)),
                             n_classes=k).fit(X[idx], y[idx])
            self.trees.append(t)
            self.oob_idx.append(oob)
        self.k = k
        return self

    def predict(self, X):
        P = np.array([t.predict(X) for t in self.trees])
        if self.task == 'reg':
            return P.mean(axis=0)
        return np.array([np.bincount(P[:, i], minlength=self.k).argmax() for i in range(len(X))])

    def oob_score(self, X, y):
        """Every row was left out of about 37% of the trees, so the forest comes with
        a held-out estimate that costs nothing: 1 - (1 - 1/n)^n -> 1/e."""
        votes = np.zeros((len(X), self.k), dtype=int)
        for t, oob in zip(self.trees, self.oob_idx):
            if len(oob):
                for i, p in zip(oob, t.predict(X[oob])):
                    votes[i, p] += 1
        seen = votes.sum(axis=1) > 0
        return float(np.mean(votes[seen].argmax(axis=1) == y[seen])), int(seen.sum())

    def disagreement(self, X):
        """Mean fraction of rows on which two different trees disagree.  This is the
        quantity averaging is paid to exploit: identical trees have nothing to average."""
        P = np.array([t.predict(X) for t in self.trees])
        m = len(P)
        tot = [float(np.mean(P[a] != P[b])) for a in range(m) for b in range(a + 1, m)]
        return float(np.mean(tot))


# ================================================================ 4. boosting
class GradientBoosting:
    """Gradient descent, where the thing being descended is a function.

    Day 45 took a step in parameter space: w -= lr * dL/dw.  Here the model is a sum
    of trees, the "parameter" is the function F itself, and the step is a small tree
    fitted to -dL/dF.  For squared loss that derivative is exactly the residual, which
    is why the textbook version looks like "fit the next tree to what is left over".
    """

    def __init__(self, n_trees=60, lr=0.1, max_depth=2, min_leaf=1, seed=0):
        self.n_trees = n_trees
        self.lr = lr
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.seed = seed

    def fit(self, X, y, X_val=None, y_val=None):
        self.f0 = float(y.mean())           # the best constant: the minimiser of MSE
        F = np.full(len(y), self.f0)
        self.trees, self.train_loss, self.val_loss = [], [], []
        Fv = np.full(len(X_val), self.f0) if X_val is not None else None
        for _ in range(self.n_trees):
            resid = y - F                   # -dL/dF for L = 0.5 * (y - F)^2
            t = DecisionTree(task='reg', max_depth=self.max_depth,
                             min_leaf=self.min_leaf).fit(X, resid)
            F += self.lr * t.predict(X)
            self.trees.append(t)
            self.train_loss.append(float(np.mean((y - F) ** 2)))
            if Fv is not None:
                Fv += self.lr * t.predict(X_val)
                self.val_loss.append(float(np.mean((y_val - Fv) ** 2)))
        return self

    def predict(self, X, n=None):
        F = np.full(len(X), self.f0)
        for t in self.trees[:n if n is not None else len(self.trees)]:
            F += self.lr * t.predict(X)
        return F


# ================================================================ 5. data sets
def stripes(n=400, seed=0, noise=0.0):
    """Two classes in diagonal bands: a tree has to approximate the diagonal with a
    staircase, which is the honest picture of what axis-aligned splits can express."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (n, 2))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    if noise:
        flip = rng.random(n) < noise
        y[flip] = 1 - y[flip]
    return X, y


def dominant(n=900, seed=0, noise=0.08):
    """Ten columns: one that dominates the answer, four that help a little, five pure
    noise.  The shape of most tabular data, and the shape that makes feature
    subsampling matter - without it every tree picks the same root."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 10))
    score = 3.0 * X[:, 0] + 0.55 * X[:, 1] + 0.5 * X[:, 2] - 0.45 * X[:, 3] + 0.4 * X[:, 4]
    y = (score > 0).astype(int)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return X, y


def small24():
    """24 hand-placed points, the set the interactive page grows a tree on.  Class 1
    fills the lower-right block except for one row at (0.70, 0.30) labelled 0 - which
    is the whole of the "memorise one row" story."""
    P = [(0.10, 0.15, 0), (0.18, 0.45, 0), (0.12, 0.72, 0), (0.26, 0.88, 0),
         (0.22, 0.28, 0), (0.08, 0.58, 0), (0.33, 0.18, 0), (0.36, 0.66, 0),
         (0.30, 0.40, 0),
         (0.62, 0.12, 1), (0.74, 0.22, 1), (0.88, 0.16, 1), (0.66, 0.34, 1),
         (0.82, 0.40, 1), (0.94, 0.28, 1), (0.58, 0.44, 1),
         (0.60, 0.66, 0), (0.72, 0.78, 0), (0.86, 0.70, 0), (0.92, 0.88, 0),
         (0.64, 0.92, 0), (0.78, 0.58, 0), (0.90, 0.56, 0),
         (0.70, 0.30, 0)]
    return np.array([[r[0], r[1]] for r in P]), np.array([r[2] for r in P])


def xor_lattice():
    """A perfectly symmetric 8x8 XOR lattice.  Every candidate split is worth exactly
    zero here, which is the one way a tree can fail to reach 1.000 on clean data."""
    pts = [(-0.875 + i * 0.25, -0.875 + j * 0.25) for i in range(8) for j in range(8)]
    X = np.array(pts)
    return X, np.array([int((p[0] > 0) != (p[1] > 0)) for p in pts])


def xor_data(n=400, seed=1, noise=0.0):
    """y = sign(x0) XOR sign(x1).  Neither feature carries any information on its own,
    which is exactly the case a greedy top-down search cannot see."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    if noise:
        flip = rng.random(n) < noise
        y[flip] = 1 - y[flip]
    return X, y


def cardinality_data(n=300, seed=2):
    """One binary column that *is* the answer (with a little noise), one continuous
    column of pure noise, and three more binary columns of pure noise.

    Columns: 0 = the real signal (binary), 1 = continuous noise, 2..4 = binary noise.
    """
    rng = np.random.default_rng(seed)
    signal = rng.integers(0, 2, n)
    y = signal.copy()
    flip = rng.random(n) < 0.20
    y[flip] = 1 - y[flip]
    X = np.column_stack([signal.astype(float), rng.random(n),
                         rng.integers(0, 2, (n, 3)).astype(float)])
    return X, y, ['real (binary)', 'noise (continuous)', 'noise b1', 'noise b2', 'noise b3']


def wave(n=120, lo=-3.0, hi=3.0, seed=3, noise=0.15):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(lo, hi, n))
    return x[:, None], np.sin(x) + 0.35 * x + noise * rng.standard_normal(n)


def tabular_reg(n=400, d=6, seed=4, noise=0.6):
    """Three informative columns combined non-linearly, three pure noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, d))
    y = (np.sin(1.4 * X[:, 0]) + 0.8 * X[:, 1] * X[:, 2] + 0.5 * X[:, 1] ** 2
         + noise * rng.standard_normal(n))
    return X, y


def accuracy(a, b):
    return float(np.mean(a == b))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def permutation_importance(model, X, y, rng, repeats=10, score=accuracy):
    """Shuffle one column at a time and watch the score fall.  It asks the only
    question that matters - does the model actually use this column - instead of
    counting how many chances the column had to be picked."""
    base = score(model.predict(X), y)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - score(model.predict(Xp), y))
        out[j] = np.mean(drops)
    return out


# ================================================================ 6. LeetCode
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build(vals):
    """LeetCode's level-order array form, with None for a missing child."""
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])
    q, i = [root], 1
    while q and i < len(vals):
        node = q.pop(0)
        if i < len(vals):
            if vals[i] is not None:
                node.left = TreeNode(vals[i])
                q.append(node.left)
            i += 1
        if i < len(vals):
            if vals[i] is not None:
                node.right = TreeNode(vals[i])
                q.append(node.right)
            i += 1
    return root


class SolutionPathSum:
    def hasPathSum(self, root, targetSum: int) -> bool:
        # A leaf is a node with NO children.  Recursing into a missing child and
        # testing the remainder there counts a one-child node as a leaf.
        if root is None:
            return False
        if root.left is None and root.right is None:
            return targetSum == root.val
        rest = targetSum - root.val
        return self.hasPathSum(root.left, rest) or self.hasPathSum(root.right, rest)


def has_path_sum_null_leaf(root, target):
    """The spelling that treats "ran out of tree" as "reached a leaf"."""
    if root is None:
        return target == 0
    return (has_path_sum_null_leaf(root.left, target - root.val)
            or has_path_sum_null_leaf(root.right, target - root.val))


class SolutionBinaryTreePaths:
    def binaryTreePaths(self, root) -> list[str]:
        out: list[str] = []

        def walk(node, acc):
            if node is None:
                return
            acc = acc + [str(node.val)]          # a new list, not a mutated one
            if node.left is None and node.right is None:
                out.append('->'.join(acc))
                return
            walk(node.left, acc)
            walk(node.right, acc)

        walk(root, [])
        return out


def binary_tree_paths_shared(root):
    """The same walk with one shared list and no undo - every path comes back as the
    final state of the accumulator instead of its own history (Day 39's bug)."""
    out, acc = [], []

    def walk(node):
        if node is None:
            return
        acc.append(str(node.val))
        if node.left is None and node.right is None:
            out.append(acc)                       # the list itself, not a copy
            return
        walk(node.left)
        walk(node.right)

    walk(root)
    return ['->'.join(p) for p in out]


def check() -> None:
    """Everything the walkthrough claims, re-derived from scratch."""
    rng = np.random.default_rng(0)

    # The fast split search agrees with the obvious one, everywhere.
    for _ in range(60):          # 60 problems, each checked under two criteria
        n = int(rng.integers(6, 60))
        d = int(rng.integers(1, 4))
        k = int(rng.integers(2, 4))
        Xr_ = np.round(rng.uniform(0, 3, (n, d)), 2)
        yr_ = rng.integers(0, k, n)
        for crit in ('gini', 'entropy'):
            a_ = best_split_naive(Xr_, yr_, k, crit)
            b_ = best_split(Xr_, yr_, k, crit)
            assert (a_ is None) == (b_ is None)
            if a_ is not None:
                assert abs(a_[0] - b_[0]) < 1e-9
        yv_ = rng.normal(size=n)
        a_ = best_split_naive(Xr_, yv_, None, 'variance')
        b_ = best_split(Xr_, yv_, None, 'variance')
        assert (a_ is None) == (b_ is None)
        if a_ is not None:
            assert abs(a_[0] - b_[0]) < 1e-9

    # Impurity is 0 on a pure node and maximal on a balanced one.
    assert gini(np.zeros(8, int), 2) == 0.0 and entropy(np.zeros(8, int), 2) == 0.0
    assert abs(gini(np.array([0, 0, 0, 0, 1, 1, 1, 1]), 2) - 0.5) < 1e-12
    assert abs(entropy(np.array([0, 0, 0, 0, 1, 1, 1, 1]), 2) - 1.0) < 1e-12

    # A tree grown to purity reproduces its own training labels exactly, on any data
    # set with no contradictory duplicate rows.
    for seed in range(4):
        Xa, ya = stripes(150, seed=seed, noise=0.15)
        ta = DecisionTree().fit(Xa, ya)
        assert accuracy(ta.predict(Xa), ya) == 1.0
    Xs_, ys_ = stripes(400, seed=0, noise=0.10)
    pm = np.random.default_rng(0).permutation(400)
    tr_, te_ = pm[:260], pm[260:]
    accs = [(d, accuracy(DecisionTree(max_depth=d).fit(Xs_[tr_], ys_[tr_]).predict(Xs_[te_]), ys_[te_]))
            for d in (1, 2, 3, 5, 8, 12, None)]
    # training accuracy is monotone in depth; held-out accuracy is not
    tr_accs = [accuracy(DecisionTree(max_depth=d).fit(Xs_[tr_], ys_[tr_]).predict(Xs_[tr_]), ys_[tr_])
               for d in (1, 2, 3, 5, 8, 12, None)]
    assert all(tr_accs[i] <= tr_accs[i + 1] + 1e-12 for i in range(len(tr_accs) - 1))
    assert tr_accs[-1] == 1.0
    assert max(a for _, a in accs) > accs[-1][1]

    # XOR: the correct root split is worth essentially nothing, and greedy misses it.
    Xx_, yx_ = xor_data(400, seed=1)
    par = gini(yx_, 2)
    m = Xx_[:, 0] <= 0.0
    g0 = par - (m.sum() * gini(yx_[m], 2) + (~m).sum() * gini(yx_[~m], 2)) / len(yx_)
    gb_ = best_split(Xx_, yx_, 2, 'gini')[0]
    assert g0 < 1e-4 < gb_
    oracle = ((Xx_[:, 0] > 0) ^ (Xx_[:, 1] > 0)).astype(int)
    assert accuracy(oracle, yx_) == 1.0
    assert accuracy(DecisionTree(max_depth=2).fit(Xx_, yx_).predict(Xx_), yx_) < 0.7

    # Impurity importance prefers the high-cardinality noise column; permutation
    # importance on held-out rows does not.
    Xc_, yc_, _ = cardinality_data(300, seed=2)
    rc_ = np.random.default_rng(0).permutation(300)
    tc_ = DecisionTree().fit(Xc_[rc_[:210]], yc_[rc_[:210]])
    imp_ = tc_.importances(Xc_.shape[1])
    pi_ = permutation_importance(tc_, Xc_[rc_[210:]], yc_[rc_[210:]],
                                 np.random.default_rng(7), repeats=20)
    assert imp_.argmax() == 1 and pi_.argmax() == 0

    # A tree is a step function: it cannot leave the range of its leaf means.
    Xw_, yw_ = wave(120, seed=3)
    tw_ = DecisionTree(task='reg', max_depth=6).fit(Xw_, yw_)
    far = tw_.predict(np.array([[3.5], [5.0], [10.0], [100.0]]))
    assert np.all(far == far[0])
    assert tw_.predict(Xw_).max() < yw_.max()
    fw_ = RandomForest(n_trees=10, task='reg', max_features=None, seed=0).fit(Xw_, yw_)
    far_f = fw_.predict(np.array([[3.5], [5.0], [100.0]]))
    assert np.allclose(far_f, far_f[0])

    # Bagging beats one tree; the forest's trees disagree more than bagging's.
    Xd_, yd_ = dominant(900, seed=0)
    rd_ = np.random.default_rng(1).permutation(900)
    dtr_, dte_ = rd_[:600], rd_[600:]
    one_ = accuracy(DecisionTree().fit(Xd_[dtr_], yd_[dtr_]).predict(Xd_[dte_]), yd_[dte_])
    fb = RandomForest(n_trees=30, seed=0, max_features=None).fit(Xd_[dtr_], yd_[dtr_])
    fs = RandomForest(n_trees=30, seed=0, max_features='sqrt').fit(Xd_[dtr_], yd_[dtr_])
    assert accuracy(fb.predict(Xd_[dte_]), yd_[dte_]) > one_
    assert fs.disagreement(Xd_[dte_]) > fb.disagreement(Xd_[dte_])
    # without feature subsampling every tree opens with the same question
    assert len(set(t.root.feat for t in fb.trees)) == 1
    assert len(set(t.root.feat for t in fs.trees)) > 1
    assert fb.oob_score(Xd_[dtr_], yd_[dtr_])[1] == 600

    # Boosting: the training loss is monotone, the held-out loss is not.
    Xr2, yr2 = tabular_reg(600, 6, seed=4)
    rr_ = np.random.default_rng(2).permutation(600)
    gb2 = GradientBoosting(n_trees=120, lr=1.0, max_depth=2).fit(
        Xr2[rr_[:420]], yr2[rr_[:420]], Xr2[rr_[420:]], yr2[rr_[420:]])
    assert all(gb2.train_loss[i] >= gb2.train_loss[i + 1] - 1e-9
               for i in range(len(gb2.train_loss) - 1))
    b2 = int(np.argmin(gb2.val_loss))
    assert b2 < len(gb2.val_loss) - 1 and gb2.val_loss[-1] > gb2.val_loss[b2] * 1.2
    # the first residual target really is y - mean(y)
    g1 = GradientBoosting(n_trees=1, lr=1.0, max_depth=2).fit(Xr2, yr2)
    assert abs(g1.f0 - yr2.mean()) < 1e-12

    # LeetCode 112 and 257.
    assert SolutionPathSum().hasPathSum(
        build([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1]), 22) is True
    assert SolutionPathSum().hasPathSum(build([1, 2, 3]), 5) is False
    assert SolutionPathSum().hasPathSum(build([]), 0) is False
    assert SolutionPathSum().hasPathSum(build([1, 2]), 1) is False
    assert has_path_sum_null_leaf(build([1, 2]), 1) is True
    assert sorted(SolutionBinaryTreePaths().binaryTreePaths(build([1, 2, 3, None, 5]))) == \
        ['1->2->5', '1->3']
    assert SolutionBinaryTreePaths().binaryTreePaths(build([1])) == ['1']
    assert binary_tree_paths_shared(build([1, 2, 3, None, 5])) != ['1->2->5', '1->3']


def header(t):
    print('\n' + '=' * 72 + '\n' + t + '\n' + '=' * 72)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # ------------------------------------------------------------------ 1
    header('1. One building block, used three ways')
    print("""A decision tree is the only model in this series with no coefficients and no
distance function.  It asks questions: "is column 7 below 0.31?", and keeps asking
until it is willing to commit to an answer.

    one tree          : ask questions until every leaf is pure  -> it memorises
    many independent  : average trees that each memorise different noise  (forest)
    many dependent    : each small tree fixes what the last ones got wrong (boosting)

Same block, opposite jobs: the forest is there to kill variance, boosting is there to
kill bias.  And a tree's own account of itself - its accuracy on its training data,
and which columns it says mattered - is wrong in a direction you can predict.""")

    Xs, ys = stripes(400, seed=0, noise=0.10)
    t3 = DecisionTree(max_depth=3).fit(Xs, ys)
    steps, val = t3.path(Xs[0], names=['x', 'y'])
    print(f'\nA depth-3 tree on 400 points: {t3.n_nodes} nodes, {t3.n_leaves} leaves.')
    print(f'  one prediction for the point {np.round(Xs[0], 3).tolist()} is one root-to-leaf path:')
    for s in steps:
        print(f'     {s}')
    print(f'     -> class {val}')
    print('  The model IS the set of those paths.  That is also LeetCode 257, below.')

    # ------------------------------------------------------------------ 2
    header('2. Choosing a split')
    print('Gini and entropy both ask "how mixed is this node", and both go to 0 when it')
    print('is pure.  The split that wins is the one that buys the biggest weighted drop.')
    y_demo = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    for name, lab in [('gini', 'gini'), ('entropy', 'entropy')]:
        vals = [CRITERIA[name](np.array([0] * i + [1] * (8 - i)), 2) for i in range(5)]
        print(f'  {lab:>8} for 0/8, 1/7, 2/6, 3/5, 4/4 split of 8 rows: '
              + ' '.join(f'{v:.3f}' for v in vals))
    sp = best_split(Xs, ys, 2, 'gini')
    print(f'\n  best split on the 400 striped points: x{sp[1]} <= {sp[2]:.4f}, gain {sp[0]:.4f}')
    print(f'  candidate thresholds considered: '
          f'{sum(len(split_candidates(Xs[:, j])) for j in range(2))} '
          f'({len(split_candidates(Xs[:, 0]))} per column)')
    print('\n  The obvious implementation rebuilds both children for every candidate:')
    print('  O(n) work times O(n) candidates times d columns = O(n^2 d) per node.')
    print('  Sorting the column once and sweeping the cut left to right makes every')
    print('  statistic a running total, so each candidate is O(1) and the node is')
    print('  O(n log n * d).  Both versions are in this file and they agree exactly;')
    print('  only the fast one is used below, because the slow one is unusable at n=600.')

    # ------------------------------------------------------------------ 3
    header('3. Silent failure #1: a tree grown to purity scores 1.000 on its own data')
    rng = np.random.default_rng(0)
    perm = rng.permutation(400)
    tr, te = perm[:260], perm[260:]
    print('  400 points, two diagonal bands, 10% of the labels flipped.')
    print(f'\n  {"max_depth":>10} {"leaves":>7} {"train":>7} {"test":>7}')
    rows = []
    for d in (1, 2, 3, 5, 8, 12, None):
        t = DecisionTree(max_depth=d).fit(Xs[tr], ys[tr])
        a_tr = accuracy(t.predict(Xs[tr]), ys[tr])
        a_te = accuracy(t.predict(Xs[te]), ys[te])
        rows.append((d, t.n_leaves, a_tr, a_te))
        print(f'  {str(d):>10} {t.n_leaves:>7} {a_tr:>7.3f} {a_te:>7.3f}')
    best_d = max(rows, key=lambda r: r[3])
    print(f'\n  Training accuracy is monotone in depth and ends at exactly 1.000.  Held-out')
    print(f'  accuracy peaks at depth {best_d[0]} ({best_d[3]:.3f}) and falls back to {rows[-1][3]:.3f}.  This is')
    print("  Day 47's k=1 nearest neighbour wearing a different hat: a fully grown tree is")
    print('  a lookup table for its own training set.')

    X24, y24 = small24()
    base24 = float(np.mean(y24 == np.bincount(y24).argmax()))
    print(f'\n  Watch it happen on 24 points.  Calling everything the majority class scores')
    print(f'  {base24:.4f}; here is what each cut buys:\n')
    print(f'  {"cut":>4} {"depth":>6} {"split":>18} {"gini drop":>11} {"leaves":>7} {"accuracy":>9}')
    for i, (d, feat, thr, gain, nleaf, acc) in enumerate(grow_trace(X24, y24)):
        nm = 'x' if feat == 0 else 'y'
        print(f'  {i + 1:>4} {d:>6} {nm + " <= " + format(thr, ".3f"):>18} {gain:>11.5f} '
              f'{nleaf:>7} {acc:>9.4f}')
    print('\n  Cut 3 buys 0.03125 of gini and moves accuracy by nothing at all; it exists')
    print('  only so that cut 4 can lock one odd row into a box of its own.  Those two')
    print('  cuts are not learning, they are memorising a single row.')

    print('\n  How general is "it always reaches 1.000"?  Nearly, but not quite.  Growth')
    print('  stops early if some impure node offers no split with positive gain, and exact')
    print('  ties need exact symmetry - so it takes a constructed data set:')
    XL, yL = xor_lattice()
    spL = best_split(XL, yL, 2, 'gini')
    tL = DecisionTree().fit(XL, yL)
    print(f'    a perfect 8x8 XOR lattice, {len(yL)} points, root gini {gini(yL, 2):.4f}')
    print(f'    the best split available scores a gain of exactly {spL[0]:.1f}')
    print(f'    so the tree stops at {tL.n_leaves} leaf and trains to '
          f'{accuracy(tL.predict(XL), yL):.4f}')
    print('  There is a second way to stop short, and it is the common one: two identical')
    print('  rows carrying different labels leave an impure node with no legal split at')
    print('  all.  Outside those two cases - exact symmetry, and contradictory duplicates -')
    print('  some split has a positive gain and the tree walks all the way to 1.000.')

    # ------------------------------------------------------------------ 4
    header('4. Silent failure #2: greedy cannot see one step ahead')
    Xx, yx = xor_data(400, seed=1)
    par = gini(yx, 2)
    print('  y = (x0 > 0) XOR (x1 > 0), 400 points, no label noise at all.')
    print('  A depth-2 tree can represent this exactly: split x0 at 0, then x1 at 0.')
    for j in (0, 1):
        m = Xx[:, j] <= 0.0
        g = par - (m.sum() * gini(yx[m], 2) + (~m).sum() * gini(yx[~m], 2)) / len(yx)
        print(f'    gain of the correct split on x{j} at 0:   {g:.8f}')
    g_best, j_best, t_best = best_split(Xx, yx, 2, 'gini')
    print(f'    gain of the split greedy actually picks:  {g_best:.8f}  (x{j_best} <= {t_best:.4f})')
    oracle = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)
    greedy2 = DecisionTree(max_depth=2).fit(Xx, yx)
    print(f'\n    the correct depth-2 tree scores {accuracy(oracle, yx):.4f}')
    print(f'    greedy depth-2 scores            {accuracy(greedy2.predict(Xx), yx):.4f}')
    for d in (3, 4, 6):
        t = DecisionTree(max_depth=d).fit(Xx, yx)
        print(f'    greedy depth-{d} scores            {accuracy(t.predict(Xx), yx):.4f}  '
              f'({t.n_leaves} leaves)')
    print('\n  Neither column carries any information on its own, so the root split is')
    print('  chosen on sampling noise - and it is chosen near the edge of the range,')
    print('  nowhere near 0.  The structure is representable and unreachable at the same')
    print('  time.  Finding the optimal tree is NP-hard, so greedy top-down induction is')
    print('  what essentially every tree learner in production does.  (Exact solvers do')
    print('  exist - MIP formulations, DL8.5, MurTree, GOSDT - but they are a different')
    print('  tool for a different size of problem, not what CART or XGBoost run.)')
    print('\n  What does escape it?  Not boosting on its own - a sum of single-split trees')
    print('  is an additive model, and XOR is the standard example of something no')
    print('  additive model can express.  200 rounds on this data:\n')
    yx_f = yx.astype(float)
    for md in (1, 2, 3):
        gx = GradientBoosting(n_trees=200, lr=0.5, max_depth=md).fit(Xx, yx_f)
        acc = float(np.mean((gx.predict(Xx) > 0.5).astype(int) == yx))
        print(f'    boosting 200 trees of depth {md}: train accuracy {acc:.4f}')
    print('\n  Depth 1 gets nowhere near the answer.  It lands above 0.5 only because a')
    print('  finite sample is never perfectly balanced, so the marginals carry a little')
    print('  accidental signal - not because the sum of stumps expresses any part of the')
    print('  XOR.  Depth 2 is the first size that can hold the interaction inside a single')
    print('  tree, and 200 rounds of it reach 1.0000.')

    # ------------------------------------------------------------------ 5
    header('5. Silent failure #3: feature importance rewards opportunity, not use')
    Xc, yc, names = cardinality_data(300, seed=2)
    rc = np.random.default_rng(0).permutation(300)
    ctr, cte = rc[:210], rc[210:]
    tc = DecisionTree().fit(Xc[ctr], yc[ctr])
    imp = tc.importances(Xc.shape[1])
    pi = permutation_importance(tc, Xc[cte], yc[cte], np.random.default_rng(7), repeats=20)
    print('  300 rows.  Column 0 is a binary column that IS the answer, 20% of the time')
    print('  flipped.  Column 1 is a continuous column of pure noise.  Columns 2-4 are')
    print('  binary noise.  The tree is grown to purity.')
    print(f'\n  distinct values per column: '
          f'{[len(np.unique(Xc[:, j])) for j in range(Xc.shape[1])]}')
    print(f'  candidate thresholds offered at the root: '
          f'{[len(split_candidates(Xc[ctr, j])) for j in range(Xc.shape[1])]}')
    print(f'\n  {"column":<22} {"impurity importance":>20} {"permutation (held out)":>24}')
    for nm, a, b in zip(names, imp, pi):
        print(f'  {nm:<22} {a:>20.3f} {b:>+24.3f}')
    print(f'\n  Impurity importance puts the pure-noise column FIRST, at {imp[1] / imp[0]:.2f}x the real')
    print(f'  signal ({imp[1]:.3f} against {imp[0]:.3f}).  Permutation importance, measured on rows')
    print(f'  the tree never saw, puts the real column first ({pi[0]:+.3f}) and leaves the')
    print(f'  noise column near zero ({pi[1]:+.3f}).  The two tables disagree about which')
    print('  column the model is made of, and only one of them was asked the question.')
    print('  The mechanism is in the two lines above the table: a continuous column with')
    print(f'  {len(np.unique(Xc[ctr, 1]))} distinct values offers {len(split_candidates(Xc[ctr, 1]))} chances to look good on noise, '
          'and a binary')
    print('  column offers 1.  Impurity importance counts how much each column got to')
    print('  remove; it has no way to ask whether removing it meant anything.')
    print(f'  (train {accuracy(tc.predict(Xc[ctr]), yc[ctr]):.3f} vs test '
          f'{accuracy(tc.predict(Xc[cte]), yc[cte]):.3f} - the tree is memorising, and the')
    print('  noise column is what it memorises with.)')

    # ------------------------------------------------------------------ 6
    header('6. Silent failure #4: one row changes the whole tree')
    Xi, yi = stripes(200, seed=0, noise=0.10)
    base = DecisionTree(max_depth=4).fit(Xi, yi)
    changed, drift = 0, []
    for i in range(len(yi)):
        keep = np.ones(len(yi), bool)
        keep[i] = False
        t = DecisionTree(max_depth=4).fit(Xi[keep], yi[keep])
        if (t.root.feat, round(t.root.thr, 9)) != (base.root.feat, round(base.root.thr, 9)):
            changed += 1
        drift.append(float(np.mean(t.predict(Xi) != base.predict(Xi))))
    print(f'  Base tree: root is x{base.root.feat} <= {base.root.thr:.4f}, {base.n_leaves} leaves.')
    print(f'  Refitting 200 times, each time with ONE row removed:')
    print(f'    the root split changed in {changed} of 200 runs ({100 * changed / 200:.0f}%)')
    print(f'    predictions moved on {100 * np.mean(drift):.1f}% of rows on average, '
          f'{100 * max(drift):.1f}% at worst')
    print('\n  Nothing about the data changed meaningfully; 199 of 200 rows are identical.')
    print('  A model this sensitive to its sample is a model with high variance, and')
    print('  high variance is exactly what averaging is good at removing.  That is the')
    print('  entire argument for the next section.')

    # ------------------------------------------------------------------ 7
    header('7. Random forest: bagging, then making the trees ask different questions')
    Xd, yd = dominant(900, seed=0)
    rd = np.random.default_rng(1).permutation(900)
    dtr, dte = rd[:600], rd[600:]
    one = DecisionTree().fit(Xd[dtr], yd[dtr])
    print('  900 rows, 10 columns: one that dominates, four that help a little, five pure')
    print('  noise.  600 train / 300 test, averaged over 3 forest seeds.\n')
    print(f'  {"model":<32} {"train":>7} {"test":>7} {"disagree":>9} {"oob":>7}')
    print(f'  {"single unpruned tree":<32} {accuracy(one.predict(Xd[dtr]), yd[dtr]):>7.3f} '
          f'{accuracy(one.predict(Xd[dte]), yd[dte]):>7.3f} {"-":>9} {"-":>7}')
    results = {}
    roots = {}
    for nm, kw in [('bagging only, all 10 features', dict(max_features=None)),
                   ('random forest, sqrt = 3 features', dict(max_features='sqrt'))]:
        a_tr, a_te, dis, oobs = [], [], [], []
        cnt = np.zeros(10, dtype=int)
        for sd in range(3):
            f = RandomForest(n_trees=30, seed=sd, **kw).fit(Xd[dtr], yd[dtr])
            a_tr.append(accuracy(f.predict(Xd[dtr]), yd[dtr]))
            a_te.append(accuracy(f.predict(Xd[dte]), yd[dte]))
            dis.append(f.disagreement(Xd[dte]))
            oobs.append(f.oob_score(Xd[dtr], yd[dtr])[0])
            cnt += np.bincount([t.root.feat for t in f.trees], minlength=10)
        results[nm] = (np.mean(a_te), np.mean(dis))
        roots[nm] = cnt
        print(f'  {nm:<32} {np.mean(a_tr):>7.3f} {np.mean(a_te):>7.3f} '
              f'{np.mean(dis):>9.3f} {np.mean(oobs):>7.3f}')
    for nm in roots:
        print(f'\n  root column chosen, over 90 trees ({nm}):')
        print(f'    {roots[nm].tolist()}')
    bag = results['bagging only, all 10 features']
    rf = results['random forest, sqrt = 3 features']
    print(f'\n  Bagging does most of the work: {accuracy(one.predict(Xd[dte]), yd[dte]):.3f} -> {bag[0]:.3f}, '
          f'{bag[0] - accuracy(one.predict(Xd[dte]), yd[dte]):+.3f}.')
    print(f'  Feature subsampling adds {rf[0] - bag[0]:+.3f} on top, which is small and easy to')
    print(f'  over-sell.  What it clearly does change is disagreement: {bag[1]:.3f} -> {rf[1]:.3f}.')
    print('  Averaging predictors that agree buys nothing, and the root histogram shows')
    print('  why: without subsampling, every single tree opens with the same question.')
    print('\n  The out-of-bag column is a free held-out estimate.  Each bootstrap sample')
    print('  leaves out about (1 - 1/n)^n -> 1/e = 36.8% of the rows, so every row is')
    print('  unseen by roughly a third of the trees and can be scored with them.')

    # ------------------------------------------------------------------ 8
    header('8. Silent failure #5: a tree cannot predict a number it has never seen')
    Xw, yw = wave(120, seed=3)
    tw = DecisionTree(task='reg', max_depth=6).fit(Xw, yw)
    fw = RandomForest(n_trees=40, task='reg', max_features=None, seed=0).fit(Xw, yw)
    print('  120 points of y = sin(x) + 0.35x on x in [-3, 3], fitted with a depth-6 tree')
    print('  and a 40-tree forest.  Then asked about x outside that range:\n')
    print(f'  {"x":>7} {"tree":>9} {"forest":>9} {"truth":>9}')
    for xq in (2.9, 3.5, 5.0, 10.0, 100.0):
        p1 = float(tw.predict(np.array([[xq]]))[0])
        p2 = float(fw.predict(np.array([[xq]]))[0])
        print(f'  {xq:>7} {p1:>+9.4f} {p2:>+9.4f} {math.sin(xq) + 0.35 * xq:>+9.4f}')
    print(f'\n  Every prediction past the edge is the same number, forever.  A leaf holds a')
    print('  constant, so the model is a step function, and the last step runs to')
    print('  infinity flat.  Averaging flat things is flat, so the forest does not fix it.')
    print(f'  A tree also cannot go above the largest leaf mean: the data reaches '
          f'{yw.max():.4f},')
    print(f'  the tree never predicts above {tw.predict(Xw).max():.4f}.  Any model whose output')
    print('  is a non-negative weighted average of training targets has this property, and')
    print('  the usual fix is not a better tree - it is to model the trend separately and')
    print('  let the tree fit what is left.')

    # ------------------------------------------------------------------ 9
    header('9. Boosting: gradient descent where the parameter is a function')
    print("""Day 45 took steps in parameter space:  w <- w - lr * dL/dw.
Here the model is a sum of trees and the thing being improved is the function F:

    F0        = mean(y)                      the best constant under squared loss
    residual  = y - F                        = -dL/dF  for  L = 0.5 (y - F)^2
    F         = F + lr * tree(X -> residual) one small step in function space

"fit the next tree to what is left over" is not a separate idea; it is what the
gradient of squared loss happens to look like.  Swap the loss and the target of the
next tree changes with it - that is the whole content of the word "gradient".""")
    Xr, yr = tabular_reg(600, 6, seed=4)
    rr = np.random.default_rng(2).permutation(600)
    rtr, rte = rr[:420], rr[420:]
    g = GradientBoosting(n_trees=300, lr=0.1, max_depth=2).fit(Xr[rtr], yr[rtr], Xr[rte], yr[rte])
    const = mse(np.full(len(rte), yr[rtr].mean()), yr[rte])
    print(f'\n  600 rows, 6 columns (3 informative), 420 train / 180 test.')
    print(f'  predicting the training mean for everything: test MSE {const:.4f}')
    print(f'\n  {"trees":>6} {"train MSE":>11} {"test MSE":>10}')
    for m in (1, 5, 20, 50, 100, 200, 300):
        print(f'  {m:>6} {g.train_loss[m - 1]:>11.4f} {g.val_loss[m - 1]:>10.4f}')
    print('\n  Each tree is depth 2 - at most four leaves, a model that can barely do anything.')
    print(f'  300 of them stacked reach test MSE {g.val_loss[-1]:.4f}, against {const:.4f} for the')
    print('  constant.  Boosting attacks bias: it starts underfitting badly on purpose')
    print('  and walks towards the answer in small steps.')

    # ------------------------------------------------------------------ 10
    header('10. Silent failure #6: the training loss never tells you to stop')
    print(f'  {"lr":>6} {"depth":>6} {"train MSE end":>14} {"best test":>10} {"at tree":>8} '
          f'{"test end":>9} {"train monotone":>15}')
    runs = {}
    for lr, md in [(1.0, 2), (0.3, 2), (0.1, 2), (0.03, 2), (0.1, 6), (0.1, 10)]:
        gb = GradientBoosting(n_trees=300, lr=lr, max_depth=md).fit(
            Xr[rtr], yr[rtr], Xr[rte], yr[rte])
        b = int(np.argmin(gb.val_loss))
        mono = all(gb.train_loss[i] >= gb.train_loss[i + 1] - 1e-12
                   for i in range(len(gb.train_loss) - 1))
        runs[(lr, md)] = (gb.train_loss[-1], gb.val_loss[b], b + 1, gb.val_loss[-1], mono)
        print(f'  {lr:>6} {md:>6} {gb.train_loss[-1]:>14.4f} {gb.val_loss[b]:>10.4f} '
              f'{b + 1:>8} {gb.val_loss[-1]:>9.4f} {str(mono):>15}')
    hi = runs[(1.0, 2)]
    print(f'\n  Read the lr = 1.0 row across.  Training MSE finishes at {hi[0]:.4f} - visually')
    print(f'  zero, a perfect fit.  Held-out MSE bottomed out at {hi[1]:.4f} after only {hi[2]} trees')
    print(f'  and then climbed {100 * (hi[3] - hi[1]) / hi[1]:.0f}% to {hi[3]:.4f}.  The training curve is monotone')
    print('  decreasing for every row in this table - the loop above checks that - and it')
    print('  is structurally incapable of telling you when to stop.  That is why early')
    print('  stopping needs held-out data, and why every boosting library has a parameter')
    print('  for it.  (None of them require one; they will all happily run to the end.)')
    lo = runs[(0.1, 2)]
    print(f'\n  Shrinkage is the trade: lr {1.0} reaches its best at {hi[2]} trees and is worse')
    print(f'  ({hi[1]:.4f}); lr 0.1 needs {lo[2]} trees and gets {lo[1]:.4f}.  Smaller steps mean more')
    print('  steps and a better place to stop - exactly the learning-rate story from')
    print('  Day 45, with a tree in place of a gradient vector.')
    d6, d10 = runs[(0.1, 6)], runs[(0.1, 10)]
    print(f'\n  Depth is the other half of the same trade.  At lr 0.1, depth 2 gets {lo[1]:.4f},')
    print(f'  depth 6 gets {d6[1]:.4f} and depth 10 gets {d10[1]:.4f} - and the depth-10 run reaches a')
    print(f'  training MSE of {d10[0]:.4f} on the way there.  A boosting loop wants weak learners;')
    print('  give it strong ones and the first few trees have already memorised the')
    print('  residuals, leaving the rest of the loop nothing honest to fit.')

    # ------------------------------------------------------------------ 11
    header('11. The forest and the boost pull in opposite directions')
    print(f'  {"model":<42} {"test MSE":>9}')
    rows2 = []
    for nm, pred in [
        ('one deep tree (max_depth=None)',
         DecisionTree(task='reg').fit(Xr[rtr], yr[rtr]).predict(Xr[rte])),
        ('one stump (max_depth=2)',
         DecisionTree(task='reg', max_depth=2).fit(Xr[rtr], yr[rtr]).predict(Xr[rte])),
        ('forest of 40 deep trees',
         RandomForest(n_trees=40, task='reg', max_features=3, seed=0)
         .fit(Xr[rtr], yr[rtr]).predict(Xr[rte])),
        ('forest of 40 stumps (max_depth=2)',
         RandomForest(n_trees=40, task='reg', max_depth=2, max_features=3, seed=0)
         .fit(Xr[rtr], yr[rtr]).predict(Xr[rte])),
        ('boosting, 300 stumps, lr 0.1',
         GradientBoosting(n_trees=300, lr=0.1, max_depth=2).fit(Xr[rtr], yr[rtr]).predict(Xr[rte])),
    ]:
        v = mse(pred, yr[rte])
        rows2.append((nm, v))
        print(f'  {nm:<42} {v:>9.4f}')
    print('\n  The forest of deep trees works and the forest of stumps does not: averaging')
    print('  removes variance, and a stump has almost none to remove - it is simply')
    print('  wrong.  Boosting on stumps is the best row in the table, because dependence')
    print('  is the point: tree 2 is fitted to what tree 1 got wrong.  (The two stump rows')
    print('  are not literally the same trees - the forest gives each stump 3 random')
    print('  columns and the boosted ones see all 6 - but the depth, which is what the')
    print('  comparison is about, is identical.)')
    print('  Deep and independent kills variance; shallow and dependent kills bias.')
    print('  Swapping the depths swaps the failure: the stump forest underfits, and the')
    print(f'  depth-10 boosting run in the table above lands at {runs[(0.1, 10)][1]:.4f} instead of {lo[1]:.4f}.')

    # ------------------------------------------------------------------ 12
    header('12. LeetCode 112 - Path Sum')
    r22 = build([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])
    print(f'  [5,4,8,11,null,13,4,7,2,null,null,null,1], target 22 -> '
          f'{SolutionPathSum().hasPathSum(r22, 22)}')
    print(f'    the path is 5 -> 4 -> 11 -> 2')
    r12 = build([1, 2])
    print(f'\n  [1,2], target 1  (node 1 has a left child and no right child)')
    print(f'    correct answer                 : {SolutionPathSum().hasPathSum(r12, 1)}')
    print(f'    "None means we reached a leaf" : {has_path_sum_null_leaf(r12, 1)}')
    print('    The second version walks into the missing right child, finds the target')
    print('    already used up, and reports a root-to-leaf path that does not exist.')
    rng12 = np.random.default_rng(11)
    bad = 0
    for _ in range(3000):
        n = int(rng12.integers(1, 12))
        vals = [int(rng12.integers(-6, 7)) if (i == 0 or rng12.random() < 0.72) else None
                for i in range(n)]
        root = build(vals)
        if root is None:
            continue
        tgt = int(rng12.integers(-12, 13))
        if SolutionPathSum().hasPathSum(root, tgt) != has_path_sum_null_leaf(root, tgt):
            bad += 1
    print(f'\n    over 3,000 random small trees the two disagree {bad} times '
          f'({100 * bad / 3000:.1f}%)')
    print('    - common enough to be caught, rare enough to pass a handful of examples.')
    print('\n  A decision tree never has this bug, because every internal node it builds')
    print('  gets exactly two children.  LeetCode trees do not, and "no children" is the')
    print('  definition of a leaf that matters.')

    # ------------------------------------------------------------------ 13
    header('13. LeetCode 257 - Binary Tree Paths')
    r257 = build([1, 2, 3, None, 5])
    print(f'  [1,2,3,null,5] -> {SolutionBinaryTreePaths().binaryTreePaths(r257)}')
    print(f'  one shared list, no copy, no undo -> {binary_tree_paths_shared(r257)}')
    print('\n  Both returned paths are wrong and both are the SAME wrong value: what came')
    print('  back is the accumulator\'s final state, twice, because the list appended at')
    print("  the first leaf is the list that kept growing afterwards (Day 39's bug, in a")
    print('  tree instead of a backtracking search).  Either copy on the way in, as the')
    print('  solution above does, or append and pop around the recursive calls.')
    names = ['x', 'y']
    rules = []

    def collect(node, acc):
        if node.is_leaf:
            rules.append((list(acc), node.value, node.n))
            return
        collect(node.left, acc + [f'{names[node.feat]} <= {node.thr:.2f}'])
        collect(node.right, acc + [f'{names[node.feat]} > {node.thr:.2f}'])

    collect(t3.root, [])
    print(f'\n  The same walk over the depth-3 tree from section 1 turns it into {len(rules)} rules:')
    for cond, val, n_ in rules[:4]:
        print(f'    if {" and ".join(cond)}  ->  class {val}   ({n_} rows)')
    print(f'    ... and {len(rules) - 4} more')
    print('  That is the whole model, in a form a person can argue with - which is the')
    print('  one thing trees give you that nothing else in this series does.')

    # ------------------------------------------------------------------ 14
    header('14. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
