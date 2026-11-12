"""Day 50 - evaluation metrics and cross-validation, in pure NumPy.

A classifier hands you a score, not a decision.  Every number you then report about
it - accuracy, precision, recall, F1, AUC - is a different ratio taken from the same
four counts, and picking one of them is picking which mistakes you are willing to
make.  Run `python evaluation.py` to print the whole story; every number below is
produced by this file and checked by an assert at the bottom.
"""

import math

import numpy as np


# ====================================================== 1. the four counts
def confusion(y_true, y_pred):
    """Return (tn, fp, fn, tp) for 0/1 labels.

    Everything else on this page is one ratio of these four numbers.  They are
    counts, so they never go undefined; the ratios built from them can.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)


def precision(y_true, y_pred):
    """Of the rows we flagged, how many were right?  Undefined when we flag nothing."""
    tn, fp, fn, tp = confusion(y_true, y_pred)
    return float('nan') if tp + fp == 0 else tp / (tp + fp)


def recall(y_true, y_pred):
    """Of the rows that really are positive, how many did we catch?"""
    tn, fp, fn, tp = confusion(y_true, y_pred)
    return float('nan') if tp + fn == 0 else tp / (tp + fn)


def f1(y_true, y_pred):
    """Harmonic mean of precision and recall - 0 when either one is 0."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    if math.isnan(p) or math.isnan(r) or p + r == 0:
        return 0.0 if not (math.isnan(p) or math.isnan(r)) else float('nan')
    return 2 * p * r / (p + r)


def specificity(y_true, y_pred):
    """Of the rows that really are negative, how many did we leave alone?"""
    tn, fp, fn, tp = confusion(y_true, y_pred)
    return float('nan') if tn + fp == 0 else tn / (tn + fp)


def balanced_accuracy(y_true, y_pred):
    """Mean of recall and specificity: each class gets one vote regardless of size."""
    return (recall(y_true, y_pred) + specificity(y_true, y_pred)) / 2


def mcc(y_true, y_pred):
    """Matthews correlation: uses all four counts, so it cannot be fooled by
    predicting one class everywhere (that makes the denominator 0 -> nan here)."""
    tn, fp, fn, tp = confusion(y_true, y_pred)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float('nan') if den == 0 else (tp * tn - fp * fn) / den


def false_positive_rate(y_true, y_pred):
    """fp / (fp + tn): of the rows that really are negative, how many did we flag?

    This is the x-axis of the ROC curve.  Like recall it is computed entirely inside
    one class, which is why both ROC axes barely notice the class balance.
    """
    tn, fp, fn, tp = confusion(y_true, y_pred)
    return float('nan') if tn + fp == 0 else fp / (tn + fp)


def apples_and_oranges(n=100, n_apples=10, caught=6, wrong=2):
    """A basket of fruit and an apple detector, for the four denominators.

    100 pieces of fruit, 10 of them apples.  The detector puts `caught` apples and
    `wrong` oranges on the apple side.  Returns (truth, prediction).
    """
    y = np.zeros(n, dtype=int)
    y[:n_apples] = 1
    p = np.zeros(n, dtype=int)
    p[:caught] = 1                              # apples it got right
    p[n_apples:n_apples + wrong] = 1            # oranges it called apples
    return y, p


def rebalance(y_true, scores, keep_negatives, seed=0):
    """Throw away negatives at random, keep every positive.

    Nothing about the model or its ranking changes; only how rare the positives are.
    Section 8 averages this over many draws to show which metrics notice and which
    do not.
    """
    y = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = rng.permutation(np.where(y == 0)[0])[:keep_negatives]
    sel = np.r_[pos, neg]
    return y[sel], np.asarray(scores, dtype=float)[sel]


def predict_at(scores, threshold):
    """A score becomes a decision only when you pick a threshold.  0.5 is a pick."""
    return (np.asarray(scores) >= threshold).astype(int)


# ====================================================== 2. the two curves
def roc_curve(y_true, scores):
    """False-positive rate against true-positive rate, over every threshold.

    Both axes are divided by a class total, so the curve does not know that one
    class is a hundred times bigger than the other.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind='mergesort')
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # keep only the last index of each run of equal scores: one point per threshold
    keep = np.r_[np.diff(s) != 0, True]
    P, N = int(y.sum()), int((1 - y).sum())
    tpr = np.r_[0.0, tp[keep] / P]
    fpr = np.r_[0.0, fp[keep] / N]
    return fpr, tpr


def pr_curve(y_true, scores):
    """Recall against precision.  Precision has the number of false positives in
    its denominator, so it feels every false positive, however many negatives exist.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind='mergesort')
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    keep = np.r_[np.diff(s) != 0, True]
    P = int(y.sum())
    rec = tp[keep] / P
    prec = tp[keep] / (tp[keep] + fp[keep])
    return rec, prec


def auc_trapezoid(x, y):
    """Area under a curve given as points, by the trapezoid rule."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.trapezoid(y, x)) if hasattr(np, 'trapezoid') else float(np.trapz(y, x))


def average_precision(y_true, scores):
    """The step-wise area under the precision-recall curve: sum over the positives
    of (how much recall this one added) x (precision once it is included).

    The trapezoid rule is wrong here - it interpolates between two operating points
    that no threshold produces - so this is the number everyone actually reports.
    """
    rec, prec = pr_curve(y_true, scores)
    rec = np.r_[0.0, rec]
    return float(np.sum(np.diff(rec) * prec))


def roc_auc_pairs(y_true, scores):
    """The definition, written out: over every (positive, negative) pair, how often
    does the positive score higher?  A tie counts half.  This is O(P x N) pairs."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    diff = pos[:, None] - neg[None, :]
    wins = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return float(wins / (len(pos) * len(neg)))


def roc_auc_rank(y_true, scores):
    """The same number from the ranks (Mann-Whitney U), in O(n log n): sort once,
    average the ranks inside each group of equal scores, and add up the positives."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s, kind='mergesort')
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0        # 1-based, ties share the mean rank
        i = j + 1
    r = np.empty(len(s), dtype=float)
    r[order] = ranks
    P, N = int(y.sum()), int((1 - y).sum())
    if P == 0 or N == 0:
        return float('nan')
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


# ====================================================== 3. splitting the data
def kfold_indices(n, k, seed=0, shuffle=True):
    """Plain k-fold: shuffle the row numbers, cut into k near-equal blocks.

    Nothing here looks at the labels, which is the whole problem when one class is
    rare - see section 10.
    """
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    return [np.sort(part) for part in np.array_split(idx, k)]


def stratified_kfold_indices(y, k, seed=0):
    """Same cut, done once per class, so every fold keeps the overall positive rate
    to within one row."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        members = np.where(y == cls)[0]
        rng.shuffle(members)
        for f, part in enumerate(np.array_split(members, k)):
            folds[f].extend(part.tolist())
    return [np.sort(np.array(f, dtype=int)) for f in folds]


def group_kfold_indices(groups, k):
    """Keep every row of a group together.  Rows from one customer, one patient or
    one document split across train and test are the same row twice."""
    uniq = np.unique(groups)
    buckets = [[] for _ in range(k)]
    sizes = [0] * k
    for g in uniq:                                   # largest group to the smallest bucket
        rows = np.where(groups == g)[0]
        t = int(np.argmin(sizes))
        buckets[t].extend(rows.tolist())
        sizes[t] += len(rows)
    return [np.sort(np.array(b, dtype=int)) for b in buckets]


def time_series_splits(n, k):
    """Train on the past, test on the next block.  Never shuffle a time series:
    a shuffled split lets the model read rows that happen after the ones it predicts.
    """
    bounds = np.linspace(0, n, k + 2).astype(int)
    out = []
    for i in range(1, k + 1):
        out.append((np.arange(0, bounds[i]), np.arange(bounds[i], bounds[i + 1])))
    return out


# ====================================================== 4. a model to evaluate
class LogisticRegression:
    """Logistic regression by gradient descent, with an L2 penalty.

    Same three lines as day 46's output layer: p = sigmoid(Xw + b), the gradient of
    the cross-entropy is X.T @ (p - y) / n, and the L2 term adds 2 * lam * w.
    """

    def __init__(self, l2=0.0, lr=0.5, n_iter=400):
        self.l2, self.lr, self.n_iter = l2, lr, n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self.predict_proba(X)
            g = X.T @ (p - y) / n + 2.0 * self.l2 * self.w
            gb = float(np.mean(p - y))
            self.w -= self.lr * g
            self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        z = np.asarray(X, dtype=float) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def standardise(X, mean=None, std=None):
    """Centre and scale.  Which rows the mean and the std come from is the point of
    section 12."""
    if mean is None:
        mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


# ====================================================== 5. datasets
def imbalanced(n=2000, pos_rate=0.01, sep=1.6, d=6, seed=0):
    """n rows, a given fraction of them positive, the two classes offset by `sep`
    standard deviations along every column."""
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * pos_rate))
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    X = rng.standard_normal((n, d))
    X[y == 1] += sep / math.sqrt(d)
    order = rng.permutation(n)
    return X[order], y[order]


def broad_and_sharp(n=2000, n_pos=20, seed=1):
    """Two score columns over the same labels, built to disagree.

    `broad` separates the whole population well but puts 135 negatives above its
    weakest positive - section 7 measures that.  `sharp` ranks half the positives
    above every negative and buries the other half low: those ten outrank only 22% of
    the negatives, which section 7 measures.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    broad = np.where(y == 1, rng.normal(0.72, 0.10, n), rng.normal(0.30, 0.20, n))
    sharp = np.where(y == 0, rng.uniform(0.00, 0.90, n), 0.0)
    half = n_pos // 2
    sharp[:half] = rng.uniform(0.95, 1.00, half)             # certain, and correct
    sharp[half:n_pos] = rng.uniform(0.15, 0.25, n_pos - half)  # buried near the bottom
    perm = rng.permutation(n)
    return y[perm], np.clip(broad[perm], 0, 1), sharp[perm]


def sparse_signal(n, d=40, n_signal=4, seed=7):
    """n rows, d columns, only the first few of which affect the label.  With d close
    to n a linear model can fit the training rows exactly using the columns that mean
    nothing, which is what the L2 penalty in section 13 is there to stop."""
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    w[:n_signal] = [2.0, -2.0, 1.5, -1.5][:n_signal]
    X = rng.standard_normal((n, d))
    p = 1.0 / (1.0 + np.exp(-(X @ w)))
    return X, (rng.random(n) < p).astype(int)


def pure_noise_table(n=60, d=400, seed=3):
    """n rows, d columns of noise, a coin-flip label.  There is nothing to learn, so
    any score above chance in section 12 was manufactured by the procedure."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)), rng.integers(0, 2, n)


# ====================================================== 6. cross-validation
def cv_predictions(X, y, folds, make_model, scale_inside=True):
    """Train on every fold but one, predict the held-out fold, keep the scores.

    `scale_inside=False` is the mistake in section 12: the column means and standard
    deviations are then computed once over all the rows, including the ones the model
    is about to be tested on.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    oof = np.zeros(len(y))                            # out-of-fold scores
    if not scale_inside:
        Xs, _, _ = standardise(X)
    for f in range(len(folds)):
        test = folds[f]
        train = np.setdiff1d(np.arange(len(y)), test)
        if scale_inside:
            Xtr, mu, sd = standardise(X[train])
            Xte, _, _ = standardise(X[test], mu, sd)
        else:
            Xtr, Xte = Xs[train], Xs[test]
        model = make_model().fit(Xtr, y[train])
        oof[test] = model.predict_proba(Xte)
    return oof


def per_fold_metric(y, oof, folds, metric_fn, threshold=0.5):
    """One number per fold.  Folds with no positives make recall 0/0, which is where
    the nans in section 10 come from."""
    out = []
    for f in folds:
        out.append(metric_fn(y[f], predict_at(oof[f], threshold)))
    return np.array(out, dtype=float)


def select_by_correlation(X, y, k):
    """Keep the k columns most correlated with the label.  Run on all the rows this
    is the fastest way to invent a signal that is not there (section 12)."""
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum()) + 1e-12
    corr = np.abs(Xc.T @ yc / denom)
    return np.argsort(-corr)[:k]


def nearest_centroid_accuracy(Xtr, ytr, Xte, yte):
    """A classifier with no knobs at all, so that section 12's numbers cannot be
    blamed on the model: put each row with the class centre it sits closest to."""
    c0 = Xtr[ytr == 0].mean(axis=0)
    c1 = Xtr[ytr == 1].mean(axis=0)
    d0 = ((Xte - c0) ** 2).sum(axis=1)
    d1 = ((Xte - c1) ** 2).sum(axis=1)
    return float(np.mean((d1 < d0).astype(int) == yte))


# ====================================================== 7. LeetCode 493
class SolutionReversePairs:
    """LeetCode 493 - Reverse Pairs.

    Count the pairs i < j with nums[i] > 2 * nums[j].  Counting the pairs where a
    positive outranks a negative - which is what AUC is - has the same shape, and
    the same O(n^2) trap: the answer is a count over pairs, but you must never build
    the pairs.  Merge sort counts a whole block at a time while it sorts.
    """

    def reversePairs(self, nums: list[int]) -> int:
        def sort_count(a):
            if len(a) < 2:
                return a, 0
            mid = len(a) // 2
            left, cl = sort_count(a[:mid])
            right, cr = sort_count(a[mid:])
            count = cl + cr
            j = 0
            for v in left:                       # both halves are sorted, so j only moves forward
                while j < len(right) and v > 2 * right[j]:
                    j += 1
                count += j
            merged, i, j2 = [], 0, 0
            while i < len(left) and j2 < len(right):
                if left[i] <= right[j2]:
                    merged.append(left[i]); i += 1
                else:
                    merged.append(right[j2]); j2 += 1
            merged.extend(left[i:]); merged.extend(right[j2:])
            return merged, count
        return sort_count(list(nums))[1]


def reverse_pairs_during_merge(nums):
    """The tempting shortcut: count while merging, instead of in a pass of its own.

    The merge advances on `left[i] <= right[j]`, but the pairs we want are
    `left[i] > 2 * right[j]`.  Those are two different orders, so the pointer stops
    in the wrong place and the count comes out low.  It still returns a number.
    """
    def sort_count(a):
        if len(a) < 2:
            return a, 0
        mid = len(a) // 2
        left, cl = sort_count(a[:mid])
        right, cr = sort_count(a[mid:])
        count, merged, i, j = cl + cr, [], 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                if left[i] > 2 * right[j]:
                    count += len(left) - i
                merged.append(right[j]); j += 1
        merged.extend(left[i:]); merged.extend(right[j:])
        return merged, count
    return sort_count(list(nums))[1]


def reverse_pairs_brute(nums):
    n = len(nums)
    return sum(1 for i in range(n) for j in range(i + 1, n) if nums[i] > 2 * nums[j])


# ====================================================== 8. LeetCode 169
class SolutionMajorityElement:
    """LeetCode 169 - Majority Element.

    Boyer-Moore: keep one candidate and a count; a value that occupies more than half
    the array survives every cancellation.  This is the baseline an imbalanced-data
    accuracy score is really measuring - always answer the majority class.
    """

    def majorityElement(self, nums: list[int]) -> int:
        count, candidate = 0, None
        for v in nums:
            if count == 0:
                candidate = v
            count += 1 if v == candidate else -1
        return candidate


def majority_baseline(y):
    """Accuracy of always answering the majority class - the number every other
    accuracy on this page has to beat before it means anything."""
    y = np.asarray(y).astype(int)
    maj = int(SolutionMajorityElement().majorityElement(y.tolist()))
    return maj, float(np.mean(y == maj))


def majority_if_any(nums):
    """LeetCode 169 promises a majority exists.  Without that promise Boyer-Moore
    still returns something, so the follow-up is a second pass that checks it."""
    c = SolutionMajorityElement().majorityElement(nums)
    return c if sum(1 for v in nums if v == c) * 2 > len(nums) else None


def one_nearest_neighbour(Xtr, ytr, Xte):
    """Copy the label of the closest training row.  It has no parameters, so when it
    scores well the only thing it can have found is a row that looks like one it
    already saw - which is exactly what group leakage hands it."""
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    return ytr[np.argmin(d2, axis=1)]


def grouped_rows(n_groups=150, per=4, d=6, within=0.25, seed=11):
    """Four rows per customer.  The label belongs to the customer, and the four rows
    sit close together, so a random split puts near-copies on both sides of it."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    glabel = rng.integers(0, 2, n_groups)
    gshift = rng.standard_normal((n_groups, d)) * 2.0
    X = gshift[g] + rng.standard_normal((n_groups * per, d)) * within
    return X, glabel[g], g


def equal_f1_pair(n=2000, n_pos=100):
    """Two sets of predictions with the same F1 and opposite behaviour.

    A flags 300 rows and catches 90 of the 100 positives; B flags 100 and catches 45.
    Both score F1 0.45, and B - the one that misses more than half the positives -
    has the higher accuracy.
    """
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    a = np.zeros(n, dtype=int)
    a[:90] = 1                       # 90 true positives
    a[n_pos:n_pos + 210] = 1         # 210 false positives -> 300 flagged
    b = np.zeros(n, dtype=int)
    b[:45] = 1                       # 45 true positives
    b[n_pos:n_pos + 55] = 1          # 55 false positives -> 100 flagged
    return y, a, b


def best_of_m_on_one_set(m, n=200, reps=40, seed=0):
    """Score m coin-flip models on one labelled set, keep the best, then score that
    same model on a second set drawn the same way.

    Nothing here has any signal: both label sets are coin flips and so is every
    model.  Whatever the first number rises to is the amount of optimism you buy by
    choosing on a set and then reporting the score from that same set.
    """
    picked, fresh = [], []
    for rep in range(reps):
        r = np.random.default_rng(seed + rep)
        y_pick, y_fresh = r.integers(0, 2, n), r.integers(0, 2, n)
        best = (-1.0, 0.0)
        for _ in range(m):
            pred = r.integers(0, 2, n)
            cand = (float(np.mean(pred == y_pick)), float(np.mean(pred == y_fresh)))
            if cand > best:
                best = cand
        picked.append(best[0])
        fresh.append(best[1])
    return float(np.mean(picked)), float(np.mean(fresh))


def precision_at_k(y_true, scores, k):
    """Of the k rows with the highest scores, how many are positive?  This is the
    number a team with a fixed review budget actually lives on."""
    y = np.asarray(y_true).astype(int)
    idx = np.argsort(-np.asarray(scores, dtype=float), kind='mergesort')[:k]
    return int(y[idx].sum()), float(y[idx].mean())

# ====================================================== 9. assertions
def check():
    rng = np.random.default_rng(0)

    # the four counts, by hand
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    assert confusion(y, p) == (4, 1, 1, 2)
    assert abs(accuracy(y, p) - 6 / 8) < 1e-12
    assert abs(precision(y, p) - 2 / 3) < 1e-12
    assert abs(recall(y, p) - 2 / 3) < 1e-12
    assert abs(f1(y, p) - 2 / 3) < 1e-12
    assert abs(specificity(y, p) - 4 / 5) < 1e-12
    assert abs(balanced_accuracy(y, p) - (2 / 3 + 4 / 5) / 2) < 1e-12

    # the apple basket: each ratio divides by a different thing
    ya, pa_ = apples_and_oranges()
    assert confusion(ya, pa_) == (88, 2, 4, 6)
    assert abs(accuracy(ya, pa_) - 0.94) < 1e-12
    assert abs(precision(ya, pa_) - 6 / 8) < 1e-12        # of the apple SIDE
    assert abs(recall(ya, pa_) - 6 / 10) < 1e-12          # of ALL the apples
    assert abs(false_positive_rate(ya, pa_) - 2 / 90) < 1e-12
    # FPR and specificity are the same number seen from the two ends
    assert abs(false_positive_rate(ya, pa_) + specificity(ya, pa_) - 1.0) < 1e-12
    # the ROC curve is exactly (FPR, recall) at every threshold - check it point by point
    yy = np.array([1, 1, 0, 0, 1, 0])
    ss = np.array([.95, .40, .80, .05, .60, .30])
    fpr_c, tpr_c = roc_curve(yy, ss)
    for t in sorted(set(ss.tolist())):
        pt = predict_at(ss, t)
        pair = (false_positive_rate(yy, pt), recall(yy, pt))
        assert any(abs(pair[0] - a_) < 1e-12 and abs(pair[1] - b_) < 1e-12
                   for a_, b_ in zip(fpr_c, tpr_c)), (t, pair)

    # dropping negatives at random moves average precision and leaves ROC-AUC alone
    yb_, broad_, _ = broad_and_sharp()
    base_auc, base_ap = roc_auc_rank(yb_, broad_), average_precision(yb_, broad_)
    aucs, aps = [], []
    for r in range(50):
        yr, sr = rebalance(yb_, broad_, 124, seed=r)
        assert int(yr.sum()) == int(yb_.sum()) and len(yr) == int(yb_.sum()) + 124
        aucs.append(roc_auc_rank(yr, sr))
        aps.append(average_precision(yr, sr))
    assert abs(np.mean(aucs) - base_auc) < 0.01      # the ROC axes do not notice
    assert np.mean(aps) - base_ap > 0.5              # precision very much does

    # predicting one class everywhere: precision and MCC go undefined, accuracy does not
    allneg = np.zeros(8, dtype=int)
    assert math.isnan(precision(y, allneg)) and math.isnan(mcc(y, allneg))
    assert abs(accuracy(y, allneg) - 5 / 8) < 1e-12

    # the pair definition of AUC and the rank formula agree, ties included
    for seed in range(20):
        r = np.random.default_rng(seed)
        n = int(r.integers(20, 120))
        yy = r.integers(0, 2, n)
        if yy.sum() == 0 or yy.sum() == n:
            continue
        ss = np.round(r.normal(size=n), 1)            # rounding manufactures ties
        assert abs(roc_auc_pairs(yy, ss) - roc_auc_rank(yy, ss)) < 1e-12
    # a constant score is a coin flip by definition: every pair is a tie
    assert abs(roc_auc_pairs(np.r_[np.ones(5), np.zeros(5)].astype(int),
                             np.full(10, 0.3)) - 0.5) < 1e-12

    # average precision is bounded, and a perfect ranking scores exactly 1
    yp = np.array([1, 1, 1, 0, 0, 0, 0])
    assert abs(average_precision(yp, np.array([.9, .8, .7, .6, .5, .4, .3])) - 1.0) < 1e-12
    yr, sr = rng.integers(0, 2, 50), rng.random(50)
    assert 0.0 <= average_precision(yr, sr) <= 1.0

    # folds really are a partition
    n, k = 103, 5
    ys = np.r_[np.ones(9), np.zeros(94)].astype(int)
    plain = kfold_indices(n, k, seed=1)
    strat = stratified_kfold_indices(ys, k)
    for folds in (plain, strat):
        assert sorted(np.concatenate(folds).tolist()) == list(range(n))
    # plain k-fold makes the fold sizes as equal as they can be
    assert max(len(f) for f in plain) - min(len(f) for f in plain) <= 1
    # stratified k-fold splits each class separately, so the sizes can differ by one
    # per class rather than one overall - here 21/21/21/21/19 for two classes
    assert max(len(f) for f in strat) - min(len(f) for f in strat) <= 2
    counts = [int(ys[f].sum()) for f in strat]
    assert max(counts) - min(counts) <= 1 and sum(counts) == 9

    # groups never straddle a fold boundary
    groups = np.repeat(np.arange(12), 4)
    for f in group_kfold_indices(groups, 4):
        others = np.setdiff1d(np.arange(len(groups)), f)
        assert not set(groups[f].tolist()) & set(groups[others].tolist())

    # time-series splits only ever test on rows after the ones they trained on
    for tr, te in time_series_splits(100, 4):
        assert tr.max() < te.min()

    # LeetCode 493 against brute force
    for seed in range(300):
        r = np.random.default_rng(seed)
        a = r.integers(-20, 20, int(r.integers(0, 12))).tolist()
        assert SolutionReversePairs().reversePairs(a) == reverse_pairs_brute(a)
    assert SolutionReversePairs().reversePairs([1, 3, 2, 3, 1]) == 2
    assert SolutionReversePairs().reversePairs([2, 4, 3, 5, 1]) == 3
    # counting inside the merge never overcounts - it stops early, it does not invent pairs
    for seed in range(400):
        r = np.random.default_rng(seed)
        a = r.integers(-20, 20, int(r.integers(2, 10))).tolist()
        assert reverse_pairs_during_merge(a) <= reverse_pairs_brute(a)

    # LeetCode 169 against a count, and the follow-up that 169 does not ask for
    for seed in range(300):
        r = np.random.default_rng(seed + 1000)
        n = int(r.integers(1, 15))
        maj = int(r.integers(0, 5))
        a = [maj] * (n // 2 + 1) + r.integers(0, 5, n - n // 2 - 1).tolist()
        r.shuffle(a)
        assert SolutionMajorityElement().majorityElement(a) == maj
    assert majority_if_any([1, 2, 3]) is None
    assert majority_if_any([2, 2, 1, 1, 1, 2, 2]) == 2
    return True


# ====================================================== 10. the whole story
def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    np.set_printoptions(suppress=True, legacy='1.25')

    # ------------------------------------------------------------------ 1
    header('1. A model that learned something, scoring worse than doing nothing')
    X, y = imbalanced(2000, 0.01, sep=1.6, d=6, seed=0)
    folds = stratified_kfold_indices(y, 5)
    oof = cv_predictions(X, y, folds, lambda: LogisticRegression())
    pred = predict_at(oof, 0.5)
    maj, base = majority_baseline(y)
    print(f'  2,000 rows, {int(y.sum())} of them positive.  Logistic regression, scored on the')
    print('  rows it was not trained on (5-fold cross-validation).\n')
    print(f'  {("always answer " + str(maj)):<25} accuracy {base:.4f}')
    print(f'  {"the trained model at 0.5":<25} accuracy {accuracy(y, pred):.4f}   '
          f'recall {recall(y, pred):.4f}   precision {precision(y, pred):.4f}')
    print(f'  it flagged {int(pred.sum())} row(s) out of 2,000 and got {confusion(y, pred)[3]} of them right.')
    print(f'\n  So the trained model is worse than a constant, and its F1 is {f1(y, pred):.4f}.')
    print(f'  Except that the same scores give ROC-AUC {roc_auc_rank(y, oof):.4f} and average')
    print(f'  precision {average_precision(y, oof):.4f}, which say it ranks the 20 positives well.')
    rank0 = np.empty(len(oof), dtype=int)
    rank0[np.argsort(-oof, kind='mergesort')] = np.arange(1, len(oof) + 1)
    pr = np.sort(rank0[y == 1])
    print(f'  Ranked by score, the 20 positives sit at places {pr[0]}, {pr[1]}, ... {pr[-2]}, {pr[-1]}')
    print(f'  out of 2,000; {int((pr <= 20).sum())} of them are in the top 20, and the median rank is '
          f'{int(np.median(pr))} - the two')
    print(f'  middle positives sit at {pr[9]} and {pr[10]}.')
    print('  That is what a ROC-AUC of 0.90 looks like from close up: mostly high, with a')
    print('  long tail - which is the first hint that one summary number is not a picture.')
    print(f'\n  Both readings are arithmetically correct.  The highest score it ever gave any')
    print(f'  row is {oof.max():.4f}, so a cut at 0.5 was always going to flag almost nothing -')
    print('  and 0.5 is a number nobody in this story chose on purpose.')

    # ------------------------------------------------------------------ 2
    header('2. Four counts, and every metric is one ratio of them')
    y8 = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p8 = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    tn, fp, fn, tp = confusion(y8, p8)
    print(f'  Eight rows.  truth {y8.tolist()}')
    print(f'                guess {p8.tolist()}\n')
    print(f'    {"":<10}{"guessed 0":>11}{"guessed 1":>12}')
    print(f'    {"really 0":<10}{tn:>11}{fp:>12}')
    print(f'    {"really 1":<10}{fn:>11}{tp:>12}\n')
    rows = [('accuracy', accuracy(y8, p8), '(tp + tn) / everything'),
            ('precision', precision(y8, p8), 'tp / (tp + fp)   of what we flagged'),
            ('recall', recall(y8, p8), 'tp / (tp + fn)   of what was there'),
            ('specificity', specificity(y8, p8), 'tn / (tn + fp)   of what we left alone'),
            ('F1', f1(y8, p8), 'harmonic mean of precision and recall'),
            ('balanced accuracy', balanced_accuracy(y8, p8), '(recall + specificity) / 2'),
            ('MCC', mcc(y8, p8), 'uses all four counts')]
    for nm, v, how in rows:
        print(f'    {nm:<18} {v:>7.4f}   {how}')
    print('\n  Picking a metric is picking which of the four counts you are willing to let')
    print('  grow.  Accuracy adds tp and tn together, so when tn is 99% of the table it is')
    print('  mostly a report about tn.')

    # ------------------------------------------------------------------ 3
    header('3. The four ratios, and which denominator each one divides by')
    ya, pa_ = apples_and_oranges()
    tn, fp, fn, tp = confusion(ya, pa_)
    print('  A basket of 100 pieces of fruit: 10 apples, 90 oranges.  An apple detector')
    print(f'  puts {tp + fp} pieces on the apple side - {tp} apples and {fp} oranges - and leaves')
    print(f'  the other {tn + fn} alone, including {fn} apples it missed.\n')
    print(f'    tp {tp}    fp {fp}    fn {fn}    tn {tn}\n')
    print(f'    accuracy  = (tp + tn) / 100      = {accuracy(ya, pa_):.4f}'
          f'   of EVERYTHING, how much did we get right')
    print(f'    precision = tp / (tp + fp) = {tp}/{tp + fp} = {precision(ya, pa_):.4f}'
          f'   of the APPLE SIDE, how much is apple')
    print(f'    recall    = tp / (tp + fn) = {tp}/{tp + fn} = {recall(ya, pa_):.4f}'
          f'   of ALL THE APPLES, how many we got')
    print(f'    FPR       = fp / (fp + tn) = {fp}/{fp + tn} = {false_positive_rate(ya, pa_):.4f}'
          f'   of ALL THE ORANGES, how many we grabbed')
    print(f'    F1        = harmonic mean         = {f1(ya, pa_):.4f}')
    print('\n  Two spellings worth keeping: preci-SI-on is the SIDE you picked, and')
    print('  rec-ALL is ALL the apples.  Precision divides by what the MODEL said;')
    print('  recall divides by what the TRUTH said; accuracy divides by everything.')
    print('\n  Why accuracy is not enough: look at what each denominator is made of.')
    print('  recall counts apples only.  FPR counts oranges only.  precision counts the')
    print('  pieces YOU put on the apple side, so the model decides how big it is.  And')
    print('  accuracy counts the whole basket, which the dataset decides and the model')
    print('  cannot change: with 90 oranges in 100 pieces of fruit, most of accuracy is')
    print('  settled before the detector is switched on.  One that gives up and calls')
    print('  every piece of fruit an orange scores')
    none = np.zeros(len(ya), dtype=int)
    print(f'    accuracy {accuracy(ya, none):.4f}   recall {recall(ya, none):.4f}   '
          f'precision {precision(ya, none)}   (0 / 0 - it flagged nothing)')
    print('  which is not far off the real detector, while finding no apples at all.')
    print('  Precision cannot be inflated the same way.  Oranges reach its denominator')
    print('  only when the detector actually reaches for them, which is the mistake it is')
    print('  being charged for - and when it reaches for nothing, precision answers 0 / 0.')
    print('\n  Which one to optimise is a question about costs, not about statistics:')
    print('    a missed apple costs more than a wasted check   -> recall')
    print('    a wasted check costs more than a missed apple   -> precision')
    print('    both cost about the same, and the classes are balanced -> accuracy')
    print('    you need one number and the classes are not balanced   -> F1')

    # ------------------------------------------------------------------ 4
    header('4. Silent failure #1: accuracy measures the size of the majority class')
    allneg = np.zeros(len(y), dtype=int)
    print(f'  Predict "{maj}" for all 2,000 rows: accuracy {accuracy(y, allneg):.4f}, '
          f'recall {recall(y, allneg):.4f},')
    print(f'  precision {precision(y, allneg)} (0 / 0), MCC {mcc(y, allneg)} (the denominator is 0).')
    print('\n  Precision and MCC refuse to answer, and that refusal is the useful part -')
    print('  they have noticed there is nothing to score.  Accuracy answers cheerfully.')
    print(f'  Finding that constant is LeetCode 169, below: the majority class here is {maj},')
    print(f'  it covers {base:.4f} of the rows, and that is the number any accuracy you report')
    print('  has to beat before it has said anything at all.')

    # ------------------------------------------------------------------ 5
    header('5. The threshold is a number you chose, even if you never typed it')
    print('  Same model, same scores, nine cuts.  Nothing is retrained between rows.\n')
    print(f'  {"cut":>6} {"flagged":>9} {"accuracy":>10} {"precision":>11} {"recall":>9} {"F1":>8}')
    sweep = []
    for t in (0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01):
        pt = predict_at(oof, t)
        sweep.append((t, int(pt.sum()), accuracy(y, pt), precision(y, pt), recall(y, pt), f1(y, pt)))
        print(f'  {t:>6.2f} {int(pt.sum()):>9} {accuracy(y, pt):>10.4f} '
              f'{precision(y, pt):>11.4f} {recall(y, pt):>9.4f} {f1(y, pt):>8.4f}')
    best = max(sweep, key=lambda r: r[5])
    print(f'\n  The best F1 is {best[5]:.4f} at a cut of {best[0]:.2f}, where the model flags '
          f'{best[1]} rows')
    print(f'  and catches {best[4] * 20:.0f} of the 20 positives.  Accuracy at that cut is '
          f'{best[2]:.4f} -')
    print(f'  LOWER than the {accuracy(y, predict_at(oof, 0.5)):.4f} it scored while catching nothing.')
    print('  Accuracy goes down as the model starts being useful, because being useful')
    print('  means spending true negatives to buy true positives.')
    precs = [r[3] for r in sweep if not math.isnan(r[3])]
    recs = [r[4] for r in sweep]
    print(f'\n  Recall is monotone - it can only rise as the cut comes down: '
          f'{all(a <= b for a, b in zip(recs, recs[1:]))}.')
    print(f'  Precision is not, and the table shows it: {precs[2]:.4f} at a cut of 0.20,')
    print(f'  {precs[3]:.4f} at 0.15, back up to {precs[4]:.4f} at 0.10.')
    wobble = 0
    for seed in range(300):
        r = np.random.default_rng(seed + 55)
        yy, ss = r.integers(0, 2, 40), r.random(40)
        pr_ = [precision(yy, predict_at(ss, t)) for t in sorted(ss, reverse=True)]
        pr_ = [v for v in pr_ if not math.isnan(v)]
        wobble += 1 if any(a > b for a, b in zip(pr_, pr_[1:])) else 0
    print(f'  Over 300 random label-and-score sets, precision falls somewhere along the')
    print(f'  sweep in {wobble} of them.  "Raise the threshold for better precision" is a')
    print('  tendency, not a guarantee; only recall comes with one.')

    # ------------------------------------------------------------------ 6
    header('6. Silent failure #2: the same F1, half the cases caught')
    ye, pa, pb = equal_f1_pair()
    print('  2,000 rows, 100 of them positive.  Two ways of flagging rows:\n')
    print(f'  {"":<10} {"flagged":>8} {"caught":>7} {"precision":>10} {"recall":>8} {"F1":>7} {"accuracy":>10}')
    for nm, pp in (('wide net', pa), ('narrow net', pb)):
        print(f'  {nm:<10} {int(pp.sum()):>8} {confusion(ye, pp)[3]:>7} '
              f'{precision(ye, pp):>10.4f} {recall(ye, pp):>8.4f} {f1(ye, pp):>7.4f} '
              f'{accuracy(ye, pp):>10.4f}')
    print(f'\n  Identical F1 of {f1(ye, pa):.4f}.  The wide net catches 90 of the 100 and wastes 210')
    print('  reviews; the narrow net catches 45 and wastes 55.  If the 100 are frauds, those')
    print('  are not two versions of the same system.  And the narrow net - the one that')
    print(f'  misses 55 of them - has the HIGHER accuracy, {accuracy(ye, pb):.4f} against '
          f'{accuracy(ye, pa):.4f}.')
    print('  One number cannot carry two independent quantities; report both, or report the')
    print('  one your job depends on.')

    # ------------------------------------------------------------------ 7
    header('7. Silent failure #3: ROC-AUC and PR-AUC put two models in opposite orders')
    yb, broad, sharp = broad_and_sharp()
    print(f'  {int(yb.sum())} positives in {len(yb)} rows ({100 * yb.mean():.1f}%), scored by two models.')
    buried = np.sort(sharp[yb == 1])[:10]
    pct = float(np.mean([(sharp[yb == 0] < v).mean() for v in buried]))
    print('  broad spreads the whole population out.  sharp puts 10 positives above every')
    shares = sorted((sharp[yb == 0] < v).mean() for v in buried)
    print(f'  negative and buries the other 10 low: those ten outrank between {100 * shares[0]:.0f}% and '
          f'{100 * shares[-1]:.0f}%')
    print(f'  of the negatives, {100 * pct:.0f}% on average.  That is where its ROC-AUC of '
          f'{roc_auc_rank(yb, sharp):.4f} comes')
    print('  from:')
    print(f'  (10 x 1.0000 + 10 x {pct:.4f}) / 20 = {(10 + 10 * pct) / 20:.4f}.')
    print()
    print(f'  {"":<8} {"ROC-AUC":>9} {"avg prec":>10} {"top 10":>8} {"top 20":>8} '
          f'{"top 50":>8} {"top 100":>9}')
    for nm, sc in (('broad', broad), ('sharp', sharp)):
        hits = [precision_at_k(yb, sc, k)[0] for k in (10, 20, 50, 100)]
        print(f'  {nm:<8} {roc_auc_rank(yb, sc):>9.4f} {average_precision(yb, sc):>10.4f} '
              f'{hits[0]:>8} {hits[1]:>8} {hits[2]:>8} {hits[3]:>9}')
    print('\n  ROC-AUC prefers broad by a wide margin.  Average precision prefers sharp by a')
    print('  wide margin.  Neither is a bug.  Look at the last four columns: if an analyst')
    print(f'  can review ten rows a day, sharp hands them {precision_at_k(yb, sharp, 10)[0]} real cases and broad '
          f'hands them {precision_at_k(yb, broad, 10)[0]}.')
    weakest = broad[yb == 1].min()
    above = int((broad[yb == 0] > weakest).sum())
    n_neg = int((1 - yb).sum())
    flagged = int((broad >= weakest).sum())
    print('\n  Why they disagree is one denominator.  To catch all 20 positives, broad has')
    print(f'  to come down to a score of {weakest:.4f}, and {above} negatives sit above that line.')
    print(f'  The ROC x-axis divides those {above} by all {n_neg:,} negatives, so they move it')
    print(f'  by {above / n_neg:.4f} - under seven percent of the way across, with the curve')
    print(f'  already at the top.  The same {above} are {above}/{flagged} = {above / flagged:.4f} of everything you')
    print(f'  flagged to get there: {100 * above / flagged:.0f}% of the review queue is wrong.  Precision says so')
    print('  and the ROC curve does not.  One set of mistakes, two denominators, and only one')
    print('  of them is the size of your working day.')
    print('  Use ROC-AUC when both classes matter and are comparable in size; use average')
    print('  precision when the positives are rare and the cost is the review queue.')

    # ------------------------------------------------------------------ 8
    header('8. Why the ROC axes are those two numbers and not any others')
    print('  recall = tp / (tp + fn) counts only rows that really are positive.')
    print('  FPR    = fp / (fp + tn) counts only rows that really are negative.')
    print('  Neither denominator mixes the classes, so neither axis knows how many of')
    print('  each class there are.  Going up the y-axis is the benefit (more of the real')
    print('  positives caught); going right along the x-axis is the cost (more of the real')
    print('  negatives falsely flagged).  Each threshold is one point; sweeping every')
    print('  threshold draws the curve, and the diagonal is what guessing at random does.')
    print('\n  That independence is measurable.  Keep all 20 positives of the broad model')
    print('  and throw negatives away at random - the model and its ranking never change:\n')
    print('  Each row below is the mean over 50 different random draws, so that the')
    print('  numbers are about the class balance and not about which negatives survived.\n')
    print(f'  {"negatives kept":>15} {"positive rate":>14} {"ROC-AUC":>9} {"avg precision":>15}')
    bal = []
    for keep in (1980, 495, 124, 40):
        aucs, aps, rates = [], [], []
        for r in range(50):
            yr, sr = rebalance(yb, broad, keep, seed=r)
            aucs.append(roc_auc_rank(yr, sr))
            aps.append(average_precision(yr, sr))
            rates.append(float(yr.mean()))
        bal.append((keep, np.mean(rates), np.mean(aucs), np.mean(aps)))
        print(f'  {keep:>15} {np.mean(rates):>14.4f} {np.mean(aucs):>9.4f} {np.mean(aps):>15.4f}')
    auc_spread = max(b[2] for b in bal) - min(b[2] for b in bal)
    ap_spread = max(b[3] for b in bal) - min(b[3] for b in bal)
    print(f'\n  The positives go from {100 * bal[0][1]:.0f}% of the rows to {100 * bal[-1][1]:.0f}%.  Over that range the')
    print(f'  mean ROC-AUC moves by {auc_spread:.5f} and the mean average precision moves by')
    print(f'  {ap_spread:.5f} - {ap_spread / auc_spread:.0f} times further.')
    singles = [roc_auc_rank(*rebalance(yb, broad, 40, seed=r)) for r in range(50)]
    print(f'\n  "Nearly invariant" is the honest phrase, and the word nearly is doing work:')
    print(f'  the 50 individual draws at 40 negatives run from {min(singles):.4f} to {max(singles):.4f}.')
    print('  With only 40 negatives left, which 40 you kept matters a lot; what does not')
    print('  drift is the average over draws.')
    print('  Precision has fp and tp in the same denominator, so it feels the balance;')
    print('  both ROC axes are normalised inside one class, so they do not.')
    print('\n  That near-invariance is the reason to use ROC-AUC and the reason not to.  It')
    print('  is what you want when the test set was sampled with a different class balance')
    print('  from production.  It is exactly what you do not want when the rarity of the')
    print('  positives IS the problem, because the number then stays high while the review')
    print('  queue fills with false positives.')

    # ------------------------------------------------------------------ 9
    header('9. AUC is a pair count, and that is why LeetCode 493 is on this page')
    P, N = int(yb.sum()), int((1 - yb).sum())
    print('  ROC-AUC is not an area you have to integrate.  It is exactly:')
    print('  "pick one positive and one negative at random - how often does the positive')
    print('  score higher?", counting a tie as half.\n')
    print(f'  pairs to consider: {P} positives x {N} negatives = {P * N:,}')
    print(f'    the definition, every pair:  {roc_auc_pairs(yb, broad):.10f}')
    print(f'    from the ranks, O(n log n):  {roc_auc_rank(yb, broad):.10f}')
    print(f'    difference:                  {abs(roc_auc_pairs(yb, broad) - roc_auc_rank(yb, broad)):.2e}')
    for n_, rate in ((10000, 0.01), (10000, 0.5)):
        pp = int(n_ * rate)
        print(f'    at {n_:,} rows with {100 * rate:g}% positives: {pp:,} x {n_ - pp:,} = '
              f'{pp * (n_ - pp):,} pairs')
    print('\n  Sorting once and reading the ranks gives the same number without building the')
    print(f'  {P * N:,} pairs.  LeetCode 493 is that move with a different condition on the pair,')
    print('  and it is in this day because the trap is identical: the answer is a count over')
    print('  pairs, and the one thing you must not do is enumerate them.')
    print('  Ties are not a detail.  A model that gives every row the same score wins no')
    print(f'  pair and loses none, so its AUC is exactly 0.5000: '
          f'{roc_auc_pairs(np.r_[np.ones(5), np.zeros(5)].astype(int), np.full(10, 0.3)):.4f}.')
    print('  Score with a coarse grid - rounded probabilities, integer buckets - and a large')
    print('  share of your pairs become ties that silently count half.')

    # ------------------------------------------------------------------ 10
    header('10. Silent failure #4: plain k-fold hands you a fold with no positives')
    print('  Same 2,000 rows, 20 positives, cut into 10 folds.\n')
    plain10 = kfold_indices(len(y), 10, seed=0)
    strat10 = stratified_kfold_indices(y, 10)
    print(f'  shuffle then cut   positives per fold {[int(y[f].sum()) for f in plain10]}')
    print(f'  cut each class     positives per fold {[int(y[f].sum()) for f in strat10]}')
    empty_runs = 0
    empty_folds = 0
    for seed in range(200):
        counts = [int(y[f].sum()) for f in kfold_indices(len(y), 10, seed=seed)]
        empty_folds += sum(1 for c in counts if c == 0)
        empty_runs += 1 if min(counts) == 0 else 0
    print(f'\n  Over 200 different shuffles, {empty_runs} of them ({100 * empty_runs / 200:.0f}%) produce at least one')
    print(f'  fold with zero positives, and {empty_folds} of the {200 * 10:,} folds are empty of positives.')
    dead = [f for f in plain10 if int(y[f].sum()) == 0]
    if dead:
        pf = predict_at(oof[dead[0]], 0.10)
        print(f'  On such a fold recall is 0 / 0 = {recall(y[dead[0]], pf)} and F1 is {f1(y[dead[0]], pf)}.')
        print('  Average that column with np.mean and the whole cross-validation score becomes')
        print('  nan; average it with np.nanmean and the fold quietly leaves the average.')
    ps = sorted(len(f) for f in plain10)
    ss = sorted(len(f) for f in strat10)
    print(f'\n  Stratifying costs one line, and here it does not even change the fold sizes:')
    print(f'  plain {ps[0]}-{ps[-1]} rows per fold, stratified {ss[0]}-{ss[-1]}.')
    tiny = np.r_[np.ones(9), np.zeros(94)].astype(int)
    print('  It can change them when the counts do not divide evenly, because each class is')
    print(f'  cut separately: 103 rows with 9 positives into 5 folds gives')
    print(f'  sizes {[len(f) for f in stratified_kfold_indices(tiny, 5)]} instead of '
          f'{[len(f) for f in kfold_indices(103, 5, seed=1)]} - one row per class rather')
    print('  than one row overall.')

    # ------------------------------------------------------------------ 11
    header('11. Silent failure #5: averaging the folds is not pooling the predictions')
    print('  Five folds, the same out-of-fold scores, cut at 0.10.  Two ways to turn five')
    print('  folds into one number.\n')
    pf_f1 = per_fold_metric(y, oof, folds, f1, 0.10)
    pf_pr = per_fold_metric(y, oof, folds, precision, 0.10)
    pooled_f1 = f1(y, predict_at(oof, 0.10))
    pooled_pr = precision(y, predict_at(oof, 0.10))
    print(f'  per-fold F1         {np.round(pf_f1, 4).tolist()}   mean {np.nanmean(pf_f1):.4f}')
    print(f'  per-fold precision  {np.round(pf_pr, 4).tolist()}   mean {np.nanmean(pf_pr):.4f}')
    print(f'\n  pooled F1 {pooled_f1:.4f}, pooled precision {pooled_pr:.4f}.')
    print(f'  The precision gap is {np.nanmean(pf_pr) - pooled_pr:+.4f}, and it comes from two folds that')
    flag = [int(predict_at(oof[f], 0.10).sum()) for f in folds]
    print(f'  flagged very few rows and got them all right.  The five folds flagged {flag}')
    print('  rows, and the two that flagged the fewest scored a precision of 1.0000.  The')
    print(f'  mean treats a fold that flagged {min(flag)} rows as equal to one that flagged {max(flag)}.')
    print('  Precision, recall and F1 are ratios, and the mean of ratios is not the ratio of')
    print('  the totals.  Pool the out-of-fold predictions and compute the metric once -')
    print('  keep the per-fold numbers for the spread, not for the headline.')

    # ------------------------------------------------------------------ 12
    header('12. Silent failure #6: three kinds of leakage, measured and ranked')
    Xg, yg, g = grouped_rows()
    n_g = len(yg)
    rows = []
    for nm, fl in (('random 5-fold', kfold_indices(n_g, 5, seed=0)),
                   ('group 5-fold', group_kfold_indices(g, 5))):
        accs = []
        for f in fl:
            tr = np.setdiff1d(np.arange(n_g), f)
            accs.append(float(np.mean(one_nearest_neighbour(Xg[tr], yg[tr], Xg[f]) == yg[f])))
        rows.append((nm, float(np.mean(accs))))
    print('  (a) The same customer on both sides of the split.  150 customers, 4 rows each,')
    print('      the label belongs to the customer.  Classifier: copy the nearest row.')
    for nm, v in rows:
        print(f'      {nm:<16} accuracy {v:.4f}')
    print(f'      A {rows[0][1]:.4f} that is really {rows[1][1]:.4f}.  The model learned nothing; it')
    print('      recognised the three siblings of each test row sitting in the training set.')

    print('\n  (b) Choosing the columns before splitting.  60 rows, 400 columns of noise, a')
    print('      coin-flip label - there is nothing in this table to find.')
    out_scores, in_scores = [], []
    for seed in range(20):
        Xn, yn = pure_noise_table(60, 400, seed=seed)
        fl = stratified_kfold_indices(yn, 5)
        cols_all = select_by_correlation(Xn, yn, 5)
        a_, b_ = [], []
        for f in fl:
            tr = np.setdiff1d(np.arange(60), f)
            a_.append(nearest_centroid_accuracy(Xn[tr][:, cols_all], yn[tr], Xn[f][:, cols_all], yn[f]))
            cols_tr = select_by_correlation(Xn[tr], yn[tr], 5)
            b_.append(nearest_centroid_accuracy(Xn[tr][:, cols_tr], yn[tr], Xn[f][:, cols_tr], yn[f]))
        out_scores.append(np.mean(a_))
        in_scores.append(np.mean(b_))
    print(f'      pick the 5 best columns on all 60 rows, then cross-validate: '
          f'{np.mean(out_scores):.4f}')
    print(f'      pick them inside each fold, from that fold\'s training rows only: '
          f'{np.mean(in_scores):.4f}')
    print(f'      Both are means over {len(out_scores)} independently drawn noise tables; the honest')
    print(f'      number is {np.mean(in_scores):.4f} against a true 0.5.')
    print('      The selection saw every label, so it found the columns that happen to match')
    print('      the test rows too.  A cross-validation that starts after the selection')
    print('      cannot undo it.')

    print('\n  (c) Standardising before splitting.  A different 60 rows this time - 8 columns')
    print('      with real signal in them, and logistic regression rather than a centroid.')
    print('      The only thing that changes between the two numbers below is where the')
    print('      column means and standard deviations were fitted.')
    Xs, ys = imbalanced(60, 0.3, sep=1.2, d=8, seed=5)
    fl = stratified_kfold_indices(ys, 5)
    inside = cv_predictions(Xs, ys, fl, lambda: LogisticRegression(), scale_inside=True)
    outside = cv_predictions(Xs, ys, fl, lambda: LogisticRegression(), scale_inside=False)
    print(f'      inside the fold  accuracy {accuracy(ys, predict_at(inside, 0.5)):.4f}')
    print(f'      over all rows    accuracy {accuracy(ys, predict_at(outside, 0.5)):.4f}')
    print(f'      largest difference between any two scores: {np.abs(inside - outside).max():.2e}')
    print('      This one is almost nothing here, and saying so is the point: the three are')
    print('      not equally bad.  A mean and a standard deviation are two numbers summarising')
    sizes = sorted(len(f) for f in fl)
    print(f'      the whole column, so the {sizes[0]} to {sizes[-1]} rows a fifth holds out barely move')
    print('      them while the rows are interchangeable.  It stops being harmless when the')
    print('      test rows are a different period or a different population - then the')
    print('      global mean is a fact about the future.  The two above are the ones that')
    print('      turn noise into a result.')

    # ------------------------------------------------------------------ 13
    header('13. Regularisation: trading training accuracy for the accuracy you will see')
    Xtr, ytr = sparse_signal(120, 40, seed=7)
    Xte, yte = sparse_signal(4000, 40, seed=8)
    fl = stratified_kfold_indices(ytr, 5)
    print('  120 training rows, 40 columns, only 4 of which affect the label.  A larger')
    print('  L2 penalty means the weights have to earn their size.\n')
    print(f'  {"lambda":>8} {"train":>8} {"5-fold CV":>11} {"test (4,000 rows)":>19} {"sum |w|":>9}')
    curve = []
    for lam in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        m = LogisticRegression(l2=lam, lr=0.5, n_iter=800).fit(Xtr, ytr)
        o = cv_predictions(Xtr, ytr, fl,
                           lambda l=lam: LogisticRegression(l2=l, lr=0.5, n_iter=800),
                           scale_inside=False)
        row = (lam, accuracy(ytr, m.predict(Xtr)), accuracy(ytr, predict_at(o, 0.5)),
               accuracy(yte, m.predict(Xte)), float(np.abs(m.w).sum()))
        curve.append(row)
        print(f'  {lam:>8.3f} {row[1]:>8.4f} {row[2]:>11.4f} {row[3]:>19.4f} {row[4]:>9.2f}')
    best_test = max(curve, key=lambda r: r[3])
    top_cv = max(r[2] for r in curve)
    tied = [r for r in curve if r[2] == top_cv]
    best_cv = tied[-1]                      # among equals, keep the strongest penalty
    no_pen = curve[0]
    print(f'\n  Unpenalised, the model gets {no_pen[1]:.4f} on the rows it trained on and '
          f'{no_pen[3]:.4f} on')
    print(f'  fresh rows - a gap of {no_pen[1] - no_pen[3]:.4f} that no amount of staring at the training')
    print(f'  score would reveal.  The best test score is {best_test[3]:.4f} at lambda {best_test[0]}.')
    print(f'\n  Cross-validation never sees those 4,000 rows.  It scores {top_cv:.4f} at '
          f'{len(tied)} different')
    print(f'  lambdas ({", ".join(str(r[0]) for r in tied)}) and cannot separate them, because 120 rows')
    print(f'  split five ways means the CV accuracy can only move in steps of 1/120 = {1 / 120:.4f}.')
    print(f'  Breaking the tie toward the strongest penalty picks lambda {best_cv[0]} and scores')
    print(f'  {best_cv[3]:.4f} on the test rows - the best of the seven.  Taking the first of the tied')
    print(f'  values instead gives lambda {tied[0][0]} and {tied[0][3]:.4f}.  The rule is worth having')
    print('  because among settings you cannot tell apart, the one that fits least is the one')
    print('  least likely to be fitting the validation folds.')
    print('  Either way the knob was chosen without spending the test set, which is the')
    print('  whole point - see the next section.')

    # ------------------------------------------------------------------ 14
    header('14. Silent failure #7: the set you chose on is no longer a test set')
    print('  Score m models on one labelled set, keep the best, then score that same')
    print('  model on a second set drawn the same way.  Every model here is a coin flip')
    print('  and so is every label, so the true accuracy of all of them is 0.5.\n')
    print(f'  {"models tried":>13} {"best, on the set you chose on":>31} {"same model, fresh set":>23}')
    for m in (1, 10, 50, 200):
        a_, b_ = best_of_m_on_one_set(m, n=200, reps=40, seed=1000)
        print(f'  {m:>13} {a_:>31.4f} {b_:>23.4f}')
    print('\n  Nothing was learned and nothing improved.  The first column rises because you')
    print('  are reporting a maximum over m noisy measurements, and the maximum of noise')
    print('  grows with m.  Every hyper-parameter you sweep, every threshold you tune and')
    print('  every architecture you try on the same held-out set adds to m.')
    print('  The fix is the split you already have: choose on the validation folds, and')
    print('  look at the test set once, at the end, after you have stopped choosing.')

    # ------------------------------------------------------------------ 15
    header('15. LeetCode 493 - Reverse Pairs')
    print('  Count the pairs i < j with nums[i] > 2 * nums[j].\n')
    for a_ in ([1, 3, 2, 3, 1], [2, 4, 3, 5, 1]):
        print(f'  {a_} -> {SolutionReversePairs().reversePairs(a_)}   '
              f'(brute force {reverse_pairs_brute(a_)})')
    n_lc = 50000
    print(f'  LeetCode allows up to {n_lc:,} elements, which is {n_lc * (n_lc - 1) // 2:,} pairs -')
    print('  the bound that rules out looking at them one at a time.')
    big = np.random.default_rng(4).integers(-1000, 1000, 2000).tolist()
    print(f'\n  2,000 numbers: {len(big) * (len(big) - 1) // 2:,} pairs exist, '
          f'{SolutionReversePairs().reversePairs(big):,} of them qualify,')
    print('  and merge sort never looks at a pair.  When the two halves are sorted and')
    print('  left[i] > 2 * right[j], every element after j on the right also qualifies, so')
    print('  one comparison settles a whole block - and because both halves are sorted, the')
    print('  right-hand pointer only ever moves forward across the whole pass.')
    print('\n  The trap is the condition.  Counting during the merge instead of in a pass of')
    print('  its own looks like free work, but the merge advances on left[i] <= right[j]')
    print('  while the pairs we want need left[i] > 2 * right[j] - two different orders.')
    bad, wrong, over = None, 0, 0
    for seed in range(2000):
        r = np.random.default_rng(seed)
        cand = r.integers(-20, 20, int(r.integers(2, 10))).tolist()
        got, want = reverse_pairs_during_merge(cand), reverse_pairs_brute(cand)
        if got != want:
            wrong += 1
            over += 1 if got > want else 0
            if bad is None:
                bad = (cand, got, want)
    print(f'  Over 2,000 random arrays it disagrees {wrong} times ({100 * wrong / 2000:.1f}%), and it')
    print(f'  overcounts in {over} of them - every disagreement is a count that is too low.')
    print(f'  First one: {bad[0]}')
    print(f'  it says {bad[1]}, the answer is {bad[2]}.  Count in one pass, merge in the next.')

    # ------------------------------------------------------------------ 16
    header('16. LeetCode 169 - Majority Element')
    print('  Return the value that appears more than n / 2 times.\n')
    for a_ in ([3, 2, 3], [2, 2, 1, 1, 1, 2, 2]):
        print(f'  {a_} -> {SolutionMajorityElement().majorityElement(a_)}')
    print('\n  Boyer-Moore keeps one candidate and one count: equal values add, different')
    print('  values cancel, and a value holding more than half the array cannot be cancelled')
    print('  away.  O(n) time, one integer of memory, no sorting and no hash map.')
    print(f'\n  It is the baseline from section 4: on the 2,000-row table it answers {maj},')
    print(f'  which is right {base:.4f} of the time.  The follow-up LeetCode does not ask is')
    print('  what happens without the promise that a majority exists:')
    print(f'    [1, 2, 3] -> Boyer-Moore says '
          f'{SolutionMajorityElement().majorityElement([1, 2, 3])}, which appears once in three rows.')
    print(f'    with a second pass to check: {majority_if_any([1, 2, 3])}')
    print('  It returns a number either way and never raises, which is the same shape as')
    print('  every failure on this page: the answer is well formed and means nothing.')

    # ------------------------------------------------------------------ 17
    header('17. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
