"""Day 45 - The gradient descent family: SGD, momentum, AdaGrad, RMSProp, Adam, AdamW.

Run `python optimizers.py` to print the whole story.  Timings depend on the machine;
every other number is exact and checked by an assert at the bottom.
"""

import math

import numpy as np


# ================================================================ 1. the update rules
# Every optimiser has the same two-line interface, so the training loop never changes:
#     g = grad(w);  w = opt.step(w, g)
class SGD:
    """w -= lr * g, optionally with a velocity that remembers past gradients."""

    name = 'SGD'

    def __init__(self, lr=0.1, momentum=0.0, nesterov=False):
        self.lr, self.momentum, self.nesterov = lr, momentum, nesterov
        self.v = None
        if momentum:
            self.name = f'SGD+Nesterov({momentum})' if nesterov else f'SGD+momentum({momentum})'

    def step(self, w, g):
        if self.v is None:
            self.v = np.zeros_like(w)
        if not self.momentum:
            return w - self.lr * g
        self.v = self.momentum * self.v + g            # a running sum of gradients
        step = self.momentum * self.v + g if self.nesterov else self.v
        return w - self.lr * step


class AdaGrad:
    """Divide by the square root of every squared gradient ever seen."""

    name = 'AdaGrad'

    def __init__(self, lr=0.5, eps=1e-8):
        self.lr, self.eps, self.s = lr, eps, None

    def step(self, w, g):
        if self.s is None:
            self.s = np.zeros_like(w)
        self.s += g * g                                 # never decays: the step keeps shrinking
        return w - self.lr * g / (np.sqrt(self.s) + self.eps)


class RMSProp:
    """AdaGrad with a forgetting factor, so the step size can recover."""

    name = 'RMSProp'

    def __init__(self, lr=0.05, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.s = lr, rho, eps, None

    def step(self, w, g):
        if self.s is None:
            self.s = np.zeros_like(w)
        self.s = self.rho * self.s + (1 - self.rho) * g * g
        return w - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    """Momentum on the gradient (m) and on its square (v), each bias-corrected.

    The flags exist to show what breaks when a detail is dropped:
      bias_correction=False  the first steps are far too small
      eps_inside=True        sqrt(v + eps) instead of sqrt(v) + eps
      weight_decay, decoupled   L2-in-the-gradient (Adam) vs AdamW
    """

    def __init__(self, lr=0.05, b1=0.9, b2=0.999, eps=1e-8, bias_correction=True,
                 eps_inside=False, weight_decay=0.0, decoupled=False):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.bias_correction, self.eps_inside = bias_correction, eps_inside
        self.weight_decay, self.decoupled = weight_decay, decoupled
        self.m = self.v = None
        self.t = 0
        self.name = 'AdamW' if (weight_decay and decoupled) else 'Adam'
        if weight_decay and not decoupled:
            self.name = 'Adam + L2'

    def step(self, w, g):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.t += 1
        if self.weight_decay and not self.decoupled:
            g = g + self.weight_decay * w              # L2 penalty folded into the gradient
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        if self.bias_correction:
            mh = self.m / (1 - self.b1 ** self.t)
            vh = self.v / (1 - self.b2 ** self.t)
        else:
            mh, vh = self.m, self.v
        denom = np.sqrt(vh + self.eps) if self.eps_inside else np.sqrt(vh) + self.eps
        w = w - self.lr * mh / denom
        if self.weight_decay and self.decoupled:
            w = w - self.lr * self.weight_decay * w    # decay applied straight to w
        return w


def make_adamw(lr=0.05, weight_decay=0.1):
    return Adam(lr=lr, weight_decay=weight_decay, decoupled=True)


# ================================================================ 2. the test landscape
class Quadratic:
    """f(w) = 0.5 * sum(curv * w^2): a bowl stretched into a ravine by `curv`.

    Everything about plain gradient descent on this bowl is known in closed form,
    so the measured step counts can be checked against the theory.
    """

    def __init__(self, curv=(1.0, 20.0)):
        self.curv = np.asarray(curv, dtype=float)

    def loss(self, w):
        return float(0.5 * np.sum(self.curv * w * w))

    def grad(self, w):
        return self.curv * w

    @property
    def lmax(self):
        return float(self.curv.max())

    @property
    def lmin(self):
        return float(self.curv.min())

    @property
    def kappa(self):
        """Condition number: how much longer the ravine is than it is wide."""
        return self.lmax / self.lmin

    @property
    def lr_divergence(self):
        """Plain GD on a quadratic converges exactly while lr < 2 / lmax."""
        return 2.0 / self.lmax

    @property
    def lr_best(self):
        return 2.0 / (self.lmin + self.lmax)

    @property
    def rate_gd(self):
        """Error multiplier per step at the best lr."""
        return (self.kappa - 1) / (self.kappa + 1)

    @property
    def rate_momentum(self):
        """Heavy-ball with the best beta: the same expression with sqrt(kappa)."""
        r = math.sqrt(self.kappa)
        return (r - 1) / (r + 1)

    @property
    def beta_best(self):
        return self.rate_momentum ** 2

    @property
    def lr_momentum_best(self):
        """Heavy-ball wants its own step size, not gradient descent's."""
        return (2.0 / (math.sqrt(self.lmax) + math.sqrt(self.lmin))) ** 2

    def steps_for(self, rate, tol=1e-6):
        """How many steps a per-step error multiplier `rate` needs to reach tol."""
        return math.log(tol) / math.log(rate)


def run(opt, prob, w0, steps=200):
    """Return the trajectory of w, one row per step (including the start)."""
    w = np.asarray(w0, dtype=float).copy()
    out = [w.copy()]
    for _ in range(steps):
        w = opt.step(w, prob.grad(w))
        if not np.all(np.isfinite(w)):
            out.append(w.copy())
            break
        out.append(w.copy())
    return np.array(out)


def steps_to(opt, prob, w0, tol=1e-6, cap=100_000):
    """How many steps until |w| falls below tol.  cap+1 means 'never got there'."""
    w = np.asarray(w0, dtype=float).copy()
    start = float(np.linalg.norm(w))
    for k in range(1, cap + 1):
        w = opt.step(w, prob.grad(w))
        if not np.all(np.isfinite(w)):
            return cap + 1
        if np.linalg.norm(w) <= tol * start:
            return k
    return cap + 1


def multiplier(lr, prob):
    """Plain GD on a quadratic multiplies each coordinate by (1 - lr * curvature)
    every step.  The whole run converges exactly when all of those are below 1."""
    return float(np.max(np.abs(1.0 - lr * prob.curv)))


def diverges(lr, prob, w0, steps=20_000):
    """Run it and look, rather than trusting the formula."""
    w = np.asarray(w0, dtype=float).copy()
    start = float(np.linalg.norm(w))
    for _ in range(steps):
        w = w - lr * prob.grad(w)
        if not np.all(np.isfinite(w)) or np.linalg.norm(w) > 1e3 * start:
            return True
    return False


# ================================================================ 3. a real problem, in mini-batches
class LeastSquares:
    """(1/n) * ||Xw - y||^2.  One column is deliberately scaled up, so the bowl is a
    ravine for exactly the same reason the toy quadratic is."""

    def __init__(self, n=400, scales=(1.0, 1.0, 8.0), noise=0.3, seed=45):
        rng = np.random.default_rng(seed)
        d = len(scales)
        self.X = rng.normal(size=(n, d)) * np.asarray(scales)
        self.w_true = np.array([2.0, -1.0, 0.5])[:d]
        self.y = self.X @ self.w_true + noise * rng.normal(size=n)
        self.n = n
        self.w_star = np.linalg.lstsq(self.X, self.y, rcond=None)[0]   # the exact optimum
        self.best = float(np.mean((self.X @ self.w_star - self.y) ** 2))

    def grad(self, w, idx=None):
        X = self.X if idx is None else self.X[idx]
        y = self.y if idx is None else self.y[idx]
        return 2.0 / len(y) * (X.T @ (X @ w - y))

    def loss(self, w):
        """Excess loss over the exact least-squares optimum, so it goes to zero."""
        return float(np.mean((self.X @ w - self.y) ** 2)) - self.best

    @property
    def kappa(self):
        c = self.curvature
        return float(c.max() / c.min())

    @property
    def curvature(self):
        return np.linalg.eigvalsh(2.0 / self.n * self.X.T @ self.X)

    @property
    def lr_divergence(self):
        return 2.0 / float(self.curvature.max())

    def sorted_by_target(self):
        """The same data with the rows in sorted order - the shape real data arrives in
        when it was grouped by label or collected over time."""
        import copy
        other = copy.deepcopy(self)
        idx = np.argsort(other.y)
        other.X, other.y = other.X[idx], other.y[idx]
        return other

    @property
    def lr_divergence_single(self):
        """A single row has its own, much sharper curvature: 2 * ||x||^2.  A batch of
        one has to respect that, not the average - which is why the smaller the batch,
        the smaller the learning rate it survives."""
        worst = float(np.max(2.0 * np.sum(self.X * self.X, axis=1)))
        return 2.0 / worst


def sgd_epochs(opt, prob, w0, batch, epochs, shuffle=True, seed=0):
    """Mini-batch training.  Returns the loss after every epoch."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, dtype=float).copy()
    order = np.arange(prob.n)
    hist = []
    for _ in range(epochs):
        if shuffle:
            rng.shuffle(order)
        for s in range(0, prob.n, batch):
            idx = order[s:s + batch]
            w = opt.step(w, prob.grad(w, idx))
            if not np.all(np.isfinite(w)):
                hist.append(float('inf'))
                return w, hist
        hist.append(prob.loss(w))
    return w, hist


# ================================================================ 4. LeetCode 462
class Solution462:
    """Minimise the total distance to a single point: the answer is the median."""

    def minMoves2(self, nums: list[int]) -> int:
        nums = sorted(nums)
        mid = nums[len(nums) // 2]
        return sum(abs(x - mid) for x in nums)


def moves_to(nums, c):
    return sum(abs(x - c) for x in nums)


def minmoves_mean(nums):
    """The tempting wrong answer: the mean minimises squared error, not absolute error."""
    c = round(sum(nums) / len(nums))
    return moves_to(nums, c)


def subgradient_median(nums, lr=0.5, steps=400, c0=0.0):
    """Gradient descent on sum|x - c|.  The derivative of |x - c| is the sign, which is
    why this walks to the median rather than the mean - and why it never settles
    exactly: the step size, not the slope, decides where it stops."""
    c = float(c0)
    traj = [c]
    for k in range(1, steps + 1):
        g = float(sum(np.sign(c - x) for x in nums))
        c -= lr / math.sqrt(k) * g            # a 1/sqrt(k) schedule, or it oscillates forever
        traj.append(c)
    return c, traj


# ================================================================ 5. LeetCode 852
class Solution852:
    """Find the top of a mountain by binary-searching the sign of the slope."""

    def peakIndexInMountainArray(self, arr: list[int]) -> int:
        lo, hi = 0, len(arr) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < arr[mid + 1]:       # still going up: the peak is to the right
                lo = mid + 1
            else:                             # already coming down: mid could be the peak
                hi = mid
        return lo


def peak_linear(arr):
    """O(n): the answer a scan gives, used only to check the binary search."""
    return int(np.argmax(arr))


def peak_probes(arr):
    """How many elements the binary search actually looks at."""
    lo, hi, probes = 0, len(arr) - 1, 0
    while lo < hi:
        mid = (lo + hi) // 2
        probes += 2
        if arr[mid] < arr[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return probes


def ternary_min(f, lo, hi, iters=100):
    """The continuous version: no derivative, just 'which third can I throw away'."""
    for _ in range(iters):
        a = lo + (hi - lo) / 3
        b = hi - (hi - lo) / 3
        if f(a) < f(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def mountain(n=4001, peak=None, seed=45):
    """A strictly-increasing-then-decreasing array."""
    peak = n // 3 if peak is None else peak
    up = np.arange(1, peak + 2) * 3
    down = up[-1] - np.arange(1, n - peak) * 2
    arr = np.concatenate([up, down]).astype(int)
    return arr.tolist(), peak


def steps_to_loss(opt, prob, w0, tol=1e-8, cap=5000):
    """Full-batch steps until the excess loss drops below tol."""
    w = np.asarray(w0, dtype=float).copy()
    for k in range(1, cap + 1):
        with np.errstate(over='ignore', invalid='ignore'):
            w = opt.step(w, prob.grad(w))
        if not np.all(np.isfinite(w)):
            return cap + 1
        if prob.loss(w) <= tol:
            return k
    return cap + 1


def best_lr_loss(make_opt, prob, w0, grid=None, tol=1e-8, cap=3000):
    if grid is None:
        grid = [10 ** e for e in np.arange(-3.0, 1.51, 0.1)]
    best = (None, cap + 1)
    for lr in grid:
        k = steps_to_loss(make_opt(lr), prob, w0, tol=tol, cap=cap)
        if k < best[1]:
            best = (lr, k)
    return best


def best_lr(make_opt, prob, w0, grid=None, tol=1e-6, cap=5000):
    """Sweep the learning rate and keep the one that reaches tol in fewest steps.

    Comparing optimisers at one shared learning rate is not a fair fight: AdaGrad,
    RMSProp and Adam divide the gradient by its own size, so their lr means something
    different.  Tuning each one is what actually happens in practice.
    """
    if grid is None:
        grid = [10 ** e for e in np.arange(-3.0, 1.51, 0.1)]
    best = (None, cap + 1)
    for lr in grid:
        k = steps_to(make_opt(lr), prob, w0, tol=tol, cap=cap)
        if k < best[1]:
            best = (lr, k)
    return best


# ================================================================ 6. Adam's details
def bias_factor(t, b1=0.9, b2=0.999):
    """How much larger an un-corrected Adam step is than a corrected one, at step t.

    corrected  = (m / (1 - b1^t)) / sqrt(v / (1 - b2^t))
    uncorrected = m / sqrt(v)
    so the ratio is (1 - b1^t) / sqrt(1 - b2^t), which is *above* 1 for a long time.
    """
    return (1 - b1 ** t) / math.sqrt(1 - b2 ** t)


def first_steps(opt, prob, w0, n=3):
    """The size of the first n parameter updates."""
    w = np.asarray(w0, dtype=float).copy()
    sizes = []
    for _ in range(n):
        nw = opt.step(w, prob.grad(w))
        sizes.append(float(np.linalg.norm(nw - w)))
        w = nw
    return sizes


def final_weights(opt, prob, w0, steps=400):
    w = np.asarray(w0, dtype=float).copy()
    for _ in range(steps):
        w = opt.step(w, prob.grad(w))
    return w


# ================================================================ the walkthrough
def header(t):
    print('\n' + '=' * 72 + '\n' + t + '\n' + '=' * 72)


FAMILIES = [
    ('SGD', lambda lr: SGD(lr)),
    ('SGD + momentum 0.9', lambda lr: SGD(lr, 0.9)),
    ('SGD + Nesterov 0.9', lambda lr: SGD(lr, 0.9, True)),
    ('AdaGrad', lambda lr: AdaGrad(lr)),
    ('RMSProp', lambda lr: RMSProp(lr)),
    ('Adam', lambda lr: Adam(lr)),
]


def main() -> None:
    np.seterr(all='ignore')          # a diverging run overflows on purpose
    q = Quadratic()
    w0 = np.array([1.0, 1.0])

    header('1. One training loop, seven update rules')
    print('  every optimiser answers the same two-line question:')
    print('      g = grad(w);  w = opt.step(w, g)')
    print(f'  the bowl: f(w) = 0.5 * ({q.curv[0]:g} w0^2 + {q.curv[1]:g} w1^2), '
          f'start at {tuple(w0.tolist())}')
    print(f'  curvature {q.lmin:g} along w0 and {q.lmax:g} along w1 -> condition number '
          f'{q.kappa:g}: a ravine {q.kappa:g}x longer than it is wide')

    header('2. The learning rate has an exact ceiling')
    print('  plain GD multiplies each coordinate by (1 - lr * curvature) every step,')
    print(f'  so it converges exactly while lr < 2 / {q.lmax:g} = {q.lr_divergence:g}')
    for lr in (0.05, 0.09, 0.0999, 0.1001, 0.11, 0.2):
        mul = multiplier(lr, q)
        print(f'    lr = {lr:<7g} largest multiplier {mul:.5f}  '
              f'{"diverges" if diverges(lr, q, w0) else "converges"}')
    print(f'  below the ceiling, lr 0.09 from (4, 1) still needs '
          f'{steps_to(SGD(lr=0.09), q, [4.0, 1.0])} steps: the steep direction caps the')
    print('  step size while the flat one sets the progress')
    print('  nothing raises: the loss just grows to inf and then to nan')
    traj = run(SGD(lr=0.11), q, w0, 400)
    print(f'    lr = 0.11 after 400 steps: w = {traj[-1]}')

    header('3. Momentum: the same ravine, the square root of the work')
    print(f'  best lr for plain GD      2 / (lmin + lmax) = {q.lr_best:.4f}')
    print(f'  error multiplier per step (kappa - 1)/(kappa + 1) = {q.rate_gd:.4f}')
    print(f'  heavy-ball with its own best lr {q.lr_momentum_best:.4f} and beta {q.beta_best:.4f}')
    print(f'  error multiplier (sqrt(kappa) - 1)/(sqrt(kappa) + 1) = {q.rate_momentum:.4f}')
    g_meas = steps_to(SGD(lr=q.lr_best), q, w0)
    m_meas = steps_to(SGD(lr=q.lr_momentum_best, momentum=q.beta_best), q, w0)
    print(f'  steps to shrink the error a millionfold:')
    print(f'    plain GD   theory {q.steps_for(q.rate_gd):6.1f}   measured {g_meas}')
    print(f'    momentum   theory {q.steps_for(q.rate_momentum):6.1f}   measured {m_meas}'
          f'   ({g_meas / m_meas:.1f}x fewer)')
    print('  that is the whole point: kappa becomes sqrt(kappa) in the step count')
    print('  beta is not a constant of nature - at a fixed lr 0.04 from (4, 1):')
    for beta in (0.0, q.beta_best, 0.9):
        print(f'    beta {beta:<7.4f} {steps_to(SGD(lr=0.04, momentum=beta), q, [4.0, 1.0]):>4} steps')

    header('4. On a real problem, with each learning rate tuned')
    ls = LeastSquares()
    print(f'  least squares, {ls.n} rows x 3 features, one column scaled up 8x')
    print(f'  curvature {np.round(ls.curvature, 2)} -> condition number {ls.kappa:.1f}, '
          f'lr ceiling {ls.lr_divergence:.4f}')
    print('  steps for the excess loss to fall below 1e-8 (lr swept per optimiser):')
    z = np.zeros(3)
    for name, make in FAMILIES:
        lr, k = best_lr_loss(make, ls, z)
        print(f'    {name:<20} best lr {lr:>7.4f}   {k:>4} steps')
    print('  comparing optimisers at one shared lr would be meaningless: AdaGrad,')
    print('  RMSProp and Adam divide by the size of the gradient, so their lr is a')
    print('  different quantity. AdaGrad wins here because this problem never changes;')
    print('  the denominator it keeps adding to is exactly what stalls it on long runs.')

    header('5. Adam without bias correction takes steps that are too big')
    print('  m and v both start at zero, so both are too small early on - but v sits')
    print('  under a square root, and the two errors do not cancel:')
    print('     uncorrected / corrected = (1 - 0.9^t) / sqrt(1 - 0.999^t)')
    for t in (1, 2, 5, 10, 20, 100, 1000, 5000):
        print(f'    step {t:>5}: {bias_factor(t):.3f}x')
    peak = max(range(1, 2001), key=bias_factor)
    print(f'  worst around step {peak} at {bias_factor(peak):.2f}x; '
          f'it is still {bias_factor(1000):.2f}x off after a thousand steps')
    a_on = first_steps(Adam(lr=0.05), ls, z)
    a_off = first_steps(Adam(lr=0.05, bias_correction=False), ls, z)
    print(f'  measured first three step sizes, lr 0.05:')
    print(f'    with correction    {[round(x, 4) for x in a_on]}')
    print(f'    without            {[round(x, 4) for x in a_off]}')
    k_on = steps_to_loss(Adam(lr=0.05), ls, z)
    k_off = steps_to_loss(Adam(lr=0.05, bias_correction=False), ls, z)
    print(f'  steps to 1e-8 at the same lr: with {k_on}, without {k_off}')

    header('6. Where the epsilon goes')
    print('  sqrt(v) + eps  and  sqrt(v + eps)  look like the same guard against')
    print('  dividing by zero. With eps = 1e-8, sqrt(eps) = 1e-4, so they are not:')
    for v in (1e-4, 1e-8, 1e-12, 1e-16):
        a = math.sqrt(v) + 1e-8
        b = math.sqrt(v + 1e-8)
        print(f'    v = {v:<8g} sqrt(v) + eps = {a:.3e}   sqrt(v + eps) = {b:.3e}'
              f'   ratio {b / a:8.1f}x')
    print('  so on coordinates whose gradients are tiny, the second form quietly')
    print('  shrinks the step by orders of magnitude. PyTorch uses sqrt(v) + eps.')

    header('7. Adam + L2 is not AdamW')
    print('  L2 regularisation adds wd * w to the gradient; Adam then divides the whole')
    print('  gradient by sqrt(v). So a weight with large gradients gets its decay')
    print('  divided by a large number - the weights that most need shrinking shrink least.')
    print('  AdamW skips the division and subtracts lr * wd * w from w directly.')
    both = (('Adam + L2', Adam(lr=0.05, weight_decay=0.1)),
            ('AdamW    ', Adam(lr=0.05, weight_decay=0.1, decoupled=True)),
            ('no decay ', Adam(lr=0.05)))
    for nm, opt in both:
        w = final_weights(opt, ls, z)
        print(f'    {nm}  w = {np.round(w, 4)}   |w| = {np.linalg.norm(w):.4f}')
    opt = Adam(lr=0.05)
    w = z.copy()
    for _ in range(400):
        w = opt.step(w, ls.grad(w))
    rv = np.sqrt(opt.v)
    rel = (1 / rv) / np.mean(1 / rv)
    print(f'  sqrt(v) per weight after 400 Adam steps: {np.round(rv, 4).tolist()}')
    print(f'  so Adam + L2 applies decay of strength {np.round(rel, 3).tolist()} '
          f'(average 1) while AdamW applies [1, 1, 1]')
    print(f'  the third feature was scaled 8x, so its gradients are the largest, and it is')
    print(f'  decayed {rel[0] / rel[2]:.0f}x less than weight 0 and {rel[1] / rel[2]:.0f}x '
          f'less than weight 1 - backwards.')
    opt2 = Adam(lr=0.05, weight_decay=0.1)
    w = z.copy()
    for _ in range(400):
        w = opt2.step(w, ls.grad(w))
    print(f'  (measured on a plain Adam run; adding the penalty moves sqrt(v) by at most '
          f'{100 * float(np.max(np.abs(rv - np.sqrt(opt2.v)) / np.sqrt(opt2.v))):.1f}%)')

    header('8. Mini-batches: noise, and why the order matters')
    print(f'  averaged over all {ls.n} rows the lr ceiling is {ls.lr_divergence:.4f},')
    print(f'  but the sharpest single row allows only {ls.lr_divergence_single:.5f} '
          f'({ls.lr_divergence / ls.lr_divergence_single:.1f}x smaller):')
    print('  a batch of one has to respect its own row, not the average.')
    for batch in (1, 8, 32, ls.n):
        _, hist = sgd_epochs(SGD(lr=0.01), ls, z, batch=batch, epochs=30)
        label = 'full batch' if batch == ls.n else f'batch {batch:<3}'
        state = 'blew up' if not math.isfinite(hist[-1]) or hist[-1] > 1e6 else f'{hist[-1]:.1e}'
        print(f'    lr 0.01, {label:<11} after 30 epochs: {state}')
    print(f'  below that single-row ceiling everything survives, and a small batch')
    print(f'  covers far more ground per pass over the data (lr 0.002, 30 epochs):')
    for batch in (1, 8, 32, ls.n):
        _, hist = sgd_epochs(SGD(lr=0.002), ls, z, batch=batch, epochs=30)
        label = 'full batch' if batch == ls.n else f'batch {batch:<3}'
        print(f'    {label:<12} {hist[-1]:.3e}   ({math.ceil(ls.n / batch)} updates per epoch)')
    print('  but it stops improving: with a fixed lr the noise keeps it circling the')
    print('  optimum at a radius the step size sets, while full batch keeps descending.')
    for ep in (10, 30, 100, 300):
        _, h8 = sgd_epochs(SGD(lr=0.002), ls, z, batch=8, epochs=ep)
        _, hf = sgd_epochs(SGD(lr=0.002), ls, z, batch=ls.n, epochs=ep)
        print(f'    after {ep:>3} epochs   batch 8 {h8[-1]:.3e}   full batch {hf[-1]:.3e}')
    print('  shuffling: the usual advice is "always shuffle", but it only pays when')
    print('  the rows are not already in random order (median of 5 seeds, 30 epochs):')
    for label, prob in (('rows already random', ls), ('rows sorted by target', ls.sorted_by_target())):
        a = np.median([sgd_epochs(SGD(lr=0.002), prob, z, 8, 30, True, sd)[1][-1] for sd in range(5)])
        b = np.median([sgd_epochs(SGD(lr=0.002), prob, z, 8, 30, False, sd)[1][-1] for sd in range(5)])
        print(f'    {label:<22} reshuffled {a:.3e}   fixed order {b:.3e}   '
              f'({b / a:>5.1f}x)')
    print('  sorted data makes every batch unrepresentative in the same way every')
    print('  epoch, so the same biased sequence of steps repeats - that is the case')
    print('  shuffling is protecting you from.')

    header('9. LeetCode 462 - Minimum Moves to Equal Array Elements II')
    for nums in ([1, 2, 3], [1, 10, 2, 9], [1, 2, 3, 100]):
        s = Solution462()
        print(f'    {str(nums):<16} median answer {s.minMoves2(nums):>4}   '
              f'using the mean instead {minmoves_mean(nums):>4}')
    nums = [1, 2, 3, 100]
    c, _ = subgradient_median(nums)
    print(f'  gradient descent on sum|x - c| from c = 0 lands at c = {c:.3f}, '
          f'cost {moves_to(nums, c):.1f}')
    print(f'  (any c in [2, 3] costs 100: with an even count the minimiser is an interval)')

    header('10. LeetCode 852 - Peak Index in a Mountain Array')
    arr, peak = mountain()
    s = Solution852()
    got = s.peakIndexInMountainArray(arr)
    print(f'  {len(arr)} elements, peak at index {peak}: binary search says {got}, '
          f'scanning says {peak_linear(arr)}')
    print(f'  elements looked at: {peak_probes(arr)} instead of {len(arr)}')
    print('  the test arr[mid] < arr[mid + 1] is the sign of the slope, so this is')
    print('  gradient ascent that halves the search space instead of taking a step')
    f = lambda x: (x - 1.234567) ** 2 + 3
    x = ternary_min(f, -10, 10)
    print(f'  the continuous twin, ternary search on a unimodal f: x = {x:.6f} '
          f'(true 1.234567), no derivative used')

    check()
    print('\nall assertions passed')


def check() -> None:
    np.seterr(all='ignore')
    q = Quadratic()
    w0 = np.array([1.0, 1.0])
    assert q.kappa == 20.0 and q.lr_divergence == 0.1
    assert abs(q.lr_best - 2 / 21) < 1e-12
    # the ceiling is exactly 2 / lmax, on both sides
    assert not diverges(0.0999, q, w0) and diverges(0.1001, q, w0)
    assert multiplier(0.0999, q) < 1.0 < multiplier(0.1001, q)
    # theory matches measurement to within a step or two
    assert abs(steps_to(SGD(lr=q.lr_best), q, w0) - q.steps_for(q.rate_gd)) <= 2
    m_meas = steps_to(SGD(lr=q.lr_momentum_best, momentum=q.beta_best), q, w0)
    assert m_meas < steps_to(SGD(lr=q.lr_best), q, w0) / 3
    # un-corrected Adam steps are larger, not smaller, and stay wrong for a long time
    assert bias_factor(1) > 3 and max(bias_factor(t) for t in range(1, 2001)) > 6
    assert bias_factor(5000) < 1.01
    ls = LeastSquares()
    z = np.zeros(3)
    assert ls.loss(ls.w_star) <= 1e-12 and ls.loss(z) > 0
    assert first_steps(Adam(lr=0.05, bias_correction=False), ls, z)[0] > \
           first_steps(Adam(lr=0.05), ls, z)[0]
    # eps inside the sqrt makes a far bigger denominator when v is small
    assert math.sqrt(1e-12 + 1e-8) / (math.sqrt(1e-12) + 1e-8) > 50
    # Adam + L2 decays the big-gradient weight least
    w_l2 = final_weights(Adam(lr=0.05, weight_decay=0.1), ls, z)
    w_wd = final_weights(Adam(lr=0.05, weight_decay=0.1, decoupled=True), ls, z)
    w_no = final_weights(Adam(lr=0.05), ls, z)
    assert np.linalg.norm(w_wd) < np.linalg.norm(w_no)
    assert abs(w_l2[2]) > abs(w_wd[2])
    # a single row is far sharper than the average, so batch 1 needs a smaller lr
    assert ls.lr_divergence_single < ls.lr_divergence / 5
    _, h1 = sgd_epochs(SGD(lr=0.01), ls, z, batch=1, epochs=30)
    _, hf = sgd_epochs(SGD(lr=0.01), ls, z, batch=ls.n, epochs=30)
    assert h1[-1] > 1e6 and math.isfinite(hf[-1])
    # below that ceiling a small batch gets further per pass over the data
    _, h8 = sgd_epochs(SGD(lr=0.002), ls, z, batch=8, epochs=30)
    _, hfull = sgd_epochs(SGD(lr=0.002), ls, z, batch=ls.n, epochs=30)
    assert h8[-1] < hfull[-1] / 100
    # ... and then plateaus while full batch keeps going
    _, h8_long = sgd_epochs(SGD(lr=0.002), ls, z, batch=8, epochs=300)
    assert h8_long[-1] > h8[-1] / 10
    # shuffling only matters once the rows arrive in a meaningful order
    srt = ls.sorted_by_target()
    sh = np.median([sgd_epochs(SGD(lr=0.002), srt, z, 8, 30, True, sd)[1][-1] for sd in range(5)])
    fx = np.median([sgd_epochs(SGD(lr=0.002), srt, z, 8, 30, False, sd)[1][-1] for sd in range(5)])
    assert fx > 5 * sh
    # every tuned optimiser actually reaches the tolerance
    for _, make in FAMILIES:
        assert best_lr_loss(make, ls, z)[1] <= 3000
    # LeetCode
    s462 = Solution462()
    assert s462.minMoves2([1, 2, 3]) == 2 and s462.minMoves2([1, 10, 2, 9]) == 16
    assert s462.minMoves2([1, 2, 3, 100]) == 100 and minmoves_mean([1, 2, 3, 100]) == 146
    for nums in ([1, 2, 3], [1, 10, 2, 9], [1, 2, 3, 100], [5]):
        best = min(moves_to(nums, c) for c in range(min(nums), max(nums) + 1))
        assert s462.minMoves2(nums) == best
    c, _ = subgradient_median([1, 2, 3, 100])
    assert 2.0 <= c <= 3.0
    s852 = Solution852()
    for n, pk in ((5, 2), (101, 50), (4001, 1333), (3, 1)):
        arr, p = mountain(n, pk)
        assert s852.peakIndexInMountainArray(arr) == p == peak_linear(arr)
    arr, _ = mountain(4001)
    assert peak_probes(arr) <= 2 * math.ceil(math.log2(len(arr))) + 2
    assert abs(ternary_min(lambda x: (x - 1.234567) ** 2 + 3, -10, 10) - 1.234567) < 1e-6


if __name__ == '__main__':
    main()
