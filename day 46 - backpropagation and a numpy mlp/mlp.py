"""Day 46 - Backpropagation and a multilayer perceptron in pure NumPy, with the
classification losses it needs: sigmoid, softmax and cross-entropy.

Run `python mlp.py` to print the whole story.  Timings depend on the machine; every
other number is exact and checked by an assert at the bottom.
"""

import math

import numpy as np


# ================================================================ 1. softmax, safely
def softmax(z, axis=-1):
    """Subtracting the row max changes nothing mathematically and everything
    numerically: exp() never sees a positive argument, so it cannot overflow."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_naive(z, axis=-1):
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(z, axis=-1):
    """log(softmax(z)) without ever forming softmax(z): log-sum-exp."""
    m = np.max(z, axis=axis, keepdims=True)
    return z - m - np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))


def cross_entropy(logits, y):
    """Mean cross entropy of integer labels y given raw logits."""
    n = len(y)
    return float(-np.mean(log_softmax(logits)[np.arange(n), y]))


def cross_entropy_naive(logits, y):
    """The textbook spelling: form the probabilities, then take their log."""
    n = len(y)
    p = softmax_naive(logits)
    return float(-np.mean(np.log(p[np.arange(n), y])))


# ================================================================ 2. sigmoid, safely
def sigmoid(x):
    """exp() only ever sees a negative argument, on both branches."""
    out = np.empty_like(np.asarray(x, dtype=float))
    pos = np.asarray(x) >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-np.asarray(x)[pos]))
    ex = np.exp(np.asarray(x)[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def sigmoid_naive(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def log_sigmoid(x):
    """log(sigmoid(x)) = -log1p(exp(-x)), written so exp() never overflows."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, -np.log1p(np.exp(-np.abs(x))),
                    x - np.log1p(np.exp(-np.abs(x))))


def log_sigmoid_naive(x):
    return np.log(sigmoid_naive(x))


def bce_with_logits(z, y):
    """Binary cross entropy straight from the logits, never forming log(p)."""
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


# ================================================================ 3. the layers
class Linear:
    """z = x @ W + b.  Backward is three lines, and each one is a transpose away
    from the forward pass."""

    def __init__(self, d_in, d_out, rng, scale=None, zero=False):
        if zero:
            self.W = np.zeros((d_in, d_out))
        else:
            s = scale if scale is not None else math.sqrt(2.0 / d_in)   # He init
            self.W = rng.normal(scale=s, size=(d_in, d_out))
        self.b = np.zeros(d_out)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dz, bug=None):
        # dz already carries the 1/n from the mean loss, so there is no second divide
        if bug == 'no_transpose':
            self.dW = self.x @ dz                # only runs at all when n == d_in
        else:
            self.dW = self.x.T @ dz
        self.db = dz.sum() if bug == 'scalar_bias' else dz.sum(axis=0)
        return dz @ self.W.T

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.dW, self.db]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dz, bug=None):
        return dz * self.mask

    params = grads = ()


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dz, bug=None):
        return dz * (1.0 - self.out ** 2)

    params = grads = ()


class MLP:
    """A list of layers, a forward pass that caches, and a backward pass that walks
    the same list in reverse - the reverse topological order of Day 13, except the
    graph is a straight line so reversing the list is enough."""

    def __init__(self, sizes, rng, act=ReLU, zero=False):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng, zero=zero))
            if i < len(sizes) - 2:
                self.layers.append(act())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dlogits, bug=None):
        for layer in reversed(self.layers):
            dlogits = layer.backward(dlogits, bug)
        return dlogits

    def loss_and_grad(self, x, y, bug=None):
        logits = self.forward(x)
        loss = cross_entropy(logits, y)
        self.backward(softmax_ce_grad(logits, y, mean=(bug != 'no_mean')), bug)
        return loss

    @property
    def params(self):
        return [p for l in self.layers for p in l.params]

    @property
    def grads(self):
        return [g for l in self.layers for g in l.grads]


def softmax_ce_grad(logits, y, mean=True):
    """The whole reason softmax and cross entropy are always written together:
    their composed derivative collapses to (p - y), and dividing by n makes it the
    gradient of the *mean* loss - which is the loss we actually reported."""
    n = len(y)
    p = softmax(logits)
    p[np.arange(n), y] -= 1.0
    return p / n if mean else p


# ================================================================ 4. gradient checking
def numeric_grad(f, a, h=1e-5):
    """Central differences over every entry of the array a, in place."""
    g = np.zeros_like(a)
    it = np.nditer(a, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        old = a[i]
        a[i] = old + h
        up = f()
        a[i] = old - h
        dn = f()
        a[i] = old
        g[i] = (up - dn) / (2 * h)
        it.iternext()
    return g


def gradcheck(net, x, y, bug=None, h=1e-5):
    """Largest relative error between backprop and central differences."""
    net.loss_and_grad(x, y, bug)
    analytic = [g.copy() for g in net.grads]
    worst = 0.0
    for p, a in zip(net.params, analytic):
        num = numeric_grad(lambda: cross_entropy(net.forward(x), y), p, h)
        denom = np.maximum(np.abs(a) + np.abs(num), 1e-12)
        worst = max(worst, float(np.max(np.abs(a - num) / denom)))
    return worst


def kink_demo(h=1e-5):
    """A gradient check that fails even though backprop is right.

    ReLU has no derivative at 0.  Back-propagation picks one side (mask = x > 0, so
    zero); central differences average both sides and get 0.5.  Feed the network a
    row of zeros and the biases start at zero, so the pre-activation is exactly 0 and
    the two disagree.  Nothing is wrong with the code - the function has a corner.
    """
    x = np.zeros((2, 3))
    x[1] = [0.4, -0.2, 0.1]
    y = np.array([0, 1])
    relu_err = gradcheck(MLP([3, 4, 2], np.random.default_rng(3)), x, y, h=h)
    tanh_err = gradcheck(MLP([3, 4, 2], np.random.default_rng(3), act=Tanh), x, y, h=h)
    net = MLP([3, 4, 2], np.random.default_rng(3))
    pre = net.layers[0].forward(x)
    return relu_err, tanh_err, int(np.sum(pre == 0.0)), pre.size


# ================================================================ 5. data and training
def spiral(points=100, classes=3, seed=46, noise=0.22):
    """The classic spiral: three arms that no straight line can separate."""
    rng = np.random.default_rng(seed)
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype=int)
    for c in range(classes):
        idx = range(points * c, points * (c + 1))
        r = np.linspace(0.05, 1.0, points)   # not 0: all three arms would share the origin
        t = np.linspace(c * 4, (c + 1) * 4, points) + rng.normal(scale=noise, size=points)
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = c
    return X, y


class Adam:
    """Day 45's optimiser, the version with bias correction and eps outside the sqrt."""

    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def accuracy(net, X, y):
    return float(np.mean(net.forward(X).argmax(axis=1) == y))


def train(net, X, y, steps=1200, lr=0.05, bug=None, every=200):
    opt = Adam(net.params, lr=lr)
    hist = []
    for t in range(1, steps + 1):
        loss = net.loss_and_grad(X, y, bug)
        opt.step(net.grads)
        if t % every == 0 or t == 1:
            hist.append((t, loss, accuracy(net, X, y)))
    return hist


def hidden_activations(net, X):
    """What the hidden layer sees, to show zero init collapsing every unit."""
    h = net.layers[0].forward(X)
    return net.layers[1].forward(h)


# ================================================================ 6. LeetCode 867
class Solution867:
    """Transpose a matrix: the operation every backward rule is built on."""

    def transpose(self, matrix: list[list[int]]) -> list[list[int]]:
        m, n = len(matrix), len(matrix[0])
        out = [[0] * m for _ in range(n)]
        for i in range(m):
            for j in range(n):
                out[j][i] = matrix[i][j]
        return out


# ================================================================ 7. LeetCode 48
class Solution48:
    """Rotate a square matrix 90 degrees clockwise, in place: transpose, then
    reverse each row."""

    def rotate(self, matrix: list[list[int]]) -> None:
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):            # j > i only, or it swaps back
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for row in matrix:
            row.reverse()


def rotate_full_loop(matrix):
    """The same rotation with j over the whole row: every pair is swapped twice,
    so the matrix comes back unchanged - and nothing raises."""
    n = len(matrix)
    out = [row[:] for row in matrix]
    for i in range(n):
        for j in range(n):
            out[i][j], out[j][i] = out[j][i], out[i][j]
    for row in out:
        row.reverse()
    return out


# ================================================================ the walkthrough
def header(t):
    print('\n' + '=' * 72 + '\n' + t + '\n' + '=' * 72)


def main() -> None:
    np.seterr(all='ignore')
    rng = np.random.default_rng(46)

    header('1. softmax: the max subtraction is not an optimisation')
    big = np.array([[1000.0, 1001.0, 1002.0]])
    print(f'  logits {big[0].tolist()}')
    print(f'    naive  {softmax_naive(big)[0]}   <- exp(1002) overflows')
    print(f'    stable {np.round(softmax(big)[0], 6)}')
    small = np.array([[-1000.0, -1001.0, -1002.0]])
    print(f'  logits {small[0].tolist()}')
    print(f'    naive  {softmax_naive(small)[0]}   <- exp(-1002) underflows, then 0/0')
    print(f'    stable {np.round(softmax(small)[0], 6)}')
    z = rng.normal(size=(4, 3)) * 3
    print(f'  on ordinary logits the two agree to '
          f'{np.max(np.abs(softmax(z) - softmax_naive(z))):.1e}, which is why this')
    print('  never shows up until the day a logit grows')

    header('2. The axis bug that still returns probabilities')
    z = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]])
    right = softmax(z, axis=1)
    wrong = softmax(z, axis=0)
    print(f'  logits for 2 examples x 3 classes:\n{z}')
    print(f'  softmax(axis=1), per example:\n{np.round(right, 4)}')
    print(f'    row sums {np.round(right.sum(axis=1), 6)}  <- what we want')
    print(f'  softmax(axis=0), per class:\n{np.round(wrong, 4)}')
    print(f'    row sums {np.round(wrong.sum(axis=1), 4)}, column sums '
          f'{np.round(wrong.sum(axis=0), 6)}')
    print('  every entry is still in [0, 1] and something still sums to 1, so no')
    print('  assertion fires; the model is simply normalised across the batch')
    print(f'  argmax per row agrees here: {right.argmax(1).tolist()} vs {wrong.argmax(1).tolist()}'
          f' - it is the loss, not the prediction, that goes wrong first')

    header('3. Cross entropy: never take the log of a probability you just formed')
    logits = np.array([[-400.0, 0.0, 400.0]])
    print(f'  logits {logits[0].tolist()} - large, but exp() of them still fits')
    print(f'    naive probabilities {softmax_naive(logits)[0]}')
    print(f'    the first one underflowed to exactly 0 - fine until it is the label')
    print(f'    label 2 (what the model predicts): stable {cross_entropy(logits, np.array([2])):.3e}, '
          f'naive {cross_entropy_naive(logits, np.array([2])):.3e}')
    print(f'    label 0 (the model is wrong):      stable {cross_entropy(logits, np.array([0])):.1f}, '
          f'naive {cross_entropy_naive(logits, np.array([0]))}')
    print('  the naive route forms a probability of exp(-800) = 0, then takes log(0)')
    print('  log_softmax never builds the probability at all: z - m - log(sum(exp(z - m)))')

    header('4. sigmoid and binary cross entropy')
    xs = np.array([-800.0, -40.0, 0.0, 40.0, 800.0])
    print(f'  x            {xs.tolist()}')
    print(f'  sigmoid      {sigmoid(xs)}')
    print(f'  naive        {sigmoid_naive(xs)}')
    print(f'  the naive sigmoid computes 1 / (1 + exp(800)) = 1 / inf and lands on the')
    print(f'  right answer by luck, after an overflow: np.exp(800.0) = {np.exp(800.0)}')
    print(f'  log_sigmoid  {np.round(log_sigmoid(xs), 4)}')
    print(f'  naive log    {log_sigmoid_naive(xs)}   <- log(0) at x = -800, and that')
    print('  is the one that matters: the loss, not the probability, is what gets')
    print('  differentiated')
    print(f'  BCE from logits, z = -800 with label 1: {bce_with_logits(np.array([-800.0]), np.array([1.0])):.1f}')
    print('  a loss of 800 is a finite, usable gradient; an inf is the end of training')

    header('5. The gradient of softmax + cross entropy is p - y')
    logits = rng.normal(size=(5, 4))
    y = rng.integers(0, 4, size=5)
    g = softmax_ce_grad(logits.copy(), y)
    num = numeric_grad(lambda: cross_entropy(logits, y), logits)
    print(f'  analytic (p - y)/n vs central differences: largest difference '
          f'{np.max(np.abs(g - num)):.2e}')
    print('  this is why the two are always implemented as one function: the softmax')
    print('  Jacobian and the log both disappear, leaving a subtraction')

    header('6. A multilayer perceptron, forward and backward')
    X, yy = spiral()
    print(f'  spiral: {len(X)} points, {yy.max() + 1} classes, 2 features')
    net = MLP([2, 64, 3], np.random.default_rng(0))
    print(f'  layers: {" -> ".join(type(l).__name__ for l in net.layers)}')
    print(f'  parameter shapes: {[tuple(p.shape) for p in net.params]}')
    print(f'  gradient check against central differences: worst relative error '
          f'{gradcheck(MLP([2, 8, 3], np.random.default_rng(1)), X[:12], yy[:12]):.2e}')

    header('7. Training it')
    net = MLP([2, 64, 3], np.random.default_rng(0))
    for t, loss, acc in train(net, X, yy):
        print(f'    step {t:>5}  loss {loss:.4f}  accuracy {acc:.3f}')
    lin = MLP([2, 3], np.random.default_rng(0))
    train(lin, X, yy)
    print(f'  one hidden layer of 64 ReLU units: accuracy {accuracy(net, X, yy):.3f}')
    print(f'  no hidden layer (softmax regression): accuracy {accuracy(lin, X, yy):.3f}')
    print('  (both figures are on the training set - this is a demonstration that the')
    print('   model can represent the shape at all, not a claim about generalisation)')
    print('  the hidden layer is what buys a curved boundary; the output layer is')
    print('  still a straight line, drawn in the space the hidden layer invented')

    header('8. Four backprop bugs that never raise, and one failure that is not a bug')
    base = MLP([2, 64, 3], np.random.default_rng(0))
    print(f'  correct backward:            gradcheck {gradcheck(MLP([2, 8, 3], np.random.default_rng(1)), X[:12], yy[:12]):.2e}')
    for bug, label in (('scalar_bias', 'db = dz.sum() not sum(axis=0)'),
                       ('no_mean', 'forgot the division by n')):
        err = gradcheck(MLP([2, 8, 3], np.random.default_rng(1)), X[:12], yy[:12], bug=bug)
        print(f'  {label:<28} gradcheck {err:.2e}')
    n_eq = 8
    netq = MLP([n_eq, 8, 3], np.random.default_rng(1))
    Xq = rng.normal(size=(n_eq, n_eq))
    yq = rng.integers(0, 3, size=n_eq)
    netq.loss_and_grad(Xq, yq, bug='no_transpose')
    print(f'  dW = x @ dz with batch {n_eq} and input width {n_eq}: runs, shapes '
          f'{tuple(netq.layers[0].dW.shape)}, gradcheck '
          f'{gradcheck(MLP([n_eq, 8, 3], np.random.default_rng(1)), Xq, yq, bug="no_transpose"):.2e}')
    print('  that one only runs at all when the batch size equals the input width -')
    print('  change the batch size and it becomes a loud shape error instead')
    zero = MLP([2, 64, 3], np.random.default_rng(0), zero=True)
    h0 = hidden_activations(zero, X[:4])
    train(zero, X, yy, steps=400)
    h1 = hidden_activations(zero, X[:4])
    print(f'  all-zero init: distinct hidden activations before training '
          f'{len(np.unique(np.round(h0, 9)))}, after {len(np.unique(np.round(h1, 9)))}'
          f' (out of {h1.size})')
    print(f'    accuracy after 400 steps {accuracy(zero, X, yy):.3f}, against '
          f'{accuracy(net, X, yy):.3f} for the randomly initialised net above')
    print('    every hidden unit gets the same gradient forever, so 64 units behave')
    print('    as one; symmetry has to be broken by the random initialisation')
    r_err, t_err, zeros, total = kink_demo()
    print(f'  and one failure that is NOT a bug: a gradient check at a ReLU kink')
    print(f'    a row of zeros makes {zeros} of {total} pre-activations exactly 0')
    print(f'    ReLU gradcheck {r_err:.2f}, the same network with tanh {t_err:.1e}')
    print('    backprop picks the x > 0 side and reports 0; central differences average')
    print('    both sides and report 0.5. Check gradients with a smooth activation.')

    header('9. LeetCode 867 - Transpose Matrix')
    m = [[1, 2, 3], [4, 5, 6]]
    print(f'  {m} -> {Solution867().transpose(m)}')
    a = np.array(m)
    print(f'  numpy: a.T is a view, strides {a.strides} -> {a.T.strides}, '
          f'shares memory {np.shares_memory(a, a.T)}')
    print('  the list-of-lists version has to move every element; numpy only swaps')
    print('  the strides, which is why x.T @ dz costs no more than x @ dz')

    header('10. LeetCode 48 - Rotate Image')
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    got = [row[:] for row in m]
    Solution48().rotate(got)
    print(f'  {m}')
    print(f'    rotated (j > i):  {got}')
    print(f'    j over every column: {rotate_full_loop(m)}   <- same as reversing rows only')
    print('  swapping every pair twice puts the matrix back, so the transpose silently')
    print('  does nothing and the answer is the input with its rows reversed')

    check()
    print('\nall assertions passed')


def check() -> None:
    np.seterr(all='ignore')
    rng = np.random.default_rng(46)
    # softmax stability
    big = np.array([[1000.0, 1001.0, 1002.0]])
    assert np.isnan(softmax_naive(big)).all() and np.allclose(softmax(big).sum(), 1.0)
    assert abs(softmax(big)[0, 2] - math.exp(0) / (math.exp(-2) + math.exp(-1) + 1)) < 1e-12
    z = rng.normal(size=(6, 4)) * 3
    assert np.max(np.abs(softmax(z) - softmax_naive(z))) < 1e-12
    # the wrong axis still produces something that sums to one, just not per row
    w = softmax(z, axis=0)
    assert np.allclose(w.sum(axis=0), 1.0) and not np.allclose(w.sum(axis=1), 1.0)
    assert ((w >= 0) & (w <= 1)).all()
    # cross entropy
    lg = np.array([[0.0, 0.0, 800.0]])
    assert cross_entropy(lg, np.array([2])) < 1e-12
    assert math.isinf(cross_entropy_naive(lg, np.array([0])))
    assert abs(cross_entropy(lg, np.array([0])) - 800.0) < 1e-6
    # sigmoid
    xs = np.array([-800.0, 0.0, 800.0])
    assert np.isinf(sigmoid_naive(xs)[0]) or sigmoid_naive(xs)[0] == 0.0
    assert abs(sigmoid(xs)[1] - 0.5) < 1e-15 and sigmoid(xs)[2] == 1.0 and sigmoid(xs)[0] == 0.0
    assert math.isinf(log_sigmoid_naive(np.array([-800.0]))[0])
    assert abs(log_sigmoid(np.array([-800.0]))[0] + 800.0) < 1e-6
    assert abs(bce_with_logits(np.array([-800.0]), np.array([1.0])) - 800.0) < 1e-6
    for x in (-30.0, -1.0, 0.0, 1.0, 30.0):
        assert abs(log_sigmoid(np.array([x]))[0] - math.log(1 / (1 + math.exp(-x)))) < 1e-9
    # (p - y) / n really is the gradient
    logits = rng.normal(size=(5, 4))
    y = rng.integers(0, 4, size=5)
    g = softmax_ce_grad(logits.copy(), y)
    assert np.max(np.abs(g - numeric_grad(lambda: cross_entropy(logits, y), logits))) < 1e-8
    # backprop agrees with central differences, and each bug does not
    X, yy = spiral()
    good = gradcheck(MLP([2, 8, 3], np.random.default_rng(1)), X[:12], yy[:12])
    assert good < 1e-6, good
    assert gradcheck(MLP([2, 8, 3], np.random.default_rng(1), act=Tanh), X[:12], yy[:12]) < 1e-6
    # the three arms must not share the origin, or one point carries three labels
    assert np.min(np.linalg.norm(X[:, None] - X[None, :], axis=-1)
                  + np.eye(len(X)) * 9) > 1e-6
    # a ReLU kink makes a correct gradient look wrong; tanh does not
    r_err, t_err, zeros, _ = kink_demo()
    assert zeros > 0 and r_err > 0.2 and t_err < 1e-6
    for bug in ('scalar_bias', 'no_mean'):
        assert gradcheck(MLP([2, 8, 3], np.random.default_rng(1)), X[:12], yy[:12], bug=bug) > 0.1
    n_eq = 8
    Xq, yq = rng.normal(size=(n_eq, n_eq)), rng.integers(0, 3, size=n_eq)
    net_q = MLP([n_eq, 8, 3], np.random.default_rng(1))
    net_q.loss_and_grad(Xq, yq, bug='no_transpose')          # runs only because n == d_in
    assert net_q.layers[0].dW.shape == (n_eq, 8)
    assert gradcheck(MLP([n_eq, 8, 3], np.random.default_rng(1)), Xq, yq, bug='no_transpose') > 0.1
    try:
        MLP([2, 8, 3], np.random.default_rng(1)).loss_and_grad(X[:12], yy[:12], bug='no_transpose')
        raise AssertionError('expected a shape error once n != d_in')
    except ValueError:
        pass
    # training
    net = MLP([2, 64, 3], np.random.default_rng(0))
    train(net, X, yy)
    assert accuracy(net, X, yy) > 0.95
    lin = MLP([2, 3], np.random.default_rng(0))
    train(lin, X, yy)
    assert accuracy(lin, X, yy) < 0.6
    # symmetry never breaks from an all-zero start
    zero = MLP([2, 64, 3], np.random.default_rng(0), zero=True)
    train(zero, X, yy, steps=400)
    h = hidden_activations(zero, X[:4])
    assert len(np.unique(np.round(h, 9))) <= 4 and accuracy(zero, X, yy) < 0.6
    # LeetCode
    assert Solution867().transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert Solution867().transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    for shape in ((2, 3), (3, 2), (1, 5), (4, 4)):
        a = rng.integers(0, 9, size=shape)
        assert Solution867().transpose(a.tolist()) == a.T.tolist()
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    got = [r[:] for r in m]
    Solution48().rotate(got)
    assert got == [[7, 4, 1], [8, 5, 2], [9, 6, 3]] == np.rot90(np.array(m), -1).tolist()
    assert rotate_full_loop(m) == [r[::-1] for r in m] != got
    for n in (1, 2, 3, 5):
        a = rng.integers(0, 9, size=(n, n))
        g2 = a.tolist()
        Solution48().rotate(g2)
        assert g2 == np.rot90(a, -1).tolist()


if __name__ == '__main__':
    main()
