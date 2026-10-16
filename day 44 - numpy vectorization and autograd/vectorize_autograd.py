"""Day 44 - NumPy vectorisation and the intuition behind automatic differentiation,
with a hand-written mini autograd.

Run `python vectorize_autograd.py` to print the whole story.  Timings depend on the
machine; every other number is exact and checked by an assert at the bottom.
"""

import math
import random
import time

import numpy as np


def best_of(fn, repeat=5):
    best = float('inf')
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


# ---------------------------------------------------------------- 1. loops vs vectorised
def sum_squares_loop(xs):
    s = 0.0
    for x in xs:
        s += x * x
    return s


def sum_squares_numpy(a):
    return float(np.dot(a, a))


def layout(a, base=None):
    """What NumPy actually stores: one buffer, a shape and the byte step per axis."""
    return {'shape': a.shape, 'strides': a.strides,
            'shares memory': bool(base is not None and np.shares_memory(a, base)),
            'C-contiguous': bool(a.flags['C_CONTIGUOUS'])}


def broadcast_shape(s1, s2):
    """NumPy's rule, written out: align from the right; each pair must be equal or 1."""
    out = []
    for i in range(1, max(len(s1), len(s2)) + 1):
        a = s1[-i] if i <= len(s1) else 1
        b = s2[-i] if i <= len(s2) else 1
        if a != b and 1 not in (a, b):
            raise ValueError(f'shapes {s1} and {s2} do not broadcast')
        out.append(max(a, b))
    return tuple(reversed(out))


def mse(pred, y):
    return float(((pred - y) ** 2).mean())


def pairwise_loop(x):
    n = len(x)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = math.sqrt(sum((x[i][k] - x[j][k]) ** 2 for k in range(len(x[i]))))
    return out


def pairwise_broadcast(x):
    diff = x[:, None, :] - x[None, :, :]          # (n, n, d) - every pair, materialised
    return np.sqrt((diff ** 2).sum(axis=-1))


def pairwise_gram(x, clip=True):
    """|a - b|^2 = |a|^2 + |b|^2 - 2 a.b : one matrix product, no (n, n, d) array."""
    sq = (x * x).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    if clip:
        d2 = np.maximum(d2, 0.0)                  # rounding can push a 0 to -2e-16
    with np.errstate(invalid='ignore'):
        return np.sqrt(d2)


# ---------------------------------------------------------------- 2. derivatives by hand
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h


class Dual:
    """Forward mode: carry (value, derivative) through every operation."""

    def __init__(self, val, der=0.0):
        self.val, self.der = val, der

    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val + o.val, self.der + o.der)
    __radd__ = __add__

    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val * o.val, self.der * o.val + self.val * o.der)
    __rmul__ = __mul__

    def sin(self):
        return Dual(math.sin(self.val), math.cos(self.val) * self.der)


# ---------------------------------------------------------------- 3. a mini autograd
class Value:
    """A scalar that remembers how it was made, so it can send gradients back."""

    accumulate = True          # the right rule: a node used twice gets both contributions

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def _wrap(self, o):
        return o if isinstance(o, Value) else type(self)(o)

    def _send(self, child, g):
        child.grad = child.grad + g if self.accumulate else g

    def __add__(self, o):
        o = self._wrap(o)
        out = type(self)(self.data + o.data, (self, o), '+')

        def _backward():
            self._send(self, out.grad)
            self._send(o, out.grad)
        out._backward = _backward
        return out

    def __mul__(self, o):
        o = self._wrap(o)
        out = type(self)(self.data * o.data, (self, o), '*')

        def _backward():
            self._send(self, o.data * out.grad)
            self._send(o, self.data * out.grad)
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = type(self)(self.data ** k, (self,), f'**{k}')

        def _backward():
            self._send(self, k * self.data ** (k - 1) * out.grad)
        out._backward = _backward
        return out

    def exp(self):
        out = type(self)(math.exp(self.data), (self,), 'exp')

        def _backward():
            self._send(self, out.data * out.grad)
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = type(self)(t, (self,), 'tanh')

        def _backward():
            self._send(self, (1 - t * t) * out.grad)
        out._backward = _backward
        return out

    def relu(self):
        out = type(self)(max(0.0, self.data), (self,), 'relu')

        def _backward():
            self._send(self, (out.data > 0) * out.grad)
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, o):
        return self + (-self._wrap(o))

    def __truediv__(self, o):
        return self * self._wrap(o) ** -1

    def __radd__(self, o):
        return self + o

    def __rmul__(self, o):
        return self * o

    def __rsub__(self, o):
        return self._wrap(o) - self

    def __rtruediv__(self, o):
        return self._wrap(o) / self

    def __repr__(self):
        return f'Value({self.data:.4f}, grad={self.grad:.4f})'

    # -- the backward pass: reverse topological order (Day 13), every node once
    def topo(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:                       # iterative DFS: no recursion limit
            v, done = stack.pop()
            if done:
                order.append(v)
                continue
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.append((v, True))
            for c in v._prev:
                if id(c) not in seen:
                    stack.append((c, False))
        return order

    def backward(self):
        order = self.topo()
        for v in order:
            v.grad = 0.0
        self.grad = 1.0
        for v in reversed(order):
            v._backward()
        return order

    def backward_naive(self):
        """The tempting version: push my gradient to my children, then recurse into
        each child at once - before the child has heard from all of its parents."""
        self.grad = 1.0
        calls = [0]

        def go(v):
            calls[0] += 1
            v._backward()
            for c in v._prev:
                go(c)
        go(self)
        return calls[0]


class AssignValue(Value):
    """The same Value with `grad = g` instead of `grad += g`."""
    accumulate = False


def numeric_grads(f, xs, h=1e-6):
    out = []
    for i in range(len(xs)):
        up = list(xs)
        dn = list(xs)
        up[i] += h
        dn[i] -= h
        out.append((f(*up) - f(*dn)) / (2 * h))
    return out


def gradcheck(f, xs, h=1e-6):
    """Largest relative difference between autograd and central differences."""
    vs = [Value(x) for x in xs]
    f(*vs).backward()
    num = numeric_grads(lambda *a: f(*[Value(t) for t in a]).data, xs, h)
    worst = 0.0
    for v, n in zip(vs, num):
        worst = max(worst, abs(v.grad - n) / max(1e-12, abs(v.grad) + abs(n)))
    return worst


# ---------------------------------------------------------------- 4. training: zero_grad
def make_line_data(n=20, seed=44):
    rng = random.Random(seed)
    xs = [rng.uniform(-1, 1) for _ in range(n)]
    ys = [3.0 * x - 0.5 + rng.gauss(0, 0.1) for x in xs]
    return xs, ys


def fit_line(xs, ys, steps=60, lr=0.1, zero_grad=True):
    """Gradient descent on w, b with Value; returns (w, b, loss history)."""
    w, b = Value(0.0, label='w'), Value(0.0, label='b')
    hist = []
    for _ in range(steps):
        loss = sum(((w * x + b) - y) ** 2 for x, y in zip(xs, ys)) * (1 / len(xs))
        if zero_grad:
            w.grad = b.grad = 0.0          # forget the previous step's gradient
        order = loss.topo()
        for v in order:
            if v is not w and v is not b:
                v.grad = 0.0
        loss.grad = 1.0
        for v in reversed(order):
            v._backward()
        w.data -= lr * w.grad
        b.data -= lr * b.grad
        hist.append(loss.data)
    return w.data, b.data, hist


# ---------------------------------------------------------------- 5. scalar graph vs matrix formula
def linreg_grad_numpy(X, y, w):
    r = X @ w - y
    return 2.0 / len(y) * (X.T @ r)


def linreg_grad_values(X, y, w):
    ws = [Value(v) for v in w]
    loss = Value(0.0)
    for i in range(len(y)):
        pred = sum((ws[j] * float(X[i, j]) for j in range(len(w))), Value(0.0))
        loss = loss + (pred - float(y[i])) ** 2
    loss = loss * (1.0 / len(y))
    order = loss.backward()
    return np.array([v.grad for v in ws]), len(order)


# ---------------------------------------------------------------- 6. LeetCode 150
class Solution150:
    """Evaluate Reverse Polish Notation; division truncates toward zero."""

    def evalRPN(self, tokens):
        st = []
        for t in tokens:
            if t in ('+', '-', '*', '/'):
                b, a = st.pop(), st.pop()
                if t == '+':
                    st.append(a + b)
                elif t == '-':
                    st.append(a - b)
                elif t == '*':
                    st.append(a * b)
                else:
                    st.append(int(a / b))      # truncate, NOT a // b (which floors)
            else:
                st.append(int(t))
        return st[0]


def eval_rpn_floor(tokens):
    st = []
    for t in tokens:
        if t in ('+', '-', '*', '/'):
            b, a = st.pop(), st.pop()
            st.append({'+': a + b, '-': a - b, '*': a * b, '/': a // b}[t])
        else:
            st.append(int(t))
    return st[0]


def rpn_graph(tokens, env):
    """The same stack machine over Value objects: evaluating builds the graph."""
    st = []
    for t in tokens:
        if t in ('+', '-', '*', '/'):
            b, a = st.pop(), st.pop()
            st.append({'+': lambda: a + b, '-': lambda: a - b,
                       '*': lambda: a * b, '/': lambda: a / b}[t]())
        else:
            st.append(env[t] if t in env else Value(float(t)))
    return st[0]


# ---------------------------------------------------------------- 7. LeetCode 566
class Solution566:
    """Reshape the Matrix: read row by row, write row by row."""

    def matrixReshape(self, mat, r, c):
        m, n = len(mat), len(mat[0])
        if m * n != r * c:
            return mat
        out = [[0] * c for _ in range(r)]
        for k in range(m * n):
            out[k // c][k % c] = mat[k // n][k % n]
        return out


# ---------------------------------------------------------------- the walkthrough
def header(t):
    print('\n' + '=' * 72 + '\n' + t + '\n' + '=' * 72)


MICRO = dict(a=2.0, b=-3.0, c=10.0, f=-2.0)
RPN_EXAMPLE = ['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']


def micrograd_example():
    a, b, c, f = (Value(MICRO[k], label=k) for k in 'abcf')
    e = a * b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    L = d * f
    L.label = 'L'
    order = L.backward()
    return {v.label: (v.data, v.grad) for v in order}, [v.label for v in order]


def linreg_data(n=200, d=5, seed=44):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0, 0.0]) + 0.1 * rng.normal(size=n)
    w = rng.normal(size=d)
    return X, y, w


def main() -> None:
    header('1. One loop in Python, one loop in C')
    rng = np.random.default_rng(44)
    a = rng.random(1_000_000)
    xs = a.tolist()
    t_loop = best_of(lambda: sum_squares_loop(xs), 3)
    t_gen = best_of(lambda: sum(x * x for x in xs), 3)
    t_mul = best_of(lambda: (a * a).sum())
    t_dot = best_of(lambda: sum_squares_numpy(a))
    print(f'  sum of squares of 1,000,000 floats (best of several runs):')
    print(f'    for loop          {t_loop * 1000:8.1f} ms')
    print(f'    sum(generator)    {t_gen * 1000:8.1f} ms')
    print(f'    (a * a).sum()     {t_mul * 1000:8.1f} ms   {t_loop / t_mul:5.0f}x faster than the loop')
    print(f'    np.dot(a, a)      {t_dot * 1000:8.1f} ms   {t_loop / t_dot:5.0f}x')
    print(f'    same answer: {sum_squares_loop(xs):.6f} vs {sum_squares_numpy(a):.6f}')
    print(f'  a Python float list: {len(xs):,} pointers to boxed floats; '
          f'the array: one {a.nbytes:,}-byte buffer of raw doubles')

    header('2. Shape and strides: views share memory')
    m = np.arange(12).reshape(3, 4)
    for name, v in (('m', m), ('m.T', m.T), ('m[:, ::2]', m[:, ::2]),
                    ('m.T.reshape(12)', m.T.reshape(12))):
        print(f'  {name:<16} {layout(v, m)}')

    header('3. Broadcasting')
    for s1, s2 in (((3,), (3, 1)), ((5, 1, 4), (3, 1)), ((200, 5), (5,))):
        print(f'  {s1} with {s2} -> {broadcast_shape(s1, s2)}  (numpy: {np.broadcast_shapes(s1, s2)})')
    try:
        broadcast_shape((3,), (4,))
    except ValueError as e:
        print(f'  {e}')
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred_col = (y + 0.1).reshape(-1, 1)           # a model that returns shape (n, 1)
    print(f'  y {y.shape}, pred {pred_col.shape}: pred - y has shape {(pred_col - y).shape}')
    print(f'  MSE with pred.ravel(): {mse(pred_col.ravel(), y):.4f}   '
          f'MSE with pred as (5, 1): {mse(pred_col, y):.4f}  (no error, no warning)')

    header('4. Silent failures: integer wrap-around and views')
    img1 = np.array([200, 100, 250], dtype=np.uint8)
    img2 = np.array([100, 100, 10], dtype=np.uint8)
    print(f'  uint8 {img1.tolist()} + {img2.tolist()} = {(img1 + img2).tolist()}')
    print(f'  average (a + b) // 2 = {((img1 + img2) // 2).tolist()}; '
          f'in int16: {((img1.astype(np.int16) + img2) // 2).tolist()}')
    v = np.arange(6)
    s = v[::2]
    s[:] = 0
    print(f'  s = v[::2]; s[:] = 0       -> v = {v.tolist()}   (a slice is a view)')
    v = np.arange(6)
    v[[0, 2]][0] = 99
    print(f'  v[[0, 2]][0] = 99         -> v = {v.tolist()}   (a fancy index is a copy: no effect)')
    v[[0, 2]] = 99
    print(f'  v[[0, 2]] = 99            -> v = {v.tolist()}')

    header('5. Pairwise distances: loops, broadcasting, one matrix product')
    pts = np.random.default_rng(7).random((300, 3))
    t1 = best_of(lambda: pairwise_loop(pts.tolist()), 1)
    t2 = best_of(lambda: pairwise_broadcast(pts))
    t3 = best_of(lambda: pairwise_gram(pts))
    diff = np.abs(pairwise_broadcast(pts) - np.array(pairwise_loop(pts.tolist()))).max()
    print(f'  300 points in 3-d: loops {t1 * 1000:.0f} ms, broadcasting {t2 * 1000:.2f} ms, '
          f'Gram trick {t3 * 1000:.2f} ms; max difference {diff:.1e}')
    for n in (300, 20_000):
        print(f'  broadcasting builds an ({n:,}, {n:,}, 3) array: {n * n * 3 * 8 / 1e6:,.1f} MB')
    many = np.random.default_rng(44).random((500, 3))
    raw = (lambda x: ((x * x).sum(1)[:, None] + (x * x).sum(1)[None, :] - 2 * x @ x.T))(many)
    with np.errstate(invalid='ignore'):
        nans = int(np.isnan(np.sqrt(raw)).sum())
    print(f'  Gram trick on 500 points without clipping: {int((raw < 0).sum())} squared distances '
          f'below zero (smallest {raw.min():.1e}) -> {nans} NaNs after sqrt')

    header('6. Derivatives three ways')
    print('  d/dx sin(x) at x = 1 (true value cos 1): error of the difference quotient')
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12):
        h = 10.0 ** -k
        print(f'    h = 1e-{k:<2}  forward {abs(forward_diff(math.sin, 1.0, h) - math.cos(1)):.1e}'
              f'   central {abs(central_diff(math.sin, 1.0, h) - math.cos(1)):.1e}')
    x = Dual(2.0, 1.0)
    fx = x * x.sin() + 3 * x
    print(f'  dual numbers: f(x) = x sin x + 3x at 2 -> f = {fx.val:.6f}, f\' = {fx.der:.12f} '
          f'(exact {math.sin(2) + 2 * math.cos(2) + 3:.12f})')
    n = 200
    vals = [random.Random(1).uniform(-1, 1) for _ in range(n)]
    g = lambda zs: sum((zs[i] * zs[i + 1] for i in range(n - 1)), 0.0)
    passes = 0
    grads_fwd = []
    for i in range(n):
        zs = [Dual(v, 1.0 if j == i else 0.0) for j, v in enumerate(vals)]
        grads_fwd.append(g(zs).der)
        passes += 1
    vs = [Value(v) for v in vals]
    g(vs).backward()
    same = max(abs(p - q.grad) for p, q in zip(grads_fwd, vs))
    print(f'  gradient of a function of {n} inputs: forward mode {passes} passes, reverse mode '
          f'1 forward + 1 backward; largest difference {same:.1e}')

    header('7. A mini autograd')
    res, order = micrograd_example()
    print('  L = (a*b + c) * f with a=2, b=-3, c=10, f=-2')
    print('  topological order:', ' '.join(order))
    for k in ('L', 'd', 'f', 'e', 'c', 'a', 'b'):
        print(f'    {k}: value {res[k][0]:6.1f}   dL/d{k} = {res[k][1]:5.1f}')
    fx3 = lambda p, q, r: (p * q + r).tanh() * p + (q / r).exp() - p ** 2
    print(f'  gradient check against central differences: worst relative error '
          f'{gradcheck(fx3, [0.5, -1.2, 2.0]):.1e}')
    x = Value(3.0)
    (x * x).backward()
    xa = AssignValue(3.0)
    (xa * xa).backward()
    print(f'  d(x*x)/dx at 3: grad += gives {x.grad}, grad = gives {xa.grad}')
    x = Value(3.0)
    q = x * 2
    nodes = len((q * q).backward())
    good = x.grad
    x = Value(3.0)
    q = x * 2
    calls = (q * q).backward_naive()
    print(f'  d((2x)^2)/dx at 3: topological order {good}, naive recursion {x.grad} '
          f'({calls} node visits instead of {nodes})')
    xs_, ys_ = make_line_data()
    w1, b1, h1 = fit_line(xs_, ys_)
    w2, b2, h2 = fit_line(xs_, ys_, zero_grad=False)
    print(f'  fitting y = 3x - 0.5 on 20 points, 60 steps:')
    print(f'    zeroing gradients:     w = {w1:.3f}, b = {b1:.3f}, final loss {h1[-1]:.4f}')
    print(f'    never zeroing them:    w = {w2:.3f}, b = {b2:.3f}, final loss {h2[-1]:.4f}'
          f'  (loss every 10 steps {[round(v, 3) for v in h2[::10]]})')

    header('8. The same gradient: a scalar graph vs one matrix formula')
    X, y, w = linreg_data()
    gv, nodes = linreg_grad_values(X, y, w)
    gn = linreg_grad_numpy(X, y, w)
    tv = best_of(lambda: linreg_grad_values(X, y, w), 3)
    tn = best_of(lambda: linreg_grad_numpy(X, y, w))
    print(f'  mean squared error, 200 rows x 5 features: Value graph {nodes:,} nodes')
    print(f'  largest difference between the two gradients {np.abs(gv - gn).max():.1e}')
    print(f'  Value graph {tv * 1000:.1f} ms, 2/n X^T (Xw - y) {tn * 1000:.3f} ms '
          f'({tv / tn:,.0f}x)')

    header('9. LeetCode 150 - Evaluate Reverse Polish Notation')
    sol = Solution150()
    for toks in (['2', '1', '+', '3', '*'], ['4', '13', '5', '/', '+'], RPN_EXAMPLE):
        print(f'  {" ".join(toks):<40} -> {sol.evalRPN(toks)}   with // : {eval_rpn_floor(toks)}')
    x, yv = Value(2.0), Value(-3.0)
    r = rpn_graph(['x', 'y', '*', 'x', '+', 'y', '/'], {'x': x, 'y': yv})
    r.backward()
    print(f'  same stack over Values: x y * x + y /  at x=2, y=-3 -> {r.data:.4f}, '
          f'd/dx {x.grad:.4f}, d/dy {yv.grad:.4f}')

    header('10. LeetCode 566 - Reshape the Matrix')
    sol = Solution566()
    for mat, r_, c_ in (([[1, 2], [3, 4]], 1, 4), ([[1, 2], [3, 4]], 2, 4),
                        ([[1, 2, 3], [4, 5, 6]], 3, 2)):
        print(f'  {mat} -> {r_} x {c_}: {sol.matrixReshape(mat, r_, c_)}')
    mm = np.arange(1, 7).reshape(2, 3)
    print(f'  numpy reshape(3, 2): {mm.reshape(3, 2).tolist()}   order="F": '
          f'{mm.reshape(3, 2, order="F").tolist()}')

    check()
    print('\nall assertions passed')


def check() -> None:
    assert broadcast_shape((3,), (3, 1)) == (3, 3)
    assert broadcast_shape((5, 1, 4), (3, 1)) == (5, 3, 4)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(mse((y + 0.1).reshape(-1, 1).ravel(), y) - 0.01) < 1e-12
    assert mse((y + 0.1).reshape(-1, 1), y) > 1
    assert ((np.array([200], np.uint8) + np.array([100], np.uint8)) // 2).tolist() == [22]
    m = np.arange(12).reshape(3, 4)
    assert np.shares_memory(m.T, m) and not np.shares_memory(m.T.reshape(12), m)
    pts = np.random.default_rng(7).random((40, 3))
    assert np.allclose(pairwise_broadcast(pts), pairwise_gram(pts), atol=1e-7)
    assert np.allclose(pairwise_broadcast(pts), np.array(pairwise_loop(pts.tolist())))
    res, order = micrograd_example()
    assert res['L'] == (-8.0, 1.0) and res['a'][1] == 6.0 and res['b'][1] == -4.0
    assert res['c'][1] == -2.0 and res['f'][1] == 4.0 and order[-1] == 'L'
    fx3 = lambda p, q, r: (p * q + r).tanh() * p + (q / r).exp() - p ** 2
    assert gradcheck(fx3, [0.5, -1.2, 2.0]) < 1e-7
    x = Value(3.0)
    (x * x).backward()
    xa = AssignValue(3.0)
    (xa * xa).backward()
    assert x.grad == 6.0 and xa.grad == 3.0
    x = Value(3.0)
    q = x * 2
    (q * q).backward()
    assert x.grad == 24.0
    x = Value(3.0)
    q = x * 2
    (q * q).backward_naive()
    assert x.grad == 48.0
    xs, ys = make_line_data()
    w1, b1, h1 = fit_line(xs, ys)
    w2, b2, h2 = fit_line(xs, ys, zero_grad=False)
    assert abs(w1 - 3) < 0.1 and abs(b1 + 0.5) < 0.1 and h2[-1] > 100 * h1[-1]
    X, y2, w = linreg_data()
    gv, _ = linreg_grad_values(X, y2, w)
    assert np.allclose(gv, linreg_grad_numpy(X, y2, w), atol=1e-10)
    x = Dual(2.0, 1.0)
    assert abs((x * x.sin() + 3 * x).der - (math.sin(2) + 2 * math.cos(2) + 3)) < 1e-12
    sol = Solution150()
    assert sol.evalRPN(RPN_EXAMPLE) == 22 and eval_rpn_floor(RPN_EXAMPLE) == 12
    assert sol.evalRPN(['2', '1', '+', '3', '*']) == 9 and sol.evalRPN(['4', '13', '5', '/', '+']) == 6
    s566 = Solution566()
    for mat, r, c in (([[1, 2], [3, 4]], 1, 4), ([[1, 2, 3], [4, 5, 6]], 3, 2), ([[1, 2], [3, 4]], 2, 4)):
        got = s566.matrixReshape(mat, r, c)
        arr = np.array(mat)
        assert got == (arr.reshape(r, c).tolist() if arr.size == r * c else mat)


if __name__ == '__main__':
    main()
