# NumPy vectorisation and automatic differentiation: making array code fast, and computing gradients you can trust

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that shows one buffer seen through
three sets of strides, broadcasts a (5, 1) prediction against a (5,) target, steps the
difference quotient towards rounding noise, runs a forward and a backward pass through a
small computation graph, replays the `grad =` and naive-recursion bugs, and evaluates
LeetCode 150 and LeetCode 566 step by step.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2044%20-%20numpy%20vectorization%20and%20autograd/imgs/day44_1.png?raw=true)

Summing the squares of a million floats took 41 ms in a Python `for` loop on my machine and 0.1 ms with `np.dot` - about three hundred times faster, same answer to the last printed digit. Later today a gradient computed by a hand-written autograd took about 16 ms; the same gradient written as one matrix formula took about 0.005 ms (single runs on my machine). Both halves of this day are the same lesson: move the loop out of the Python interpreter and into one call that works on a whole array.

This is the first day of the machine-learning part of the series, and the silent-failure theme of the last few days carries straight over. None of these raise an error:

- Predictions of shape `(5, 1)` against targets of shape `(5,)` give a mean squared error of 4.01 instead of 0.01, because NumPy quietly builds a 5 x 5 table of every prediction against every target.
- Averaging two `uint8` images pixel by pixel turns 200 and 100 into 22, not 150.
- `v[[0, 2]][0] = 99` changes nothing at all.
- An autograd that writes `grad = g` instead of `grad += g` says the derivative of `x * x` at 3 is 3, not 6.
- An autograd that walks the graph without a topological order says the derivative of `(2x)²` at 3 is 48, not 24.
- A training loop that forgets to reset gradients ends at a loss of 3.09 instead of 0.0083.

Yesterday (Day 43) was about splitting work across threads and processes. Vectorisation is the other kind of parallelism: one call, one tight loop in C, often over SIMD instructions that handle several numbers at a time - and NumPy releases the GIL inside its numeric loops.

Run `python vectorize_autograd.py` to print every number on this page (timings will differ on your machine).

## Why the loop is slow and the array is fast

A Python list of floats is a list of pointers. Each pointer leads to a separate float object with its own type tag and reference count, and `s += x * x` makes the interpreter look up the type of `x`, find its multiplication, create a new float object for the result, and do the same for the addition - a million times. A NumPy array is one buffer of raw 8-byte doubles plus one type for all of them, so `np.dot(a, a)` is a single call that runs a C loop over contiguous memory.

```text
for loop              41.4 ms
sum(generator)        61.9 ms
(a * a).sum()          1.2 ms      36x faster than the loop
np.dot(a, a)           0.1 ms     284x
```

(One run; the exact ratios move around, the order of magnitude does not.) `(a * a).sum()` is slower than `np.dot` because it first writes a whole second array of a million squares and then reads it back; `np.dot` multiplies and adds in one pass.

## Shape and strides: what an array really is

An array is a buffer, a shape and *strides*: how many bytes to jump to take one step along each axis. `np.arange(12).reshape(3, 4)` is twelve integers with shape `(3, 4)` and strides `(32, 8)` - one row is 4 x 8 bytes, one column step is 8 bytes.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2044%20-%20numpy%20vectorization%20and%20autograd/imgs/day44_2.png?raw=true)

```text
m                {'shape': (3, 4), 'strides': (32, 8), 'shares memory': True, 'C-contiguous': True}
m.T              {'shape': (4, 3), 'strides': (8, 32), 'shares memory': True, 'C-contiguous': False}
m[:, ::2]        {'shape': (3, 2), 'strides': (32, 16), 'shares memory': True, 'C-contiguous': False}
m.T.reshape(12)  {'shape': (12,), 'strides': (8,), 'shares memory': False, 'C-contiguous': True}
```

Transposing swaps the strides; taking every other column doubles one of them. No number moves, so both are free, and both are *views*: writing into them writes into `m`. The last line is the exception: reading `m.T` row by row means jumping around the buffer in a pattern no single pair of strides can describe, so `reshape` has to copy.

That leads to the first silent failure. `s = v[::2]; s[:] = 0` zeroes every other element of `v`, because a slice is a view. But `v[[0, 2]]` - indexing with a list - always copies, so `v[[0, 2]][0] = 99` writes 99 into a temporary copy and throws it away. `v` is unchanged and nothing complains. (`v[[0, 2]] = 99`, a single assignment, does write into `v`.)

### LeetCode 566 - Reshape the Matrix

**The task.** Given a matrix and a new number of rows and columns, rearrange its numbers into the new shape, reading the old matrix row by row and filling the new one row by row. If the new shape cannot hold exactly the same numbers, return the original matrix unchanged.

**Input / output.** `mat`, an `m x n` list of lists, and two integers `r` and `c`. Return an `r x c` list of lists, or `mat` itself when `m * n != r * c`.

**Example.** `mat = [[1, 2], [3, 4]]` with `r = 1, c = 4` reads 1, 2, 3, 4 and writes them into one row: `[[1, 2, 3, 4]]`. With `r = 2, c = 4` there would be 8 places for 4 numbers, so the answer is the original `[[1, 2], [3, 4]]`.

**Constraints.** `m` and `n` are between 1 and 100, the values between -1000 and 1000, and `r`, `c` between 1 and 300. Small enough that any linear solution passes; the only thing to get right is the order, and the size check.

Element number `k` in reading order sits at row `k // n`, column `k % n` of the old matrix and at row `k // c`, column `k % c` of the new one - which is exactly the arithmetic strides do:

```python
class Solution:
    def matrixReshape(self, mat: list[list[int]], r: int, c: int) -> list[list[int]]:
        m, n = len(mat), len(mat[0])
        if m * n != r * c:
            return mat
        out = [[0] * c for _ in range(r)]
        for k in range(m * n):
            out[k // c][k % c] = mat[k // n][k % n]
        return out
```

In NumPy this is `np.array(mat).reshape(r, c)`, which is a view when the input is contiguous. The trap is the order: `reshape(3, 2, order='F')` reads column by column and turns `[[1, 2, 3], [4, 5, 6]]` into `[[1, 5], [4, 3], [2, 6]]` instead of `[[1, 2], [3, 4], [5, 6]]` - a valid-looking matrix, just the wrong one.

[leetcode.com/problems/reshape-the-matrix](https://leetcode.com/problems/reshape-the-matrix/)

## Broadcasting, and the bug it hides

Broadcasting lets arrays of different shapes meet in one operation. The rule: line the shapes up from the right; each pair of sizes must be equal or contain a 1, and a 1 is stretched to match.

```python
def broadcast_shape(s1, s2):
    out = []
    for i in range(1, max(len(s1), len(s2)) + 1):
        a = s1[-i] if i <= len(s1) else 1
        b = s2[-i] if i <= len(s2) else 1
        if a != b and 1 not in (a, b):
            raise ValueError(f'shapes {s1} and {s2} do not broadcast')
        out.append(max(a, b))
    return tuple(reversed(out))
```

`(200, 5)` with `(5,)` is `(200, 5)`: add a bias to every row. `(3,)` with `(3, 1)` is `(3, 3)`. That second one is where the bug lives. Many models return predictions as a column, shape `(n, 1)`, while targets come as a flat `(n,)`:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2044%20-%20numpy%20vectorization%20and%20autograd/imgs/day44_3.png?raw=true)

```text
y (5,), pred (5, 1): pred - y has shape (5, 5)
MSE with pred.ravel(): 0.0100   MSE with pred as (5, 1): 4.0100  (no error, no warning)
```

Every prediction is compared with every target, and `.mean()` happily averages all 25 cells. Training on that loss pushes every prediction towards the mean of all targets. The fix is one `.ravel()` (or `reshape(-1)`), and the habit is an `assert pred.shape == y.shape` before every loss.

The other silent failure in this section is integer wrap-around. A `uint8` holds 0 to 255, and NumPy does not widen it for you: `200 + 100` is 44, so averaging two images as `(a + b) // 2` gives 22 where it should give 150. Convert first: `(a.astype(np.int16) + b) // 2` gives `[150, 100, 130]`.

## Vectorising a double loop: pairwise distances

The distance between every pair of 300 points in 3-d took about 205 ms with Python loops (one run, like every timing in this section). Broadcasting writes the same thing in one line, `x[:, None, :] - x[None, :, :]`, and took about 4.7 ms. But it builds an `(n, n, 3)` array of every difference: 2.2 MB for 300 points, and 9,600 MB for 20,000. Vectorising traded time for memory.

The identity `|a - b|² = |a|² + |b|² - 2 a·b` avoids the big array: one matrix product `x @ x.T` plus two vectors, about 0.63 ms. It has its own trap. The distance from a point to itself should be 0, but `|a|² + |a|² - 2 a·a` is computed in floating point and can come out slightly negative - as low as -8.9e-16 in this run. On 500 random points, 63 squared distances came out below zero, and `np.sqrt` turned all 63 into NaN with only a runtime warning. Clip at zero before the square root - every production implementation does.

## Derivatives, three ways

Training a model means following the slope of a loss downhill, so we need derivatives. The first way needs no theory: nudge the input and watch the output.

```python
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)
```

For `sin` at 1, whose derivative is `cos 1`:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2044%20-%20numpy%20vectorization%20and%20autograd/imgs/day44_4.png?raw=true)

```text
h = 1e-2   forward 4.2e-03   central 9.0e-06
h = 1e-5   forward 4.2e-06   central 1.1e-11
h = 1e-8   forward 3.0e-09   central 2.6e-09
h = 1e-12  forward 4.3e-05   central 1.2e-05
```

Making `h` smaller helps until about `1e-5`, then hurts: `f(x+h)` and `f(x-h)` agree in nearly every digit, and subtracting them leaves mostly rounding noise. Difference quotients are perfect for *checking* gradients and useless for training a model with a million weights, because you need two evaluations of the whole function per weight.

The second way, **forward mode**, carries the derivative along with the value. A dual number is a pair `(value, derivative)` with the product rule built into multiplication:

```python
class Dual:
    def __init__(self, val, der=0.0):
        self.val, self.der = val, der

    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val * o.val, self.der * o.val + self.val * o.der)
```

`f(x) = x sin x + 3x` at 2 gives `f' = 3.077003753731`, exact to every digit. But one pass gives the derivative with respect to *one* input. For a function of 200 inputs that took 200 passes.

The third way, **reverse mode**, is what PyTorch and JAX use for training: one forward pass that records the computation, and one backward pass that produces the derivative with respect to *every* input at once. For the same 200-input function: 1 forward and 1 backward pass, identical numbers.

## A mini autograd

Every `Value` remembers which values it was made from and keeps a small function that sends its gradient back to them using the local rule - for `out = a * b`, the gradient reaching `a` is `b * out.grad`:

```python
def __mul__(self, o):
    o = self._wrap(o)
    out = Value(self.data * o.data, (self, o), '*')

    def _backward():
        self.grad += o.data * out.grad
        o.grad += self.data * out.grad
    out._backward = _backward
    return out
```

`backward()` puts the graph in topological order - the Day 13 algorithm, here as an iterative depth-first search - and runs the `_backward` functions from the output back to the inputs:

```python
def backward(self):
    order = self.topo()            # every node after all of its inputs
    for v in order:
        v.grad = 0.0
    self.grad = 1.0
    for v in reversed(order):      # every node after all of its users
        v._backward()
    return order
```

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2044%20-%20numpy%20vectorization%20and%20autograd/imgs/day44_5.png?raw=true)

For `L = (a*b + c) * f` with `a=2, b=-3, c=10, f=-2`: `L = -8`, and the gradients are `dL/da = 6`, `dL/db = -4`, `dL/dc = -2`, `dL/df = 4`. A gradient check against central differences on a messier function, `tanh(p*q + r) * p + exp(q/r) - p²`, agreed to a relative error of 1.3e-10.

Two lines in that code carry the correctness, and getting either one wrong still returns a number:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2044%20-%20numpy%20vectorization%20and%20autograd/imgs/day44_6.png?raw=true)

**`+=`, not `=`.** In `x * x`, both inputs are the same node, so `_backward` sends two contributions of 3 to `x`. Accumulating gives 6, the right derivative of `x²`. Assigning keeps only the second write: 3.

**The topological order.** The tempting backward pass is "run my `_backward`, then recurse into each input". For `q = 2x; y = q * q`, the recursion reaches `q` twice, and each visit pushes `q`'s full gradient into `x` again: 48 instead of 24, using 7 visits where the topological order needs 4. Running every node only after all of its users have reported is exactly what the order guarantees.

The third bug lives in the training loop, not the autograd. Gradients accumulate by design, so they must be reset between steps. Fitting `y = 3x - 0.5` on 20 points for 60 steps:

```text
zeroing gradients:     w = 3.020, b = -0.534, final loss 0.0083
never zeroing them:    w = 6.089, b = -1.029, final loss 3.0924
```

Without the reset every step uses the sum of all previous gradients, the weights overshoot, and the loss bounces around (4.40, 2.18, 2.59, 0.015, 2.78, 2.40 every ten steps) instead of settling. This is why PyTorch training loops call `optimizer.zero_grad()` every iteration.

## Back to vectorisation: the gradient as one formula

A scalar autograd is correct but slow. The mean squared error of a linear model on 200 rows and 5 features is a Value graph of 4,408 nodes, and its gradient took about 16 ms in one run. Working the same derivative out by hand on whole arrays gives `2/n · Xᵀ(Xw - y)`: about 0.005 ms, and the two gradients agree to 5.3e-15. Real autograd libraries do exactly this: their nodes are whole arrays and each node's backward rule is a vectorised formula, so the graph has a handful of nodes instead of thousands.

## LeetCode 150 - Evaluate Reverse Polish Notation

**The task.** Compute the value of an arithmetic expression written in Reverse Polish Notation, where each operator comes after its two operands, so no parentheses are needed.

**Input / output.** `tokens`, a list of strings, each either an integer or one of `+`, `-`, `*`, `/`. Return the integer value of the expression. Division between two integers keeps only the integer part, rounding toward zero.

**Example.** `["2", "1", "+", "3", "*"]` means `(2 + 1) * 3 = 9`: push 2, push 1, `+` pops both and pushes 3, push 3, `*` pops both and pushes 9. `["4", "13", "5", "/", "+"]` is `4 + 13 / 5 = 4 + 2 = 6`.

**Constraints.** Up to ten thousand tokens; every number is between -200 and 200; the expression is always valid, never divides by zero, and every intermediate result fits in a 32-bit integer. The constraint that bites is "rounding toward zero" for negative quotients.

The stack from Day 02 does all the work:

```python
class Solution:
    def evalRPN(self, tokens: list[str]) -> int:
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
```

The trap is Python's `//`, which rounds *down*: `6 // -132` is -1, while truncating toward zero gives 0. The positive examples pass either way. LeetCode's longer example, `10 6 9 3 + -11 * / * 17 + 5 +`, is 22 with `int(a / b)` and 12 with `//`.

Run the same stack machine over `Value` objects instead of integers and evaluating the expression *builds the computation graph*. `x y * x + y /` at `x = 2, y = -3` gives 1.3333, and one `backward()` gives `d/dx = 0.6667` and `d/dy = -0.2222` - exactly `(y + 1)/y` and `-x/y²`. That is all an autograd tape is: record operations while you evaluate, then replay them backwards.

[leetcode.com/problems/evaluate-reverse-polish-notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

## Complexity

| Operation | Time | Space |
|---|---|---|
| Elementwise NumPy op on n values | O(n), one C loop | O(n) for the result |
| Transpose, slice, contiguous reshape | O(1), new strides only | O(1) (a view) |
| Pairwise distances, broadcasting | O(n² d) | O(n² d) temporary |
| Pairwise distances, Gram trick | O(n² d) as one matrix product | O(n²) |
| Central difference gradient, n inputs | 2n function evaluations | O(1) extra |
| Forward mode (dual numbers), n inputs | n passes | O(1) per value |
| Reverse mode (Value.backward) | 1 forward + 1 backward, O(nodes) | O(nodes) to store the graph |
| LeetCode 566 / LeetCode 150 | O(m n) / O(tokens) | O(m n) / O(tokens) |

## References

- [NumPy: the N-dimensional array, strides and views](https://numpy.org/doc/stable/reference/arrays.ndarray.html) and [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Automatic differentiation - Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation) (forward mode, dual numbers, reverse mode)
- [Numerical differentiation - Wikipedia](https://en.wikipedia.org/wiki/Numerical_differentiation) (choice of step size)
- Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd) - the scalar autograd design this day's `Value` follows
- Baydin, Pearlmutter, Radul and Siskind, *Automatic Differentiation in Machine Learning: a Survey*, JMLR 2018
- [PyTorch: a gentle introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
