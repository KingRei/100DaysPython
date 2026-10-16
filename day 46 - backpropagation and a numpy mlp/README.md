# Backpropagation in pure NumPy: a two-layer network, the three lines that train it, and five ways to get them wrong without an error

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that overflows a naive softmax and
then fixes it, normalises on the wrong axis and shows the result still looking like
probabilities, collapses softmax and cross entropy into p − y, runs one forward and one
backward pass through a two-layer network rule by rule, replays four backprop bugs and
one gradient check that fails on correct code, and solves LeetCode 867 and LeetCode 48.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2046%20-%20backpropagation%20and%20a%20numpy%20mlp/imgs/day46_1.png?raw=true)

Backpropagation through a linear layer is three lines, and each one is a transpose away from the forward pass:

```python
dW = x.T @ dz
db = dz.sum(axis=0)
dx = dz @ W.T
```

That is the entire algorithm. A network is a list of layers, the forward pass caches what the backward pass will need, and the backward pass walks the same list in reverse. With those three lines and a classification loss, a two-layer network separates a three-armed spiral that no straight line can touch: 100% against 53% for the same model with the hidden layer removed.

The interesting part is not the three lines. It is that every way of getting them slightly wrong produces a training loop that runs to completion and hands you a model:

- `softmax` on the wrong axis still returns numbers between 0 and 1, and something still sums to 1.
- Cross entropy that builds the probabilities first takes `log(0)` and returns `inf`.
- `db = dz.sum()` instead of `dz.sum(axis=0)` is a scalar, and a scalar broadcasts.
- `dW = x @ dz` without the transpose runs fine whenever the batch size happens to equal the input width.
- Initialising every weight to zero leaves 64 hidden units behaving as one, forever.

And one failure that is *not* a bug: a gradient check on a ReLU network can report a relative error of 1.0 while the code is perfectly correct.

Day 44 built a scalar autograd; Day 45 replaced the update rule at the end of the loop. This is the model that goes in the middle. Run `python mlp.py` to print every number here.

## The losses an output layer needs, written so they cannot overflow

A classifier's last layer produces one real number per class - logits - and `softmax` turns them into a distribution. The textbook spelling is `exp(z) / exp(z).sum()`. Subtracting the row maximum first changes nothing mathematically, because the constant cancels in the ratio:

```python
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)
```

Numerically it is the difference between an answer and a `nan`:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2046%20-%20backpropagation%20and%20a%20numpy%20mlp/imgs/day46_4.png?raw=true)

```text
logits [1000.0, 1001.0, 1002.0]
  naive  [nan nan nan]          <- exp(1002) overflows, then inf/inf
  stable [0.090031 0.244728 0.665241]
```

On ordinary logits the two agree to `1.1e-16`, which is exactly why this never shows up in testing. It shows up the first time a logit grows, which is the first time something else has already gone slightly wrong - and then the `nan` spreads into every weight on the next backward pass.

Cross entropy has the same shape of problem one level up. Written as `-log(softmax(z)[label])`, it forms a probability and then takes its log. With logits `[-400, 0, 400]`, `exp(-800)` underflows to exactly zero, so the probability of class 0 is `0.0` and its log is `-inf`:

```text
label 0 (the model is wrong):   stable 800.0,  naive inf
```

A loss of 800 is enormous but finite, and it produces a usable gradient that pushes the model away from being confidently wrong. An `inf` is the end of training. The fix is to never build the probability:

```python
def log_softmax(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return z - m - np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
```

The two-class version is the same story. `1 / (1 + np.exp(-x))` at `x = -800` computes `1 / (1 + inf)` and happens to land on the right answer, `0.0`, after an overflow. The damage is downstream: `log(0.0)` is `-inf`, and the loss is the thing being differentiated. `log_sigmoid(-800)` returns `-800.0`, and binary cross entropy taken straight from the logits with `np.logaddexp(0, z) - y * z` never forms a probability at all.

## The axis bug that still returns probabilities

This one deserves its own section because it is the most common and the least visible. A batch of logits has shape `(n_examples, n_classes)`, and the softmax has to run across the classes:

```text
softmax(axis=1), per example:     row sums [1. 1.]        <- what we want
softmax(axis=0), per class:       row sums [1.475 1.525]  column sums [1. 1. 1.]
```

With `axis=0` every entry is still in `[0, 1]` and something still sums to 1 - the classes are just being normalised across the batch instead of within each example. No assertion fires. On this example even the `argmax` of each row is unchanged, `[0, 1]` either way, so the predictions look fine and only the loss is wrong. A model trained this way is being told that examples compete with each other for a class, which is a coherent but completely different problem.

## Why softmax and cross entropy are one function

Differentiate them separately and you need the softmax Jacobian - a full `k x k` matrix per example - and then a division by `p` from the log. Compose them and almost everything cancels, leaving a subtraction:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2046%20-%20backpropagation%20and%20a%20numpy%20mlp/imgs/day46_3.png?raw=true)

```python
def softmax_ce_grad(logits, y, mean=True):
    n = len(y)
    p = softmax(logits)
    p[np.arange(n), y] -= 1.0
    return p / n if mean else p
```

The gradient of the loss with respect to the logits is the predicted distribution minus the one-hot label. Checked against central differences, the largest difference is `2.47e-11`. That identity is why every framework exposes a fused `cross_entropy(logits, labels)` and why passing it probabilities instead of logits is a documented mistake: by the time you have the probabilities, the cancellation has already been thrown away.

## The layers

With the loss gradient in hand, the rest is mechanical. A linear layer caches its input going forward:

```python
def forward(self, x):
    self.x = x
    return x @ self.W + self.b

def backward(self, dz):
    self.dW = self.x.T @ dz
    self.db = dz.sum(axis=0)
    return dz @ self.W.T
```

Each of the three lines is forced by shapes, and each has a reading. `dW = x.T @ dz` sums, over the batch, the outer product of what came in with the gradient of what went out. `db = dz.sum(axis=0)` undoes a broadcast: the bias was added to every row, so its gradient collects from every row. `dx = dz @ W.T` is the same matrix the forward pass used, transposed, which is what makes the backward pass cost about the same as the forward one.

ReLU's backward is `dz * (x > 0)`, which means the whole network is:

```python
def backward(self, dlogits):
    for layer in reversed(self.layers):
        dlogits = layer.backward(dlogits)
```

`reversed(self.layers)` is Day 13's reverse topological order, except that a plain feed-forward network is a straight line, so reversing a list is enough. The moment a network has skip connections or shares a layer between two branches, the list is no longer sufficient and a real topological sort is - which is exactly the step Day 44's scalar autograd had to do.

## Checking it

A hand-written gradient is a claim, and central differences are how the claim gets tested. Perturb one entry of one parameter array by `h`, measure the loss on both sides, divide:

```text
gradcheck, correct backward:  7.85e-09
```

That number is the whole reason to write the check before trusting anything below.

## Training it

A three-armed spiral, 300 points, three classes, trained full-batch with Day 45's Adam:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2046%20-%20backpropagation%20and%20a%20numpy%20mlp/imgs/day46_2.png?raw=true)

```text
step     1  loss 1.1068  accuracy 0.633
step   200  loss 0.0036  accuracy 1.000
step  1200  loss 0.0001  accuracy 1.000

one hidden layer of 64 ReLU units:     accuracy 1.000
no hidden layer (softmax regression):  accuracy 0.530
```

Both figures are on the training set: this is a demonstration that the model can represent the shape at all, not a claim about generalisation. The comparison is the point. The output layer is still a linear classifier drawing straight boundaries - look at the right-hand panel, which is all a linear model can ever do. The hidden layer does not add curvature to the boundary directly; it bends the *space*, and then the output layer draws its straight lines in the bent space. That is the whole idea of a deep network in one picture.

## Four bugs that never raise

**`db = dz.sum()` instead of `dz.sum(axis=0)`.** The correct gradient has one entry per output unit. `dz.sum()` collapses it to a scalar, and `b -= lr * scalar` broadcasts without complaint, moving every bias by the same amount. Gradient check: `1.00e+00`.

**Forgetting the division by n.** The loss reported is the mean over the batch; if the gradient is the gradient of the *sum*, every step is `n` times larger than intended. Nothing breaks visibly - it looks like a learning rate that is suddenly too high, which is the hardest kind of bug to attribute. Gradient check: `8.46e-01`.

**`dW = x @ dz` without the transpose.** `x` is `(n, d_in)` and `dz` is `(n, d_out)`, so this is a shape error - unless `n == d_in`, in which case it runs and produces a `(d_in, d_out)` array of the right shape full of wrong numbers. With a batch of 8 and an input width of 8, the gradient check says `1.00e+00`; change the batch size and the same line becomes a loud `ValueError`. A bug that depends on the batch size is a bug that will appear in production and not in your test.

**All-zero initialisation.**

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2046%20-%20backpropagation%20and%20a%20numpy%20mlp/imgs/day46_5.png?raw=true)

```text
all-zero init: 1 distinct hidden activation value out of 256
accuracy after 400 steps 0.333, against 1.000 for the randomly initialised net
```

If every weight is identical, every hidden unit computes the same thing, receives the same gradient and takes the same step - forever. Sixty-four units behave as one, and `0.333` on three classes is chance. Symmetry has to be broken by the initialisation; this is the reason random init is not an implementation detail. Biases can start at zero because the weights already break the symmetry.

## One failure that is not a bug

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2046%20-%20backpropagation%20and%20a%20numpy%20mlp/imgs/day46_6.png?raw=true)

ReLU has no derivative at exactly zero. Backpropagation picks a side - `mask = x > 0`, so zero. Central differences average both sides: `(h - 0) / 2h = 0.5`. Feed the network a row of zeros, and with biases starting at zero the pre-activations land exactly on the corner:

```text
4 of 8 pre-activations are exactly 0
ReLU gradcheck 1.00,  the same network with tanh 3.4e-10
```

The code is right and the check is right; the function has a corner. I found this the honest way - my first spiral started every arm at radius 0, so all three classes shared the origin, and that one point made the gradient check fail on a correct implementation. Two lessons came out of it: gradient-check with a smooth activation, and look at your data when a check fails before assuming the gradient is wrong.

## LeetCode 867 - Transpose Matrix

**The task.** Return the transpose of a matrix: the element at row `i`, column `j` moves to row `j`, column `i`.

**Input / output.** `matrix`, an `m x n` list of lists of integers. Return the `n x m` transposed list of lists.

**Example.** `[[1, 2, 3], [4, 5, 6]]` becomes `[[1, 4], [2, 5], [3, 6]]`: the first row becomes the first column.

**Constraints.** `m` and `n` are between 1 and 1000, and `m * n` is at most 10⁵, so a linear pass over every element is expected. The matrix is not necessarily square, which is what rules out the in-place swap used in the next problem.

```python
class Solution:
    def transpose(self, matrix: list[list[int]]) -> list[list[int]]:
        m, n = len(matrix), len(matrix[0])
        out = [[0] * m for _ in range(n)]
        for i in range(m):
            for j in range(n):
                out[j][i] = matrix[i][j]
        return out
```

This is the operation every backward rule above is built on, and it is worth noticing what it costs. The list-of-lists version has to move every element. NumPy's `a.T` moves nothing: it returns a view with the strides swapped, `(24, 8)` becoming `(8, 24)`, sharing memory with the original - the Day 44 result. That is why `x.T @ dz` costs no more than `x @ dz`, and why a backward pass costs roughly what a forward pass costs rather than several times more.

[leetcode.com/problems/transpose-matrix](https://leetcode.com/problems/transpose-matrix/)

## LeetCode 48 - Rotate Image

**The task.** Rotate an `n x n` matrix by 90 degrees clockwise, in place.

**Input / output.** `matrix`, a square list of lists. Return nothing; modify the input. Allocating another `n x n` matrix and copying it back is explicitly disallowed.

**Example.** `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` becomes `[[7, 4, 1], [8, 5, 2], [9, 6, 3]]`: the first column, read bottom to top, becomes the first row.

**Constraints.** `n` is between 1 and 20, so the work is trivial and the whole problem is the in-place requirement. Being square is what makes an in-place transpose possible at all.

A clockwise rotation is a transpose followed by reversing each row:

```python
class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):        # j > i only, or every pair swaps twice
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for row in matrix:
            row.reverse()
```

The trap is the inner loop's starting index. With `for j in range(n)`, every pair `(i, j)` is swapped once as `(i, j)` and once as `(j, i)`, so the matrix comes back exactly as it started and the transpose silently does nothing. The function still returns, still modifies the matrix, and still produces a square matrix of the same numbers - just the input with its rows reversed:

```text
rotated (j > i):       [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
j over every column:   [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
```

That is the same flavour of failure as the whole day: an off-by-one in a loop bound that leaves every shape intact and every value plausible.

[leetcode.com/problems/rotate-image](https://leetcode.com/problems/rotate-image/)

## Complexity

| Operation | Time | Space |
|---|---|---|
| Forward pass, batch n, layer d_in x d_out | O(n · d_in · d_out) | O(n · d_out) activations |
| Backward pass, same layer | O(n · d_in · d_out), about the same as forward | O(d_in · d_out) gradients |
| Caching for backward | - | one activation array per layer |
| softmax / log_softmax over k classes | O(n · k) | O(n · k) |
| softmax + cross entropy gradient | O(n · k), one subtraction | O(n · k) |
| Gradient check, P parameters | 2P forward passes | O(1) extra |
| LeetCode 867 | O(m · n) | O(m · n) output |
| LeetCode 48 | O(n²) | O(1) |

## References

- Rumelhart, Hinton and Williams, *Learning representations by back-propagating errors*, Nature 1986
- [CS231n: Backpropagation, intuitions](https://cs231n.github.io/optimization-2/) and [the spiral dataset](https://cs231n.github.io/neural-networks-case-study/)
- [Backpropagation - Wikipedia](https://en.wikipedia.org/wiki/Backpropagation) and [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp)
- Glorot and Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010 (Xavier initialisation)
- He, Zhang, Ren and Sun, [*Delving Deep into Rectifiers*](https://arxiv.org/abs/1502.01852), ICCV 2015 (the He initialisation used here)
- [PyTorch docs: torch.nn.functional.cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) - takes logits, not probabilities, for the reason in this article
