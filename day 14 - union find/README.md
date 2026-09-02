# Union-Find (Disjoint Set Union)

More details in:
https://medium.com/100-days-of-python/day-14-union-find-%E5%8F%AA%E5%9B%9E%E7%AD%94%E4%B8%80%E5%80%8B%E5%95%8F%E9%A1%8C%E7%9A%84%E8%B3%87%E6%96%99%E7%B5%90%E6%A7%8B-8a0819fe1560

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline) that animates union by rank, path compression, the
chain a naive union builds, and LeetCode 547 step by step.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2014%20-%20union%20find/imgs/day14_1.png?raw=true)

Some questions have nothing to do with paths. *Are these two accounts the same person?
How many separate networks are left after that cable failed? Does adding this edge close
a cycle?* All you need is a partition of a set of elements into groups, the ability to
merge two groups, and a way to ask whether two elements belong to the same group.

**Union-Find**, also called the disjoint set union (DSU) structure, answers exactly those
questions using nothing but an integer array. Each set is stored as a tree; every element
holds the index of its parent, and a root points at itself. The root is therefore the id
of the set, so `find(x)` walks up to the root and two elements are in the same set exactly
when their roots match.

A `union` that blindly hangs one root under the other is correct but fragile: a handful of
unlucky merges builds a single long chain and every `find` degrades to O(n). Two short
additions fix that permanently.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2014%20-%20union%20find/imgs/day14_2.png?raw=true)

## Union by rank (or size)

Always attach the shorter tree under the taller one. `rank[r]` is an upper bound on the
height of the tree rooted at `r`; when the two ranks differ the taller tree absorbs the
shorter one and its height does not change at all. Only a tie can add a level, so the
height stays O(log n). Tracking `size[r]` instead works just as well and gives you "how
many elements are in this set" for free.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2014%20-%20union%20find/imgs/day14_3.png?raw=true)

## Path compression

`find` already had to walk from `x` up to the root. Walk that path a second time and point
every node on it straight at the root. The next `find` on any of them is a single hop, and
the work is never repeated.

With both optimisations the amortised cost per operation is O(α(n)), where α is the inverse
Ackermann function. α(n) is at most 4 for any input that will ever exist, so in practice
every operation is constant time.

## LeetCode 547 - Number of Provinces

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2014%20-%20union%20find/imgs/day14_4.png?raw=true)

`isConnected[i][j] == 1` means city *i* and city *j* are directly connected; a province is
a connected component. Union every connected pair and the number of remaining sets is the
answer. The matrix is symmetric with a diagonal of ones, so only the strict upper triangle
carries information.

A traversal solves this one too, in the same O(n²) - the matrix has to be read either way.
Union-Find earns its keep when the edges arrive over time: BFS has to restart from scratch
after every new edge, while Union-Find simply absorbs it.

## Complexity

| Operation | Naive | Union by rank + path compression |
|---|---|---|
| `find` | O(n) worst case | O(α(n)) amortised |
| `union` | O(n) worst case | O(α(n)) amortised |
| `connected` | O(n) worst case | O(α(n)) amortised |
| space | O(n) | O(n) |

## Run it

```sh
python union_find.py     # standalone walkthrough, no Jupyter required
```

Or open `demo.html` in a browser for the animated version (中文 / English toggle).

## References
- [Wikipedia - Disjoint-set data structure](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)
- [Wikipedia - Ackermann function](https://en.wikipedia.org/wiki/Ackermann_function#Inverse)
- [LeetCode 547 - Number of Provinces](https://leetcode.com/problems/number-of-provinces/)
