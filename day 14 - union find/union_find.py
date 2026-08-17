"""Day 14 - Union-Find (disjoint set union), with path compression and
union by rank/size.

Run it:  python3 union_find.py
"""
from collections import defaultdict
from random import Random


# --------------------------------------------------------------------------
# the naive version - correct, but a find can degrade to O(n)
# --------------------------------------------------------------------------
class NaiveUnionFind:
    """No compression, no rank: always hang y's root under x's root."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.steps = 0            # how many pointer hops find() has paid

    def find(self, x):
        while self.parent[x] != x:
            self.steps += 1
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        self.parent[ry] = rx      # blind attach - this is what builds a chain
        return True


# --------------------------------------------------------------------------
# the real thing
# --------------------------------------------------------------------------
class UnionFind:
    """Disjoint set union over 0..n-1.

    parent[i] == i marks a root, and the root *is* the set id.
    rank[r] is an upper bound on the height of the tree rooted at r.
    size[r] is the number of elements in that set (handy on its own).
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.count = n            # number of disjoint sets right now

    # -- find with path compression ---------------------------------------
    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # second pass: point everything on the path straight at the root
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def find_recursive(self, x):
        """Same thing, three lines, but recursion depth is the tree height."""
        if self.parent[x] != x:
            self.parent[x] = self.find_recursive(self.parent[x])
        return self.parent[x]

    # -- union by rank -----------------------------------------------------
    def union(self, x, y):
        """Merge the two sets. Returns False if they were already merged."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:     # keep the taller tree on top
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:    # only a tie can grow the height
            self.rank[rx] += 1
        self.count -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def set_size(self, x):
        return self.size[self.find(x)]

    def groups(self):
        """{root: sorted members} - only for printing, O(n α(n))."""
        out = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# LeetCode 547 - Number of Provinces
# --------------------------------------------------------------------------
def find_circle_num(is_connected):
    """isConnected[i][j] == 1 means city i and city j are directly linked.
    Count the connected components."""
    n = len(is_connected)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):          # the matrix is symmetric
            if is_connected[i][j]:
                uf.union(i, j)
    return uf.count


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
def main():
    print('building a forest')
    print('-----------------')
    uf = UnionFind(7)
    for a, b in [(0, 1), (0, 2), (2, 3), (4, 5)]:
        uf.union(a, b)
        print(f'  union({a}, {b})  ->  parent = {uf.parent}   sets = {uf.count}')
    print(f'  groups     : {uf.groups()}')
    print(f'  connected(3, 1) = {uf.connected(3, 1)}   '
          f'connected(3, 5) = {uf.connected(3, 5)}')
    print(f'  set_size(3) = {uf.set_size(3)}')
    assert uf.count == 3
    assert uf.connected(3, 1) and not uf.connected(3, 5)
    assert uf.set_size(3) == 4

    print()
    print('path compression in action')
    print('--------------------------')
    chain = UnionFind(4)
    chain.parent = [0, 0, 1, 2]        # a deliberate 3 -> 2 -> 1 -> 0 chain
    print(f'  before find(3): parent = {chain.parent}')
    chain.find(3)
    print(f'  after  find(3): parent = {chain.parent}   '
          '(everything now points at the root)')
    assert chain.parent == [0, 0, 0, 0]

    print()
    print('why union by rank matters')
    print('-------------------------')
    n = 20000
    naive = NaiveUnionFind(n)
    for i in range(1, n):
        naive.union(i, i - 1)          # worst case: builds one long chain
    before = naive.steps
    for i in range(n):
        naive.find(i)
    print(f'  naive   : {naive.steps - before:>12,} pointer hops for {n:,} finds')

    smart = UnionFind(n)
    for i in range(1, n):
        smart.union(i, i - 1)
    hops = 0
    for i in range(n):
        x = i
        while smart.parent[x] != x:
            hops += 1
            x = smart.parent[x]
    print(f'  rank+pc : {hops:>12,} pointer hops for {n:,} finds')
    print('  same answers, but the tree never gets tall enough to matter')
    assert naive.steps - before > 50 * hops

    print()
    print('LeetCode 547 - Number of Provinces')
    print('----------------------------------')
    cases = [
        ([[1, 1, 0], [1, 1, 0], [0, 0, 1]], 2),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3),
        ([[1, 1, 0, 0, 0],
          [1, 1, 0, 0, 0],
          [0, 0, 1, 1, 0],
          [0, 0, 1, 1, 0],
          [0, 0, 0, 0, 1]], 3),
    ]
    for grid, want in cases:
        for row in grid:
            print('    ' + ' '.join(str(v) for v in row))
        got = find_circle_num(grid)
        print(f'    -> {got} province(s)   {"ok" if got == want else "WRONG"}')
        print()
        assert got == want

    print('randomised cross-check against a set-of-sets model')
    print('--------------------------------------------------')
    rng = Random(14)
    for trial in range(200):
        n = rng.randint(1, 30)
        uf = UnionFind(n)
        model = [{i} for i in range(n)]
        for _ in range(n * 2):
            a, b = rng.randrange(n), rng.randrange(n)
            uf.union(a, b)
            sa = next(s for s in model if a in s)
            sb = next(s for s in model if b in s)
            if sa is not sb:
                sa |= sb
                model.remove(sb)
            assert uf.count == len(model)
            for x in range(n):
                for y in range(n):
                    same = any(x in s and y in s for s in model)
                    assert uf.connected(x, y) == same
    print(f'  200 random trials, every connected() query matched the model')

    print()
    print('all assertions passed')


if __name__ == '__main__':
    main()
