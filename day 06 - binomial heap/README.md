# Binomial Heap

More details in:
https://medium.com/100-days-of-python/day-06-binomial-heap-88ca2edb8255

![Binomial Heap](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Binomial_Trees.svg/500px-Binomial_Trees.svg.png)

A binomial heap is implemented as a set of binomial trees (compare with a binary heap, which has a shape of a single binary tree), which are defined recursively as follows:

1. A binomial tree of order 0 is a single node
2. A binomial tree of order k has a root node whose children are roots of binomial trees of orders k−1, k−2, ..., 2, 1, 0 (in this order).

Time Complexity:

| Operation   | find-min    | delete-min    | insert      | decrease-key  | merge      |
| :------     | :-----      | :-----        | :-----      | :-----        | :-----     |
| Binary      | Θ(1)        | Θ(log(n))     | O(log(n))   | Θ(log(n))     | Θ(n)       |
| Fibonacci   | Θ(1)        | O(log(n))     | Θ(1)        | Θ(1)          | Θ(1)       |


## References

- [Wikipedia](https://en.wikipedia.org/wiki/Binomial_heap)
- [Youtube](https://www.youtube.com/watch?v=m2T3cZWniXI)
- [Camdemy](http://u.camdemy.com/media/474)
