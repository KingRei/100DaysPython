# Tree 

More details in:
https://medium.com/100-days-of-python/day-03-tree-fe4bfcb4c8e7

![tree](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Binary_tree.svg/300px-Binary_tree.svg.png)

In computer science, a tree is a widely used abstract data type (ADT) that simulates a hierarchical tree structure, with a root value and subtrees of children with a parent node, represented as a set of linked nodes.

## Binary Search Tree

![BST](https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Binary_search_tree.svg/300px-Binary_search_tree.svg.png)

A binary search tree is a rooted binary tree, whose internal nodes each store a key (and optionally, an associated value) and each have two distinguished sub-trees, commonly denoted left and right. The tree additionally satisfies the binary search property, which states that the key in each node must be greater than or equal to any key stored in the left sub-tree, and less than or equal to any key stored in the right sub-tree. The leaves (final nodes) of the tree contain no key and have no structure to distinguish them from one another.

Time Complexity:

| Average | access    | search    | insert    | delete    || Worst   | access    | search    | insert    | delete    |
| :------ | :-----    | :-----    | :-----    | :-----    || :------ | :-----    | :-----    | :-----    | :-----    |
|         | Θ(log(n)) | Θ(log(n)) | Θ(log(n)) | Θ(log(n)) ||         | O(n)      | O(n)      | O(n)      | O(n)      |

## References

- [Wikipedia](https://en.wikipedia.org/wiki/Tree_(data_structure))
- [YouTube - Hacker Rank](https://www.youtube.com/watch?v=oSWTXtMglKE&list=PLLXdhg_r2hKA7DPDsunoDZ-Z769jWn4R8&index=8)
