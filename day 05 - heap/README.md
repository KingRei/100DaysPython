# Heap

More details in:
https://medium.com/100-days-of-python/day-05-data-structure-priority-queue-binary-heap-6a35149c8c16

![Heap](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Max-Heap.svg/501px-Max-Heap.svg.png)

In computer science, a heap is a specialized tree-based data structure that satisfies the heap property: if P is a parent node of C, then the key (the value) of P is either greater than or equal to (in a max heap) or less than or equal to (in a min heap) the key of C. The node at the "top" of the heap (with no parents) is called the root node.

The datas of binary heap are actually stored a array list like this:
![Heap](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Heap-as-array.svg/603px-Heap-as-array.svg.png)

Time Complexity:

| Operation       | find-min    | delete-min    | insert      | decrease-key  | merge      |
| :------         | :-----      | :-----        | :-----      | :-----        | :-----     |
| Binary Heap     | Θ(1)        | Θ(log(n))     | O(log(n))   | Θ(log(n))     | Θ(n)       |


## References

- [Wikipedia](https://en.wikipedia.org/wiki/Heap_(data_structure))
- [Youtube - Hacker Rank](https://www.youtube.com/watch?v=t0Cq6tVNRBA&index=5&t=0s&list=PLLXdhg_r2hKA7DPDsunoDZ-Z769jWn4R8)
