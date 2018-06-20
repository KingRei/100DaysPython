# Heap

More details in:
https://medium.com/100-days-of-python/day-05


In computer science, a heap is a specialized tree-based data structure that satisfies the heap property: if P is a parent node of C, then the key (the value) of P is either greater than or equal to (in a max heap) or less than or equal to (in a min heap) the key of C. The node at the "top" of the heap (with no parents) is called the root node.


![Heap](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Max-Heap.svg/501px-Max-Heap.svg.png)

Time Complexity:

| Time Complexity    | access    | search    | insert    | delete    |
| :------            | :-----    | :-----    | :-----    | :-----    |
| Average            | Θ(log(n)) | Θ(log(n)) | Θ(log(n)) | Θ(log(n)) |
| Worst              | O(n)      | O(n)      | O(n)      | O(n)      |


## References

- [Wikipedia](https://en.wikipedia.org/wiki/Heap_(data_structure))
- [Toutube - Hacker Rank](https://www.youtube.com/watch?v=t0Cq6tVNRBA&index=5&t=0s&list=PLLXdhg_r2hKA7DPDsunoDZ-Z769jWn4R8)
