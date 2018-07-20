# Graph

More details in:
https://medium.com/100-days-of-python/day-11-data-structure-graph-a4229c3dabaf

![img](https://cdn-images-1.medium.com/max/800/1*Q9n58avTamrRmY66Ne0Hug.png)

In computer science, a graph is an abstract data type that is meant to implement the undirected graph and directed graph concepts from mathematics, specifically the field of graph theory.

Different data structures for the representation of graphs are used in practice:

- Adjacency list
Vertices are stored as records or objects, and every vertex stores a list of adjacent vertices. This data structure allows the storage of additional data on the vertices. Additional data can be stored if edges are also stored as objects, in which case each vertex stores its incident edges and each edge stores its incident vertices.

- Adjacency matrix
A two-dimensional matrix, in which the rows represent source vertices and columns represent destination vertices. Data on edges and vertices must be stored externally. Only the cost for one edge can be stored between each pair of vertices.

- Incidence matrix
A two-dimensional Boolean matrix, in which the rows represent the vertices and columns represent the edges. The entries indicate whether the vertex at a row is incident to the edge at a column.


## References

- [Wiki](https://en.wikipedia.org/wiki/Graph_(abstract_data_type))
- [YouTube-Introduction to Graphs](https://www.youtube.com/watch?v=gXgEDyodOJU&index=9&list=PLLXdhg_r2hKA7DPDsunoDZ-Z769jWn4R8)
- [YouTube- Graphs representation](https://www.youtube.com/watch?v=k1wraWzqtvQ&index=10&list=PLLXdhg_r2hKA7DPDsunoDZ-Z769jWn4R8)
- [Keith Horwood's Blog](https://medium.com/@keithwhor/using-graph-theory-to-build-a-simple-recommendation-engine-in-javascript-ec43394b35a3)
