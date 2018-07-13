# Hash Table - Open Addressing

More details in:
https://medium.com/100-days-of-python/day-10

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Hash_table_4_1_1_0_0_1_0_LL.svg/300px-Hash_table_4_1_1_0_0_1_0_LL.svg.png)

Typically, the domain of a hash function (the set of possible keys) is larger than its range (the number of different table indices), and so it will map several different keys to the same index which could result in collisions.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Hash_table_5_0_1_1_1_1_0_SP.svg/380px-Hash_table_5_0_1_1_1_1_0_SP.svg.png)

In another strategy from chaining, called open addressing. When a new entry has to be inserted, the buckets are examined, starting with the hashed-to slot and proceeding in some probe sequence, until an unoccupied slot is found. When searching for an entry, the buckets are scanned in the same sequence, until either the target record is found, or an unused array slot is found, which indicates that there is no such key in the table.The name "open addressing" refers to the fact that the location ("address") of the item is not determined by its hash value. 

Well-known probe sequences include:

- Linear probing, in which the interval between probes is fixed (usually 1)
- Quadratic probing, in which the interval between probes is increased by adding the successive outputs of a quadratic polynomial to the starting value given by the original hash computation
- Double hashing, in which the interval between probes is computed by a second hash function


## References

- [Wiki](https://en.wikipedia.org/wiki/Hash_table)
- [YouTube- HackerRank](https://www.youtube.com/watch?v=shs0KM3wKv8)
