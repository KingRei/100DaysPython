# Concurrency and parallelism in Python: threads vs processes vs asyncio, catching races and deadlocks, and measuring parallelism with work and depth

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that plays two threads under one GIL
against two processes, steps through all six interleavings of two `x += 1`, walks five
philosophers into the one deadlocked state (and out of it with lock ordering), runs ten
`asyncio` fetches with and without a blocking call, crawls LeetCode 1242's site in waves,
schedules tree sums and Hillis-Steele and Blelloch scans on 4 workers, and shuffles a
map-reduce word count into the wrong buckets.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2043%20-%20concurrency%20and%20parallelism/imgs/day43_1.png?raw=true)

Two CPU-heavy Python functions, run on two threads, took 0.244 s on my 2-core machine - slower than running them one after the other (0.207 s). The same two functions on two *processes* took 0.156 s. Twenty 50 ms network-style waits took one second in a loop and 0.058 s on twenty threads. Same machine, same `threading` module: once a 17x win, once a loss.

The rule behind those numbers is one sentence: **in CPython, threads overlap waiting, not computing.** Which tool to reach for - threads, processes, `asyncio` - depends only on whether the job spends its time computing or waiting. And whether it is worth splitting at all depends on a second question with a precise answer: how much of the computation can run at the same time? That is measured by two numbers, *work* and *depth*.

The other half of the day is about bugs. Concurrency bugs behave like yesterday's randomness bugs: they do not crash, they depend on luck, and the tests pass.

- Four threads adding 1 to a shared counter 200,000 times each gave exactly 800,000 every time - until I put one function call between reading the counter and writing it back. Then between a quarter and two thirds of the updates vanished, depending on the run.
- Five dining philosophers who all pick up their left fork first can freeze forever - in 120 of 2,430,842,120 possible interleavings. One in twenty million: no test run will ever find it.
- A map-reduce word count where two mappers disagree about which reducer gets a word reported only part of the real count - about half - for 51 of 67 words. With Python's own `hash()` that happens on almost every run on Windows and macOS, and never with Linux's classic `fork` workers.

Day 42 tested randomized code by replaying *every* possible sequence of random choices. Today does the same for threads: a small checker that runs every possible interleaving, so a one-in-twenty-million deadlock is simply one of the states it visits.

Run `python concurrency.py` to print every number on this page (timings will differ on your machine; everything else is exact).

## The GIL: computing vs waiting

CPython executes bytecode on one thread at a time. A global interpreter lock (the GIL) is passed between threads, by default every 5 ms. So two threads doing pure-Python arithmetic take turns on one core, and the second core sits idle. A thread that is *waiting* - `time.sleep`, a socket read, a disk read - releases the GIL first, so waits overlap freely.

```python
def burn(n):
    s = 0
    for i in range(n):
        s += i * i
    return s
```

One run on my 2-core VM:

```text
2 CPU jobs, serial   : 0.207 s   speed-up 1.00x
2 CPU jobs, threads  : 0.244 s   speed-up 0.85x
2 CPU jobs, processes: 0.156 s   speed-up 1.33x
20 waits of 50 ms, serial : 1.004 s   speed-up 1.0x
20 waits of 50 ms, threads: 0.058 s   speed-up 17.5x
```

Across runs the processes came out between 1.3 and 2 times faster than the loop - each process has its own interpreter and its own GIL, so both cores work, minus the cost of starting workers and shipping arguments to them. Threads never meaningfully beat the loop on the CPU job; the hand-offs usually cost them a little.

The practical table:

| The job mostly...                       | Use                                            |
|-----------------------------------------|------------------------------------------------|
| waits (network, disk, subprocesses)     | threads, or asyncio for thousands of waits      |
| computes in pure Python                 | processes (`multiprocessing`, `ProcessPoolExecutor`) |
| computes inside NumPy / C extensions    | threads often work: many release the GIL        |

(Python 3.13 added an optional build without the GIL. In that build, threads do run Python code in parallel - and every race in the next section stops being rare.)

Processes are not free either. Squaring 50,000 numbers took about 4 ms in a loop; `pool.map(square, nums, chunksize=1)`, which sends each number to a worker as its own message, took several seconds - more than a thousand times slower. With the default chunk size (big batches per message) it took 7 to 17 ms across runs: still slower than the loop, because squaring a number is cheaper than sending it to another process. A process pool pays off only when each task carries much more work than it costs to send it.

## A race you can run

`x = x + 1` is three steps: read `x`, add one, write `x`. If another thread writes between my read and my write, its update is overwritten and lost.

```python
def bump_inline(box):
    box.v += 1                      # read, add, write - one line

def bump_with_call(box):
    v = box.v                       # read
    _noop()                         # any function call: logging, validation, ...
    box.v = v + 1                   # write the value read *before* the call
```

Four threads, 200,000 calls each, so the answer must be 800,000:

```text
box.v += 1                      -> 800,000
read, call a function, write    -> 566,862  (29% of the updates lost)
same, inside `with lock:`       -> 800,000
```

The first line is the trap. In the CPython I am running (3.10), the interpreter only switches threads at a few places, such as jumps back to the top of a loop and function calls, so the three steps of `box.v += 1` happen never to be split. The test passes. That is an accident of this version, not a guarantee: older versions did split it, the free-threaded build does, and the moment someone adds a log line between the read and the write, 29 to 64 percent of the updates disappeared across my runs. The real fix is the third line: a `threading.Lock` makes read-and-write one step that nobody can cut into.

## Every interleaving, counted

A run of real threads shows one interleaving, chosen by the operating system. To see all of them I model each thread as a list of atomic steps and let a checker run every order in which they can interleave - each thread's own steps stay in order, everything else is fair game. For two threads each doing `r = x; x = r + 1`:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2043%20-%20concurrency%20and%20parallelism/imgs/day43_2.png?raw=true)

Four of the six interleavings lose an update: whenever both reads happen before both writes. With more threads it gets worse quickly:

```text
2 threads, one x += 1 each:     6 interleavings, correct (2) in   2 = 33.3%
3 threads, one x += 1 each:    90 interleavings, correct (3) in   6 = 6.7%
4 threads, one x += 1 each:  2520 interleavings, correct (4) in  24 = 1.0%
3 threads with a lock: 6 interleavings possible, all give [3]
```

The correct interleavings are exactly the ones where the threads happen to run one after another - 2, 6, 24 of them, the number of ways to order the threads. A lock does not make the bad interleavings rare; it makes them impossible. Only the 6 one-after-another orders remain.

The checker understands six kinds of step - read, write, acquire a lock, release it, wait for an event, set it - and a state is just "where is each thread, what is in memory, who holds which lock". Counting interleavings one by one would take forever, so it counts them per state and reuses the count whenever two different histories arrive at the same state:

```python
def count_schedules(threads, shared):
    @lru_cache(maxsize=None)
    def go(state):
        nxt = [i for i in range(len(threads)) if _enabled(threads, state, i)]
        if not nxt:                                   # finished, or nobody can move
            return ((_outcome(threads, state), 1),)
        total = Counter()
        for i in nxt:
            for k, v in go(_step(threads, state, i)):
                total[k] += v
        return tuple(total.items())
    return Counter(dict(go(_start(threads, shared))))
```

If nobody can move and not every thread has finished, that is a deadlock - which is the next section.

## Deadlock, and LeetCode 1226

Two threads, two locks. Thread 1 takes A then B; thread 2 takes B then A. Of the 6 interleavings, 2 end with each thread holding one lock and waiting forever for the other. If both take A first, only 2 interleavings exist and none deadlocks. That is the whole cure: **agree on one order for all locks.**

### LeetCode 1226 - The Dining Philosophers

**The task.** Five philosophers sit around a round table with one fork between each pair of neighbours, five forks in all. To eat, a philosopher needs both the fork on their left and the one on their right. Write the method each philosopher calls when they want to eat so that no philosopher can be stuck forever.

**Input / output.** Implement `wantsToEat(philosopher, pickLeftFork, pickRightFork, eat, putLeftFork, putRightFork)`. `philosopher` is a number from 0 to 4 going around the table; the five other arguments are functions to call in a valid order: pick up both forks, eat, put both down. Each philosopher is its own thread and calls `wantsToEat` `n` times. The judge records the sequence of calls and checks that no fork is ever held by two philosophers and that everybody finishes.

**Example.** With `n = 1`, every philosopher eats once. One valid record is: philosopher 4 picks up both forks, eats, puts both down; then philosopher 2 does the same; then 1, 0 and 3. Any order is accepted as long as neighbours never hold the same fork at the same time and the program finishes.

**Constraints.** `n` is between 1 and 60. The constraint that matters is not a number but the setup: all five threads start at the same time and the scheduler, not you, decides who runs. A solution that works "almost always" is a solution that hangs on the judge.

In my numbering philosopher `i` has fork `i` on the left and fork `(i + 1) % 5` on the right. If everybody takes the left fork first, the checker finds this:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2043%20-%20concurrency%20and%20parallelism/imgs/day43_3.png?raw=true)

```text
n = 2:    24 states, 1 deadlocked; 2 of 6 interleavings deadlock (1 in 3)
n = 3:   140 states, 1 deadlocked; 6 of 234 interleavings deadlock (1 in 39)
n = 4: 1,050 states, 1 deadlocked; 24 of 1,167,000 interleavings deadlock (1 in 48,625)
n = 5: 9,302 states, 1 deadlocked; 120 of 2,430,842,120 interleavings deadlock (1 in 20,257,017)
n = 5, lower-numbered fork first: 8,447 states, 0 deadlocked
```

There is exactly one bad state - everybody holding their left fork - and with five philosophers only one interleaving in about twenty million reaches it. You will not see it in testing. You will see it in production, on the busiest day, when the timing is just so. Real threads confirm it once the timing is forced: when all five philosophers grab their left fork and wait at a `threading.Barrier`, all five ([0, 1, 2, 3, 4]) then find their right fork taken.

The fix is the lock-ordering rule from above. The deadlock needs a cycle - everyone waiting on the next one around the table - and a cycle needs at least one philosopher to take a higher-numbered fork before a lower one. Make everybody take the lower-numbered fork first and philosopher 4 reaches for fork 0 before fork 4; the cycle cannot close. The checker agrees: 8,447 states, none stuck.

```python
class DiningPhilosophers:
    def __init__(self, n=5):
        self.n = n
        self.forks = [threading.Lock() for _ in range(n)]

    def wantsToEat(self, philosopher, pickLeftFork, pickRightFork, eat,
                   putLeftFork, putRightFork):
        left, right = philosopher, (philosopher + 1) % self.n
        lo, hi = min(left, right), max(left, right)
        with self.forks[lo], self.forks[hi]:
            pickLeftFork()
            pickRightFork()
            eat()
            putLeftFork()
            putRightFork()
```

Five threads, three meals each: 15 meals, and replaying the log shows no fork was ever held by two philosophers.

[leetcode.com/problems/the-dining-philosophers](https://leetcode.com/problems/the-dining-philosophers/)

## LeetCode 1114 - Print in Order

**The task.** One object has three methods, `first`, `second` and `third`. Three different threads each call one of them, and the threads may start in any order. Make sure the output is always "first", then "second", then "third".

**Input / output.** Write a class `Foo` with `first(printFirst)`, `second(printSecond)` and `third(printThird)`; each receives a function that prints its word. The test input is the order in which the three threads are started, a permutation of `[1, 2, 3]`; the output must always be `"firstsecondthird"`.

**Example.** For `[1, 3, 2]` the thread calling `first` starts first, then the one calling `third`, then the one calling `second`. `third` must wait until `second` has printed, so the output is still `"firstsecondthird"`.

**Constraints.** The input is always some ordering of `[1, 2, 3]`. The thing that matters: you cannot make the threads *start* in order, so each method has to *wait* until its predecessor is done - and waiting with a `while` loop that keeps checking a flag burns a core and, in Python, competes for the GIL with the very thread it is waiting for.

Without synchronisation the checker finds all 6 outputs possible; with two `threading.Event`s, only one:

```python
class Foo:
    def __init__(self):
        self.done1 = threading.Event()
        self.done2 = threading.Event()

    def first(self, printFirst):
        printFirst()
        self.done1.set()

    def second(self, printSecond):
        self.done1.wait()          # sleeps, without spinning, until first() is done
        printSecond()
        self.done2.set()

    def third(self, printThird):
        self.done2.wait()
        printThird()
```

Starting real threads in all 6 orders printed `firstsecondthird` every time. An `Event` is the smallest tool for "this happens after that"; `Barrier` (all wait until everyone arrives), `Condition` and `Queue` are bigger versions of the same idea.

[leetcode.com/problems/print-in-order](https://leetcode.com/problems/print-in-order/)

## asyncio: one thread, many waits

`asyncio` gets the overlap of threads without threads. Every `async def` function is a task that runs until it reaches an `await` on something slow, and hands control back to an event loop, which runs another task. One thread, no GIL contention, and thousands of waiting tasks cost almost nothing.

The catch is in "until it reaches an `await`". Ten fetches of 0.1 s each:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2043%20-%20concurrency%20and%20parallelism/imgs/day43_6.png?raw=true)

```text
await asyncio.sleep         : 0.10 s   start times [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
time.sleep inside async def : 1.00 s   start times [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
await, Semaphore(3)         : 0.40 s   start times [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3]
```

```python
async def fetch_blocking(i, delay, log, t0):
    log.append((i, 'start', time.perf_counter() - t0))
    time.sleep(delay)                   # blocks the only thread the loop has
    log.append((i, 'end', time.perf_counter() - t0))
    return i
```

The second line is the silent failure: it is still an `async def`, `asyncio.gather` still returns the ten results, nothing warns - it just runs one task at a time, ten times slower. Any blocking call does this: `requests.get`, a CPU-heavy loop, a synchronous database driver. The fix is an `await`-able version of the call, or `await asyncio.to_thread(blocking_call)` to push it onto a thread. `asyncio.Semaphore(3)` in the third line is the deliberate version of the same limit: at most three fetches at once, so 10 fetches take 4 rounds.

## LeetCode 1242 - Web Crawler Multithreaded

**The task.** Starting from one URL, find every page you can reach by following links, but only pages on the same host name as the start URL. Fetching a page is slow, so the fetches have to run in parallel.

**Input / output.** You get `startUrl` and an `HtmlParser` with one method, `getUrls(url)`, which returns the list of links on that page. Return every URL reachable from `startUrl` whose host name equals the start URL's host name, in any order, each once. All URLs use `http://`, have no port, and the host name is the part between `http://` and the next `/`.

**Example.** If `http://news.yahoo.com/` links to `http://news.yahoo.com/world` and `http://news.google.com/`, and the world page links to `http://news.yahoo.com/world/asia`, the answer is the three `news.yahoo.com` pages; the Google page is not followed.

**Constraints.** Up to 1,000 URLs, each up to 300 characters, and `startUrl` is one of them. What matters is not the size but the cost: a single-threaded crawl that fetches one page at a time is too slow for the judge, so the fetches must overlap. That makes it a waiting job - exactly where Python threads help.

My test site has 16 pages on `news.yahoo.com` and links out to `news.google.com`, `yahoo.com` and `news.yahoo.com.evil.example`; every fetch takes 50 ms. The solution crawls breadth-first in waves: all pages of the current wave are fetched at once by a thread pool.

```python
from concurrent.futures import ThreadPoolExecutor

class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser', workers=16) -> list[str]:
        hostname = lambda url: url.split('/')[2]     # the part after http://
        host, seen, frontier = hostname(startUrl), {startUrl}, [startUrl]
        with ThreadPoolExecutor(workers) as pool:
            while frontier:
                nxt = []
                for urls in pool.map(htmlParser.getUrls, frontier):
                    for v in urls:
                        if hostname(v) == host and v not in seen:
                            seen.add(v)
                            nxt.append(v)
                frontier = nxt
        return list(seen)
```

```text
serial:  16 pages, 16 fetches, 0.80 s
threads: 16 pages, 16 fetches in 5 waves, 0.26 s (= waves x 50 ms; the longest link chain is 5 pages)
`host in url` instead of ==: 17 pages, extra ['http://news.yahoo.com.evil.example/ai']
```

Look at what the time depends on. The serial crawl costs one fetch per page: 16 x 50 ms. The parallel crawl costs one fetch per *wave*: 5 x 50 ms, because page 5 of the longest chain of links cannot be fetched before page 4 has been read. More threads cannot beat that 5. That is the next section in one example. Two traps hide in the statement: `seen` is only touched by the main thread here, so it needs no lock (a version where every worker adds to `seen` itself needs one), and checking `host in url` instead of comparing host names lets `news.yahoo.com.evil.example` in.

[leetcode.com/problems/web-crawler-multithreaded](https://leetcode.com/problems/web-crawler-multithreaded/)

## Work and depth: how parallel can it be?

Draw a computation as a graph where every node is one operation and an arrow means "needs the result of". Two numbers describe it:

- **Work W**: the number of operations - the time on one worker.
- **Depth D**: the longest chain of operations that each need the previous one - the time on infinitely many workers.

With `p` workers, a scheduler that never leaves a worker idle while something is ready finishes in `T_p` steps with `max(W/p, D) ≤ T_p ≤ W/p + D` (Brent's theorem). So `W/D` is the most parallelism there is to have.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2043%20-%20concurrency%20and%20parallelism/imgs/day43_4.png?raw=true)

Summing 8 numbers with a loop is a chain: work 7, depth 7, and no number of workers helps, because every addition needs the previous total. Adding in pairs as a tree does the same 7 additions with depth 3. My `Dag` class builds these graphs, evaluates them on real numbers to prove they compute the right thing, and simulates a greedy scheduler with `p` workers.

Prefix sums are the interesting case. Day 22 needed one to turn counts into positions in counting sort, and met the two famous parallel versions there; now a scheduler can say when each one wins. Hillis-Steele: in round `d`, every position adds the value `2^d` places to its left. Blelloch: build partial sums up a tree, then pass "the sum of everything to my left" back down. For 1024 numbers:

```text
scan, loop           W=1023  D=1023  T_1=1023  T_4=1023  T_16=1023  T_64=1023  T_256=1023  T_1024=1023
scan, Hillis-Steele  W=9217  D=10    T_1=9217  T_4=2305  T_16=577  T_64=145  T_256=37  T_1024=10
scan, Blelloch       W=2036  D=18    T_1=2036  T_4=509  T_16=128  T_64=34  T_256=19  T_1024=18
Hillis-Steele beats Blelloch only from p = 545 workers
```

(Day 22 quoted 2,046 additions in 20 rounds for Blelloch. That count includes the additions of the zero that the down-sweep starts from; this graph skips them, which saves 10 additions and 2 rounds.)

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2043%20-%20concurrency%20and%20parallelism/imgs/day43_5.png?raw=true)

Hillis-Steele has the smallest depth, 10, but does nine times the work. On 4 workers it needs 2,305 steps - more than twice as slow as the plain loop's 1,023. The "parallel algorithm" loses to the serial one. Blelloch keeps the work at about two loops' worth and wins on every machine with fewer than 545 workers. Hillis-Steele only makes sense when workers outnumber the data, which is the situation inside one GPU thread block - and that is where it is actually used. Day 20's parallel merge sort made the same trade: depth from 2,046 down to 235 for 84% more work.

Amdahl's law is the special case where a fixed share `s` of the work is a chain that cannot be split: the speed-up on `p` workers is `1 / (s + (1 - s)/p)`.

```text
s = 5%: p=2: 1.9x  p=4: 3.5x  p=8: 5.9x  p=16: 9.1x  p=64: 15.4x   limit 20x
s = 10%: p=2: 1.8x  p=4: 3.1x  p=8: 4.7x  p=16: 6.4x  p=64: 8.8x   limit 10x
```

Five percent of serial work, and 64 cores give you 15x. Before buying cores, find the chain.

## Map-reduce: the reduce has to be honest

Map-reduce is the work-depth idea for data that does not fit on one machine. **Map** each document to word counts on its own (depth 1, all documents at once). **Shuffle**: send every word to reducer number `bucket(word)`. **Reduce**: each reducer adds up the counts it received. The answers from all reducers are written out one after another as the result.

On 40 documents of 16,000 words, 2 processes and 4 reducers give exactly `Counter`'s answer. Then the failures, all silent:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2043%20-%20concurrency%20and%20parallelism/imgs/day43_7.png?raw=true)

```text
two mappers that bucket words differently: 51 of 67 words wrong, e.g. "of" 821 instead of 1640
hash() buckets, fork workers: 0 words wrong
hash() buckets, spawn workers (default on Windows, macOS): 53 words wrong (this run)
latency chunks of [3, 1, 6]: mean of means 200.00, true mean 130.00
ten 0.1s: loop 0.9999999999999999, tree 1.0, math.fsum 1.0
```

**Every mapper must put a word in the same bucket.** If one mapper sends "of" to reducer 0 and another to reducer 3, each reducer counts half, and the output contains "of" twice; whichever line is read last wins. The natural code - `hash(word) % reducers` - does exactly this on Windows and macOS: Python randomises `hash()` of strings per process (the same lesson as Day 27's MinHash), and worker processes started with `spawn` each get a new random seed. Workers made with `fork` (the Linux default before Python 3.14) inherit the parent's seed, and the bug never shows. The fix is a hash that does not depend on the process, such as `hashlib.blake2b`.

**The reduce must combine pieces of different sizes correctly.** The average of three chunk averages (3, 1 and 6 values) is 200; the true mean is 130, because the single 400 ms outlier counts as a third of the answer. Reduce `(sum, count)` pairs and divide once at the end.

**Order changes floating-point results.** Adding ten 0.1s in a loop gives 0.9999999999999999; adding them as a tree gives 1.0. A parallel reduce adds in a different order on 2 workers than on 4, so the last digits change with the number of workers. Use `math.fsum` when it matters, or accept that a parallel float sum is not bit-for-bit reproducible.

The thread through all three: a reduce step can be split across workers only if the operation is associative - `(a + b) + c == a + (b + c)` - and the pieces carry everything needed to combine them. Integer `+`, `max` and `min`, set union and Day 32's HyperLogLog merge (take the max of each register) all qualify. Mean, and floating-point `+` if you need the exact same bits, do not.

## Complexity

| Pattern | Time with p workers | Notes |
|---------|---------------------|-------|
| Threads on CPU-bound Python | about the serial time | one GIL |
| Processes on CPU-bound Python | about serial / p, plus start-up and sending data | needs big tasks |
| Threads / asyncio on waits | about the longest single wait per wave | asyncio scales to thousands |
| Tree sum of n numbers | W = n - 1, D = log₂ n | |
| Hillis-Steele scan | W ≈ n log₂ n, D = log₂ n | only when p ≈ n |
| Blelloch scan | W ≈ 2n, D ≈ 2 log₂ n | the default parallel scan |
| Greedy schedule | max(W/p, D) ≤ T_p ≤ W/p + D | Brent |
| Interleaving checker | grows with the number of states (9,302 for 5 philosophers) | memoised on state |

## References

- [Global interpreter lock - Python docs glossary](https://docs.python.org/3/glossary.html#term-global-interpreter-lock) and [PEP 703 - Making the GIL optional](https://peps.python.org/pep-0703/)
- [Python docs: `threading`](https://docs.python.org/3/library/threading.html), [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) (start methods: fork, spawn, forkserver) and [`asyncio`](https://docs.python.org/3/library/asyncio.html)
- [Dining philosophers problem - Wikipedia](https://en.wikipedia.org/wiki/Dining_philosophers_problem) and [Deadlock - Wikipedia](https://en.wikipedia.org/wiki/Deadlock_(computer_science))
- [Prefix sum - Wikipedia](https://en.wikipedia.org/wiki/Prefix_sum) (Hillis-Steele and Blelloch scans)
- [Analysis of parallel algorithms - Wikipedia](https://en.wikipedia.org/wiki/Analysis_of_parallel_algorithms) (work, depth/span, Brent's theorem) and [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law)
- Jeffrey Dean and Sanjay Ghemawat, *MapReduce: Simplified Data Processing on Large Clusters*, OSDI 2004
- Guy Blelloch, *Prefix Sums and Their Applications*, 1990
