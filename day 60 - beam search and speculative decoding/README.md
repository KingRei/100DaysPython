# Beam search and speculative decoding: searching for a better sentence, and getting the same one faster

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates the gap between
the best token and the best sentence, a beam keeping several prefixes alive, the two
things beams get wrong, one round of draft-and-verify, the proof that the output
distribution does not move, and the two knobs that control the speedup.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2060%20-%20beam%20search%20and%20speculative%20decoding/imgs/day60_1.png?raw=true)

The previous day ended on a measurement: greedy decoding produces the **44th** most
likely sentence out of the 1,250 this grammar admits. Picking the best character at
every step is not picking the best sentence, and no temperature or top-p setting
closes that gap, because it is a search problem rather than a sampling one.

Beam search is the search. Speculative decoding is the opposite kind of idea: leave
the output distribution untouched and spend a cheap model's guesses to produce it with
fewer calls to the expensive one. Both are checked here against an enumeration of
every legal sentence, so "did the search find the best sentence" and "did the speedup
change the answers" have real answers.

## Beam search

| width | rank found | log p |
|---|---|---|
| 1 (greedy) | 44 | -6.717 |
| 2 | 44 | -6.717 |
| 4 | 10 | -6.502 |
| 8, 16, 32 | 5 | -6.407 |

Width 1 is greedy - and reproducing that exactly requires a stable sort, because this
model has genuine ties. Widening climbs from rank 44 to rank 5 and then stops: the
best sentence's prefix was pruned early, so no width reaches rank 1. A beam is a
heuristic, not a maximiser. The cost is linear in the width: w sequences, w KV caches
and w times the attention work at every step.

## The two things beams get wrong

**Wider is not monotonically better.** On a model trained on a smaller corpus, a
width-3 beam scores *worse* than width 2. Keeping more prefixes alive lets a mediocre
early prefix survive and crowd out the one that would have won - the effect
translation papers call the beam search curse, where past a width of 5 to 10 the BLEU
score falls while model probability keeps rising.

**Highest total probability means shortest.** Every character multiplies the score by
something below 1. Here the bias is mild (correlation 0.061 over sentences spanning 21
to 38 characters), but visible: the top 10 by total log probability average 28.8
characters against 36.5 for the top 10 by log probability per character. Dividing by
`length ** alpha` is the standard patch, and a pure heuristic.

**And a beam returns neighbours, not options.** Its 16 best sentences share only 8
distinct subjects, against 12.8 for 16 sampled sentences. That is not a bug: the most
probable outputs resemble each other by construction, which is why beam search belongs
where one answer is correct (translation, speech recognition, constrained extraction)
and chat models use the previous day's sampling.

## Speculative decoding

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2060%20-%20beam%20search%20and%20speculative%20decoding/imgs/day60_2.png?raw=true)

A cheap draft proposes `gamma` tokens; the expensive model verifies all of them in
**one** forward pass, because scoring gamma positions is one pass while generating them
would have been gamma. Each proposal is accepted with probability `min(1, p/q)`, the
first rejection is replaced by a draw from the normalised positive part of `p - q`, and
a round where everything is accepted emits a free bonus token.

Adding the two routes gives back `p` exactly - the measured error is 1e-16, which is
floating point, not approximation. Two shortcuts that sound equivalent are not: trusting
the draft moves the distribution by up to 0.367, and dropping the coin flip while
accepting whenever `p >= q` moves it by up to 0.043. Neither raises an error.

| draft order | accept rate | tokens / round | grammatical |
|---|---|---|---|
| 1 | 0.173 | 1.69 | 93% |
| 2 | 0.605 | 3.42 | 93% |
| 3 | 0.837 | 4.35 | 90% |
| 5 | 0.929 | 4.72 | 88% |

The quality column is the point: it does not move. A bad draft costs speed, never
correctness.

| gamma | accept | measured tokens/round | iid theory | speedup |
|---|---|---|---|---|
| 1 | 0.911 | 1.91 | 1.91 | 1.74x |
| 2 | 0.910 | 2.82 | 2.74 | 2.35x |
| 4 | 0.842 | 4.37 | 3.65 | 3.12x |
| 8 | 0.760 | 7.08 | 3.82 | 3.94x |

The textbook formula `(1 - a^(gamma+1)) / (1 - a)` assumes proposals are accepted
independently. They are not: the draft is right through a whole predictable stretch and
wrong at branch points, so long runs of acceptance are far more common than a coin with
the same average gives. That is why deployments report accepted *length* rather than
accept rate.

`decoding.py` closes with the matching sections for vLLM (beam search moved out of
`SamplingParams` into `LLM.beam_search()`, `--speculative-config`, and the `ngram`
method that needs no draft model at all) and SGLang (`--speculative-algorithm EAGLE3`,
tree drafting via `--speculative-eagle-topk`, `NEXTN` for models with their own MTP
head, and why accepted length has to be read against batch size).

## Complexity

| Operation | Time | Space |
|---|---|---|
| greedy | O(L · V) | O(1) sequences |
| beam search, width w | O(L · w · V) | O(w) sequences of KV cache |
| exact best sentence | O(number of sequences) | - |
| speculative round | 1 target pass + gamma draft passes | O(gamma) |

## References
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [On NMT Search Errors and Model Errors: Cat Got Your Tongue?](https://arxiv.org/abs/1908.10090)
