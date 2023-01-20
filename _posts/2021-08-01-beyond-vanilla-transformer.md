---
layout: post
title:  "Beyond the vanilla transformer"
description: "A short glance at promising developments of NLPs state-of-the-art architecture"
date:   2021-08-01 11:32:14 +0100
categories:
- artificial intelligence
- models
---


It is 2017, and the paper *“Attention is all you need”* provides NLP a new promising architecture: the Transformer. Four years later – an eternity in the field of deep learning – the Transformer is still a state-of-the-art architecture and is used for various NLP tasks. But the world did not stand still – at least not in this field, and many more papers have taken up the Transformer and offered new approaches for further development. I want to discuss some promising ones here.

We will start with the paper already published in 2019: *“Transformer XL: Attentive Language Models Beyond a Fixed-Length Context”*. The following paper will be *“Reformer: The Efficient Transformer”*, before the most recent article (June 2021): *“Packing: Towards 2x NLP BERT Acceleration”* will be discussed.

There are many more promising papers about the Transformer, but I had to make a selection so that this article doesn’t go beyond the scope. I hope I’m not doing an injustice to any author of a paper 🙃!

### Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
This paper deals with Transformers’ limited capacity of handling texts without a fixed length. The standard Transformer uses a maximum length, truncation, and padding to handle different sequence lengths. At the same time, this fixed length is the barrier for extended dependency learning, in which the attention mechanism grasps connections beyond this limit.

While the short-term dependency may not cause problems for most use cases, the fixed-length limit may decrease performance due to truncation and its property of not respecting semantic boundaries. The authors refer to this as context fragmentation.

Their solution sounds simple: by not computing the hidden states from scratch, the last state can serve as a memory state for the current segment. With this recurrent connection, information will not get lost but will be passed on from state to state. This obviously fights the addressed problem of losing semantic information and leads to a longer-term dependency.

The changes compared to the vanilla transformer can be found in the attention layer. The key and the value are going to be extended by the cache from the previous segment. I am not going into more details. But I can highly recommend reading the chapter Segment-Level Recurrence with State Reuse from the Transformer XL paper.

Using cache from prior segments inside the attention layer demands some additional changes. The question the authors are asking is: *“[…] how can we keep the positional information coherent when we reuse the states?”* (Dai, et al. 2019). While in the default Transformers, the sequence order is provided by the positional encoding. When caching is used, the positional encoding of the previous state (which we just talked about) is not taken into account. This leads to poor performance. To conquer this issue, the authors’ approach is to only encode the relative positional information in the hidden states and inject it into the attention score. From their perspective this approach is more intuitive:

> *For instance, when a query vector qτ,i attends on the key vectors kτ,≤i, it does not need to know the absolute position of each key vector to identify the temporal order of the segment.* (Dai, et al. 2019)

The relative distance is enough that the query vector can distinguish between elements by their different distance. Furthermore, the absolute position can be obtained from the relative distance. Please read more on this in the original paper.

To wrap it up: improving the attention mechanism by caching the previous segment seems to be a promising step. Not only that it increases the long-term dependency but also fights the problem of context fragmentation which occurs by splitting texts.

### Reformer: The Efficient Transformer

Transformers are enormous models with a lot of parameters. Training them from scratch can realistically only be done by the industry or equality big players. Luckily we can use pre-trained models like BERT. But even the 24-layer version of BERT can hardly be trained on a single GPU system. If we do not want to end up that such models can only be trained and ran by big cooperations, Transformers need to become more efficient.

Kitaev, Kaiser, and Levshaya — the authors of the *Reformer* paper, introduce three changes that should address the mentioned problem:

* Using reversible layers to store only a single copy of the activations for the whole model. This would decrease the memory needed for storing the activations by the factor of N.

* Splitting activations inside the feed-forward layers and processing them as chunks helps to decrease the memory needed by those layers. Feed-forward layers are often much more extensive than, e.g., attentions layers and are therefore responsible for high memory usage.

* Using locality-sensitive hashing attention, changing the factor inside the attention layer from *O(L²)* to *O(L log L)*

The authors are convinced that *“[…] these techniques […] have negligible impact on the training process compared to the standard Transformer”* (Kitaev, et al. 2020). I will talk about the first and the last idea — in my opinion, they are the most interesting ones. So that it does not get too far out of hand, the second approach will not be discussed any further.

#### Reversible Transformer
In their paper: *The reversible residual net-work: Backpropagation without storing activations*, Gomez et al. are presenting the Reversible residual network (RevNets). The idea is that every activation can be recovered from the following layer and does not need to be stored during backpropagation — trading off computation for more memory. The residual layer (also called skip connection) performs a function that operates on a single input and a single output:



$$
\begin{aligned}
& x \mapsto y \\
& y=x+F(x)
\end{aligned}
$$

The reversible layer works with two inputs:

$$
\left(x_{1}, x_{2}\right) \mapsto\left(y_{1}, y_{2}\right)
$$

with those equations (and reverse by subtraction):

$$
\begin{array}{ll}
y_{1}=x_{1}+F\left(x_{2}\right) & y_{2}=x_{2}+G\left(y_{1}\right) \\
x_{2}=y_{2}-G\left(y_{1}\right) & x_{1}=y_{1}-F\left(x_{2}\right)
\end{array}
$$

In the context of the Reversible Transformer, this means, that the feed-forward layer (G) and the attention layer (F) are combined inside the reversible block.

*Y₁ = X₁ + Attention(X₂) and Y₂ = X₂ + FeedForward(Y₁)*

With this implementation, there is no more need to store the activations; hence memory is saved. Even if that means slightly more computation is needed.

#### Locality-sensitive hashing attention

The heart of the Transformer is the attention mechanism. Not without reason, the initial paper is called Attention is all you need. So it was only a matter of time before approaches were developed that made this mechanism more efficient.

In the standard Transformer the scaled dot-product attention is used:

$$
{\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d}_{k}}\right) V}
$$

The input contains the queries and keys as well as the values. The dot product of all queries and keys is calculated before it gets scaled. Afterward, the softmax function is applied. And finally, the matrix multiplication of the intermediate matrix and the value matrix produces the attention score.

The authors were looking for a more memory-efficient attention mechanism and came up with the idea of *locality-sensitive hashing attention (LSH)*. The intention is to lower the complexity from *O(L²)* to *O(L log L)*. Roughly speaking, this algorithm is intended to group data points into so-called buckets so that data points close to one another are getting the same hash — with a high probability. Comparing to other hashing technics the hash collisions are maximized, not minimized. With the help of the nearest neighbors, we can focus on the keys which are close to the query.

Following you see the LSH attention formulated in the paper:

$$
o_{i}=\sum_{j \in \widetilde{\mathcal{P}}_{i}} \exp \left(q_{i} \cdot k_{j}-z\left(i, \mathcal{P}_{i}\right)\right) v_{j} \text { where } \mathcal{P}_{i}=\{j: i \geq j\}
$$

‘P’ refers to the set of which ‘i’ is a part, and ‘z’ denotes the partition function. The idea behind LSH is that set P elements are restricted to attend to elements of a single hash bucket:

$$
\boldsymbol{P}_{i}=\left\{j: h\left(q_{i}\right)=h\left(k_{j}\right)\right\}
$$

LSH as an approximation for full attention has the capability to reduce memory usage by increasing computational cost — growing with the number of hashes. The hope is that this will make large transformers more accessible — not only to institutions runnings them on multiple GPUs.

### Packing: Towards 2x NLP BERT Acceleration
As already discussed, pre-training BERT is due to its substantial need for computation power only feasible for the big industry or research facilities. Kosec, Fu, and Krell — the authors of this paper want to reduce this barrier by increasing the efficiency by removing padding. They present an algorithm to speed up pre-training by 2x, using the Wikipedia dataset.

Processing padding tokens can be seen as a waste of compute. The broader the text-length distribution, the more compute is lost in this process. The packing approach wants to fill each sequence length fully, so the need for padding vanishes. This idea is based on the assumption that sequences are interchangeable, so the order does not matter. The method of packing is a classical programming problem ([bin-packing](https://en.wikipedia.org/wiki/Bin_packing_problem) or [cutting stock problem](https://en.wikipedia.org/wiki/Cutting_stock_problem)). Since the problem is specific to NLP, two algorithms are proposed:

* Shortest-pack-first histogram-packing (SPFHP)
* Non-negative least squares histogram-packing (NNLSHP)

It is not essential to understand the details of these algorithms; it is much more important to analyze the problems caused by packing. The authors are doing a great job by addressing them in detail. Let us talk about them a bit.

As the name implies, BERT (Bidirectional Encoder Representations from Transformers) makes use of bi-directional self-attention. Packing, however — to fill up the segment and leave no space for padding — creates segments with no corresponding sequences. Language models with casual attention like GPT3 don’t face this problem since they only attend to previous tokens. The paper introduces a mask for the attention layer to prevent contamination between different sequences. Let us look at some code provided in the article:

```
mask = np.array([[1, 1, 1, 2, 2]]) # input
zero_one_mask = tf.equal(mask, mask.T) # 0, 1 mask #for use with softmax:
softmax_mask = tf.where(zero_one_mask , 0, -1000)
```

This implementation created a block-diagonal mask that reduces the padding and can be simply implemented by frameworks like numpy.

Another challenge packing is facing concerns the calculation of the loss and accuracy. In BERT the cross-entropy loss is calculated per-sequence. When packing is used, the loss would not anymore be computed for the sequence. The model would converge to a different optimum. To fix this, the authors’ idea is:

> *“To implement per-sequence loss, we effectively ‘unpack’ the incoming logits and labels by working with the per-token loss. We compute the loss on all tokens belonging to the first sequence, then all tokens belonging to the second sequence, and so on“* (Kosec, et al. 2021).

There is a nice code example in the paper that explains their implementation — check it out.

Unlike the other two articles, the packing approach is aimed at optimizing pre-training. Even though this process will probably take a lot of time and money, the improvement shown here is promising.

Now that we have looked at a variety of approaches to improving the Transformer, it has to be said that all of them are exclusive ideas for optimizing the performance and not the qualitative output. This is probably due to my selection and the fact that the Transformer as a promising architecture is increasingly only available to big players, or at least created by them. GTP3 is a paradigm example for this.

I hope these descriptions were understandable and accessible. Some concepts explained are not particularly easy to understand, so the explanation itself is not the most straightforward task. Thanks for reading till the end. See you next time — au revoir.

### References
Aidan N. Gomez, Mengye Ren, Raquel Urtasun, and Roger B. Grosse: The Reversible Residual Network: Backpropagation Without Storing Activations. [arXiv: 1707.04585](https://arxiv.org/abs/1707.04585), 2017

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin: Attention Is All You Need. [arXiv: 1706.03762](https://arxiv.org/abs/1706.03762), 2018

Matej Kosec, Sheng Fu, and Mario Michael Krell: Packing: Towards 2x NLP BERT Acceleration. [arXiv: 2107.02027](https://arxiv.org/abs/2107.02027), 2021

Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya: Reformer: The Efficient Transformer. [arXiv: 2001.04451](https://arxiv.org/abs/2001.04451), 2020

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. [arXiv: 1901.02860](https://arxiv.org/abs/1901.02860), 2019
