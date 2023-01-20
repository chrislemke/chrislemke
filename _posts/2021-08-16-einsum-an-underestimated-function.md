---
layout: post
title:  "einsum - an underestimated function"
description: "One function — many possibilities. How to use linear algebra for deep learning in a clear and simple way"
date:   2021-08-16 21:32:09 +0100
categories:
- mathematics
- linear algebra
---


Linear algebra plays a fundamental role in the field of deep learning. It is always about shapes, transpose, etc. Libraries like PyTorch, Numpy, and Tensorflow offer a lot of functions for this. But you may forget one or the other or confuse a function with one from another library.

Even though Albert Einstein certainly did not had this problem, he helps us with the so-called Einstein notation. The idea is simple: The sum characters are omitted to improve the overview, and instead, over twice occurring indices are summed.

$$
(A \cdot B)_{i j}=\sum_{k=1}^{n} A_{i k} \cdot B_{k j}
$$

Turns into:

$$
(A \cdot B)_{i j}=A_{i k} \cdot B_{k j}
$$

Thanks, Albert 🙏

With the *Einstein notation* and the einsum function, we can calculate with vectors and matrixes using only a single function: *torch.einsum(equation, *operands)*. I will use [Pytorch’s einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html?highlight=einsum#torch.einsum) function in the upcoming code, but you may use [NumPy’s](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) or the one from [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/einsum) — they are interchangeable. We will see different usages of einsum, together with the native PyTorch function.

Most important for this einsum magic to understand is the concept of the indices. There are two types of them:

* Free indices — specified in the output
* Summation indices — all other

Let’s check out a short example:

```
torch.einsum(‘ik, kj->ij’, X, Y)
```

Probably you already understand what is happing here: it is matrix multiplication. i and j are the so-called free indices, and k is a summation index. The latter can be defined as that index where the summation happens. If we image the matrix multiplication as nested loops, i and j would be the outer loops, and the k-loop would be the summation-loop:

<script src="https://gist.github.com/chrislemke/aa094ae29d0c7690c3db2dda88cb8420.js"></script>
Quite simple, right? So let’s get started!

### Permutation
This may be used for other things, but transposing a vector or a matrix seems to be the most famous use case.

<script src="https://gist.github.com/chrislemke/ac599b9bc5d29cda68387a06e6709bfd.js"></script>
We simply have to switch the identifiers — et voila. Simple, even if the X.T is also an exquisite solution 😉.

### Summation

<script src="https://gist.github.com/chrislemke/7e136df8522ac4e097dbf20ed96f439e.js"></script>
In this case — the simple summation, we do not return an index. The output is a scalar. Or, to be precise, a tensor with only one value.

### Row and column summation
<script src="https://gist.github.com/chrislemke/0ef6fd55c22b2faa17ee545be307d61b.js"></script>
One index makes the difference — summing up by rows or columns.

### Element wise multiplication
Pytorch’s implementation is super simple — just using the multiplication operator (*). What does it look like with einsum?

<script src="https://gist.github.com/chrislemke/0221d509986cc0b38fab48b2a22af563.js"></script>
Here the indices are always arranged equally. i, j multiplied by i, j gives a new matrix with the same shape.

### Dot product
Probably one of the better-known operations. Also called scalar product. As the name suggests, it returns a scalar.

<script src="https://gist.github.com/chrislemke/825b063f2354bdb5bd73e8731676fb49.js"></script>
The *einsum* function does not have an output index, which implies that it returns a scalar.


### Outer product
The outer product of two coordinate vectors is a matrix.

<script src="https://gist.github.com/chrislemke/daf0137b4c290c8bb9c23a5bffe1773f.js"></script>

### Matrix-Vector multiplication
To multiply a matrix by a vector, the matrix must have as many columns as the vector has rows.

<script src="https://gist.github.com/chrislemke/7ec0319b1cc28effcb1fc4c58367f52e.js"></script>

This is a good example of how the einsum function is handling two operations: transposing y and multiplication.

### Matrix-Matrix multiplication
One of the most important calculations in deep learning is matrix multiplication. But also in other fields of machine learning, this function is often used.

<script src="https://gist.github.com/chrislemke/4e1b1a7f961f6ec1f67034e44645eb1f.js"></script>

### Batch matrix multiplication
Last but not least, let’s have a look at batch matrix multiplication. Even if Pytorch’s implementation is concise and straightforward, it is nice to have one function for all linear algebra computations.

<script src="https://gist.github.com/chrislemke/401aabf56c51811ced19a0195a665e01.js"></script>

I hope those few examples made einsum a bit more clearer. There is so much more about it (e.g., broadcasting). But for now, this should be it. And then there is also [einops](https://github.com/arogozhnikov/einops). A whole library full of tensor operations — check it out. See you next time — au revoir.
