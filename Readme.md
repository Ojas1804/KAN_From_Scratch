If you're writing a KAN from scratch, think of it less as "implementing a neural network" and more as "implementing a spline function approximator for every edge."

A good coding roadmap would be:

---

## Step 1: Define the Layer Dimensions

Suppose your layer is

[
n_{in} \rightarrow n_{out}
]

Example:

[
3 \rightarrow 5
]

This means:

* 3 input neurons
* 5 output neurons
* (3 \times 5 = 15) edge functions

Unlike an MLP where you'd store

[
W \in \mathbb{R}^{3\times5}
]

you will store spline coefficients for every edge.

---

## Step 2: Create the Knot Grid

Choose:

* spline degree (p)
* number of intervals (G)

Example:

[
G=10,\quad p=3
]

Create knots:

[
t=[-1,-0.8,-0.6,\dots,1]
]

In code terms:

```text
grid_size = G
degree = p
knots = [...]
```

This grid remains fixed during training.

---

## Step 3: Determine Number of Basis Functions

For B-splines:

[
M = G + p
]

approximately.

Example:

```text
G = 10
p = 3
M = 13
```

Each edge function becomes

[
\phi(x)
=======

\sum_{m=1}^{13}
c_m B_m(x)
]

Thus each edge needs 13 trainable numbers.

---

## Step 4: Initialize Parameters

Instead of

```text
weight[i][j]
```

store

```text
coeff[i][j][m]
```

Shape:

[
(n_{in}, n_{out}, M)
]

Example:

[
(3,5,13)
]

This is the equivalent of your weight matrix.

---

## Step 5: Implement B-Spline Basis Evaluation

This is the first major component.

Given:

```text
x
knots
degree
```

compute

[
B_1(x),B_2(x),\dots,B_M(x)
]

using Cox–de Boor recursion.

Output:

```text
basis[M]
```

For a single input value:

```text
x = 0.42

basis =
[
0,
0.11,
0.54,
0.31,
0.04,
...
]
```

Only a few entries are nonzero.

This sparsity is important for efficiency.

---

## Step 6: Implement One Edge Function

Input:

```text
x
coeff[M]
basis[M]
```

Compute:

[
\phi(x)
=======

\sum_m c_m B_m(x)
]

Programmatically:

```text
edge_output =
dot(coeff, basis)
```

That's one edge.

---

## Step 7: Compute All Edge Outputs

Suppose

[
x=[x_1,x_2,x_3]
]

For every pair:

[
(i,j)
]

evaluate

[
\phi_{ij}(x_i)
]

Result:

```text
edge_outputs
shape:
[input_dim][output_dim]
```

Example:

```text
[
 [0.2, 0.7, -0.1, ...],
 [0.3, 0.1, 0.5, ...],
 [-0.4, 0.2, 0.6, ...]
]
```

---

## Step 8: Aggregate Into Neurons

For each output neuron:

[
h_j
===

\sum_i \phi_{ij}(x_i)
]

Programmatically:

```text
for each output neuron j:
    h[j] = sum(edge_outputs[:,j])
```

Equivalent to summing down the input dimension.

Result:

```text
h
shape = [output_dim]
```

---

## Step 9: Build a KAN Layer Class

Your layer should expose:

### Parameters

```text
coefficients
knots
degree
```

### Methods

```text
evaluate_basis(x)

evaluate_spline(x, coeff)

forward(x)
```

The forward pass:

```text
for each input neuron i:
    compute basis(x_i)

for each edge (i,j):
    evaluate spline

sum edge outputs

return output
```

---

## Step 10: Build a Network

Exactly like MLPs.

Example:

```text
KANLayer(3,10)

KANLayer(10,10)

KANLayer(10,1)
```

Forward:

[
x
\rightarrow
KAN_1
\rightarrow
KAN_2
\rightarrow
KAN_3
\rightarrow
\hat y
]

---

## Step 11: Autograd Handles Backprop

If using PyTorch:

You do **not** manually derive gradients.

As long as:

[
\phi(x)
=======

\sum_m c_m B_m(x)
]

is built from differentiable operations,

PyTorch computes:

[
\frac{\partial L}
{\partial c_m}
]

automatically.

The trainable parameters are simply:

```text
coefficients
```

---

## Step 12: Add Grid Extension (Advanced)

One trick in the original KAN paper:

Start with a coarse grid.

Example:

```text
5 intervals
```

After some training:

```text
10 intervals
```

Then:

```text
20 intervals
```

This gradually increases function complexity.

Implementation:

1. Train spline.
2. Insert extra knots.
3. Interpolate old coefficients.
4. Continue training.

This is called **grid refinement**.

---

## Step 13: Optimize Efficiency

A naive implementation loops:

```text
for i
    for j
        evaluate spline
```

which is slow.

Efficient implementations:

1. Compute basis for all inputs at once

[
[input_dim, M]
]

2. Store coefficients as

[
[input_dim, output_dim, M]
]

3. Use tensor contractions (einsum/matmul).

Then an entire KAN layer becomes essentially:

[
\text{output}
=============

\text{einsum}
(
\text{basis},
\text{coefficients}
)
]

without Python loops.

---

## Mental Model

When coding, think of a KAN layer as:

```text
Input
  ↓
Evaluate spline basis
  ↓
Apply edge-specific spline coefficients
  ↓
Sum contributions
  ↓
Output
```

instead of the MLP pipeline:

```text
Input
  ↓
Matrix multiplication
  ↓
Activation
  ↓
Output
```

The hardest part is not the network itself; it's implementing the B-spline basis evaluation correctly and efficiently. Once you can compute

[
B_1(x), B_2(x), \dots, B_M(x)
]

for any input (x), the rest of the KAN layer is mostly tensor bookkeeping.
