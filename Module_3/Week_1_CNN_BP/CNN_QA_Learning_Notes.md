# CNN Learning Notes — Q&A Walkthrough

These notes compile everything covered across our conversation — from the basic
convolution operation up to batching, epochs, and the math behind dense layers.

---

## 1. The Convolution Operation — Core Idea

**Setup**: 7×7 input, 3×3 filter, no padding, stride 1.

- Input has 49 pixels; the filter has 9 weights.
- The filter slides over the image, and at each position it takes the 9 pixels
  underneath, multiplies them by the 9 weights, and sums into **1 output value**.

**Output size formula**:
$$
\text{Output size} = \frac{N - F + 2P}{S} + 1
$$
where `N` = input size, `F` = filter size, `P` = padding, `S` = stride.

For a 7×7 input, 3×3 filter, no padding, stride 1:
$$
\frac{7 - 3 + 0}{1} + 1 = 5 \rightarrow \text{5×5 output (25 values)}
$$

**Why 5×5, intuitively**: the filter's top-left corner can only start at rows/columns
0 through 4 (5 valid positions) before the 3×3 window would fall off the image edge.

---

## 2. Filter as a "Neuron"

Each 3×3 (or 5×5, etc.) patch acts like **one input sample** to a tiny neuron:

$$
z = w_1x_1 + w_2x_2 + \dots + w_9x_9 + b, \qquad \text{output} = \text{activation}(z)
$$

- `x_1...x_9` = the 9 pixels in that patch (flattened)
- `w_1...w_9` = the 9 filter weights — **the same weights reused at every patch**
- This weight sharing is exactly what makes it a "convolution" instead of 25
  independent neurons — it lets the network detect the same pattern (edge, curve)
  no matter where it appears in the image.

---

## 3. Weight Sharing — Within an Image and Across Images

| Sharing type | Shared? |
|---|---|
| Same filter, across patches (within one image) | ✅ Yes |
| Same filter, across different images in the dataset | ✅ Yes |
| Different filters (filter 1 vs filter 2) | ❌ No — independent weights |

- **Within one image**: the filter's 9 (or 25) weights are reused at every sliding
  position.
- **Across images**: image #2 uses the *exact same* filter weights as image #1.
  The network doesn't learn a new filter per image — a vertical-edge detector
  looks the same whether it's scanning image 1 or image 2.
- Weights only change **during training**, when backpropagation updates them
  based on the loss computed across a batch. Between training steps, the same
  weights are used on every image.

---

## 4. Multiple Filters

Each filter is **independent** — its own separate set of weights + bias — and
learns to detect a **different pattern**.

```
Input image (7x7)
   ├── Filter 1 (3x3, weights set A) → Feature map 1 (5x5)
   └── Filter 2 (3x3, weights set B) → Feature map 2 (5x5)
```

Output shape becomes `5 × 5 × 2` — that `2` is the **depth** (number of channels)
of the output. More filters = more distinct patterns detected in parallel
(e.g., one filter for vertical edges, one for horizontal edges, etc.).

---

## 5. General CNN Architecture

A typical CNN repeats a block, then finishes with dense layers:

```
Input Image
   → [Conv → (BatchNorm) → ReLU → Pool]   ← Block 1 (edges)
   → [Conv → (BatchNorm) → ReLU → Pool]   ← Block 2 (parts)
   → [Conv → (BatchNorm) → ReLU → Pool]   ← Block 3 (whole shapes)
   → Flatten
   → Fully Connected layer(s)
   → Output (class scores / probabilities)
```

As you go deeper:
- **Channels increase** (e.g. 16 → 32 → 64) — more filters, more patterns captured.
- **Spatial size shrinks** (e.g. 128×128 → 32×32 → 8×8) — pooling compresses it.
- **Abstraction increases** — edges → parts → whole objects.

**Supporting pieces**:
- **BatchNorm** — rescales activations between layers for faster, more stable training.
- **Dropout** — randomly disables neurons during training to prevent overfitting.
- **Stride** — how far the filter jumps each step (1 = check every position).
- **Padding** — a border of zeros so edge pixels are processed and/or output size
  is preserved.
- **Softmax / LogSoftmax** — turns final raw scores into probabilities summing to 1.

Only the filter weights and dense layer weights are actually learned via
backpropagation — nobody hand-designs the filters.

---

## 6. Concrete Example: 32×32 RGB Image, 10 Classes

| Layer | Type | Filters/Units | Kernel | Stride | Padding | Activation | Pooling | Output Shape |
|---|---|---|---|---|---|---|---|---|
| Input | — | — | — | — | — | — | — | 3 × 32 × 32 |
| Conv1 | Convolution | 16 | 3×3 | 1 | 1 | ReLU | — | 16 × 32 × 32 |
| Pool1 | MaxPool | — | 2×2 | 2 | 0 | — | Max | 16 × 16 × 16 |
| Conv2 | Convolution | 32 | 3×3 | 1 | 1 | ReLU | — | 32 × 16 × 16 |
| Pool2 | MaxPool | — | 2×2 | 2 | 0 | — | Max | 32 × 8 × 8 |
| Conv3 | Convolution | 64 | 3×3 | 1 | 1 | ReLU | — | 64 × 8 × 8 |
| Pool3 | MaxPool | — | 2×2 | 2 | 0 | — | Max | 64 × 4 × 4 |
| Flatten | — | — | — | — | — | — | — | 1024 |
| Dense1 | Fully connected | 128 | — | — | — | ReLU | — | 128 |
| Dense2 (output) | Fully connected | 10 | — | — | — | Softmax | — | 10 |

**Padding=1, kernel=3** keeps size unchanged: `(32-3+2)/1+1 = 32` — the
standard "same padding" trick.
**Pool 2×2, stride 2** always halves size: `(32-2+0)/2+1 = 16`.
**Flatten**: `64 × 4 × 4 = 1024`, which becomes Dense1's input size.

---

## 7. Dense (Fully Connected) Layers — No "Filters"

Dense layers don't have filters — they have **neurons/units**, each connected
to *every* input value (no sliding).

| | Convolution | Dense (fully connected) |
|---|---|---|
| Building block | Filter (kernel) | Neuron / unit |
| How it scans | Slides, same weights reused everywhere | No sliding — every neuron connects to every input once |
| Weights per unit | e.g. 3×3×channels | Equal to number of inputs |

**Example: `nn.Linear(1024, 128)`**
- Weight matrix: `128 × 1024` (one row per output neuron)
- Bias vector: `128`
- Total parameters: `128 × 1024 + 128 = 131,200`

**Per-neuron computation** (neuron #1 of 128):
$$
y_1 = \text{ReLU}\left(\sum_{i=1}^{1024} w_{1,i} \cdot x_i + b_1\right)
$$
Each of the 128 neurons has its *own* separate set of 1024 weights — this is
why dense layers often have far more parameters than conv layers, despite
looking "smaller" on paper (no weight sharing to keep the count down).

---

## 8. Worked LeNet-Style Example (from real notebook code)

```python
nn.Conv2d(1, 6, kernel_size=5),   # Conv1
nn.MaxPool2d(kernel_size=2),      # Pool1
nn.ReLU(),
nn.Conv2d(6, 16, kernel_size=5),  # Conv2
nn.MaxPool2d(kernel_size=2),      # Pool2
nn.ReLU()
```

| Layer | Operation | Filters | Kernel | Stride | Padding | Output Shape |
|---|---|---|---|---|---|---|
| Input | — | — | — | — | — | 1 × 28 × 28 |
| Conv1 | Convolution | 6 | 5×5 | 1 | 0 | 6 × 24 × 24 |
| Pool1 | MaxPool | — | 2×2 | 2 | 0 | 6 × 12 × 12 |
| ReLU1 | Activation | — | — | — | — | 6 × 12 × 12 |
| Conv2 | Convolution | 16 | 5×5 | 1 | 0 | 16 × 8 × 8 |
| Pool2 | MaxPool | — | 2×2 | 2 | 0 | 16 × 4 × 4 |
| ReLU2 | Activation | — | — | — | — | 16 × 4 × 4 |

### Why channels go 1 → 6 → 16

- **Conv1**: `in_channels=1` (grayscale image), `out_channels=6` (6 independently
  learned filters → 6 stacked feature maps).
- **Conv2**: `in_channels=6` — **must match** Conv1's `out_channels`, since Conv2
  receives Conv1's output. Each of Conv2's filters is really shaped `5×5×6`
  (it looks across all 6 incoming channels at once). `out_channels=16` is a
  fresh design choice for this layer.

**The rule**:
```
Conv2d(in_channels,  out_channels, ...)
              ↑              ↓
     must match       becomes in_channels
     previous layer's   of the NEXT conv layer
     out_channels
```

### Missing ReLU between Linear layers — why it matters

```python
nn.Linear(16 * 4 * 4, 120),   # Linear1
nn.Linear(120, 84),            # Linear2  ← no ReLU before this!
nn.ReLU(),
nn.Linear(84, 10)
```

Two back-to-back `nn.Linear` layers with nothing non-linear between them
collapse mathematically into **one** linear layer:
$$
y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'
$$
So Linear1 → Linear2 (no activation between) has the same expressive power as
a single `Linear(256, 84)` — extra parameters and computation, but no extra
*capability* to learn non-linear patterns. **Rule of thumb**: put an activation
after every hidden linear layer; only the final output layer typically skips it.

---

## 9. Batching — How 5 Samples Flow Through the Network

**Core rule**: batch size is just a new leading dimension. The *same weights*
apply to each sample independently — 5 parallel forward passes, not 5 different
networks.

```
shape = (batch, channels, height, width)
```

| Layer | Per-sample math | Output shape (batch=5) |
|---|---|---|
| Input | — | (5, 1, 28, 28) |
| Conv1 | (28-5)/1+1 = 24 | (5, 6, 24, 24) |
| Pool1 | (24-2)/2+1 = 12 | (5, 6, 12, 12) |
| ReLU1 | shape unchanged | (5, 6, 12, 12) |
| Conv2 | (12-5)/1+1 = 8 | (5, 16, 8, 8) |
| Pool2 | (8-2)/2+1 = 4 | (5, 16, 4, 4) |
| ReLU2 | shape unchanged | (5, 16, 4, 4) |
| Flatten | 16×4×4 = 256 | (5, 256) |
| Linear1 | matrix multiply | (5, 120) |
| Linear2 | matrix multiply | (5, 84) |
| ReLU3 | shape unchanged | (5, 84) |
| Linear3 | matrix multiply | (5, 10) |

**Verified in code** — total learnable parameters stayed **44,426** regardless
of whether batch size was 1, 5, or 32. Parameters never multiply by batch size.

**Linear1 math with batch=5**:
$$
Y = XW^T + b
$$
- `X`: (5, 256) — 5 rows, one per sample
- `W`: (120, 256) — fixed, same matrix used for all 5 samples
- `Wᵀ`: (256, 120)
- `Y = X @ Wᵀ + b`: (5, 120)

Under the hood, this is 5 independent `(1×256)·(256×120)` multiplications,
batched into one `(5×256)·(256×120)` matrix multiply for speed.

| Thing | Changes with batch size? |
|---|---|
| Weights / filters / biases | ❌ No — always fixed |
| Number of parallel forward passes | ✅ Yes |
| Output tensor's first dimension | ✅ Yes — always equals batch size |
| Loss value | ✅ Usually averaged over the batch |
| Gradient per weight | ✅ Averaged/summed across the batch before the weight update |

---

## 10. Epoch vs Batch

| | Batch | Epoch |
|---|---|---|
| What it is | A small group of samples processed together before one weight update | One complete pass through the entire training dataset |
| What happens | 1 forward + 1 backward + 1 optimizer step | Many batches, until all data is seen once |
| Controls | `batch_size` | `num_epochs` |

**Worked example** — 2,000 training images, `batch_size=100`:
$$
\text{batches per epoch} = \frac{2000}{100} = 20
$$
So one epoch = 20 batches. With `num_epochs=35`, the network sees the full
dataset 35 times, and weights get updated `35 × 20 = 700` times total.

**Analogy**: batch = one small stack of flashcards reviewed before pausing to
update what you've learned; epoch = going through the *entire* flashcard deck
once. `num_epochs=35` means going through the whole deck 35 times.

**Why both exist**:
- Batches → more frequent weight updates than waiting for the whole dataset,
  and manageable memory usage.
- Epochs → a single pass usually isn't enough; repeated passes let the network
  refine its weights until the loss stops improving.

---

## 11. Linear Layer Weight Matrix Shape — Math vs Hyperparameter

For `nn.Linear(in_features, out_features)`:
$$
\text{weight.shape} = (\text{out\_features}, \text{in\_features}), \qquad
\text{bias.shape} = (\text{out\_features},)
$$

So `nn.Linear(256, 120)` → weight shape `(120, 256)`:
- Row `i` = neuron `i`'s own 256 weights (one per input feature).
- Verified in code: `linear1.weight.shape == torch.Size([120, 256])`.

**Why the transpose in the math**:
$$
Y = XW^T + b
$$
| Tensor | Shape | Meaning |
|---|---|---|
| X (input) | (5, 256) | 5 samples, 256 features each |
| W (stored) | (120, 256) | 120 neurons, 256 weights each |
| Wᵀ | (256, 120) | flipped so matmul works |
| X @ Wᵀ | (5, 120) | 5 samples, 120 outputs each |

PyTorch stores weights as `(out_features, in_features)` so each neuron's
weights sit together as one contiguous row — convenient for indexing/init —
and the transpose happens internally during the forward computation.

### Which numbers are math-fixed vs your choice?

| Number | Fixed by math or hyperparameter? |
|---|---|
| `in_features` (e.g. 256) | **Fixed by math** — must exactly match whatever the previous layer (e.g. Flatten) produced. Setting it to anything else throws a shape-mismatch error. |
| `out_features` (e.g. 120) | **Hyperparameter** — your design choice. No formula derives it from `in_features`. |

**Guides for choosing `out_features`** (not strict math, just heuristics):
1. **Funnel shape** — gradually shrink toward the number of output classes
   (e.g. 256 → 120 → 84 → 10).
2. **Rule-of-thumb ranges** — often a fraction of `in_features` (1/2, 1/4, 1/8).
3. **Common conventions** — powers of 2 (64, 128, 256) for GPU efficiency,
   though LeNet's 120/84 show it's still flexible.
4. **Experimentation / tuning** — try a few sizes, keep whichever gives the
   best validation accuracy.
5. **Trade-off**:

| Larger hidden size | Smaller hidden size |
|---|---|
| More capacity, more parameters, slower | Less capacity, faster, less overfitting risk |

### Output shape — which part is math, which is hyperparameter

Output shape `(5, 120)` has two numbers, each fixed a different way:

| Position | Value | Fixed by math or hyperparameter? |
|---|---|---|
| 1st (batch) | 5 | Fixed by *your input tensor's shape* — set by `DataLoader`'s `batch_size`, not by the layer itself |
| 2nd (out_features) | 120 | Fixed by the layer's weight matrix, which *you* chose when defining `nn.Linear(256, 120)` |

**Verified in code**: the same layer (same weights, shape `(120,256)`) produces
output `(1,120)`, `(5,120)`, or `(32,120)` depending purely on how many samples
were in the input batch — the layer's own definition never changes.

**Bottom line**: both numbers in any layer's output shape trace back to a
choice you made — batch size when building your `DataLoader`, and
`out_features` when defining the layer — matrix multiplication just mechanically
enforces them during the forward pass.

---

## 12. Quick Reference — Key Formulas

$$
\text{Conv/Pool output size} = \frac{N - K + 2P}{S} + 1
$$

$$
\text{Linear weight shape} = (\text{out\_features}, \text{in\_features})
$$

$$
Y = XW^T + b \quad \text{(Linear layer forward pass)}
$$

$$
\text{batches per epoch} = \frac{\text{total training samples}}{\text{batch size}}
$$
