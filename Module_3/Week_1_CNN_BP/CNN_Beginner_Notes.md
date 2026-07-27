# CNN Notes — Cats vs Dogs Classifier (based on your notebook)

Your notebook builds a CNN in PyTorch that looks at a grayscale 128×128 image and
decides: **cat or dog?** These notes explain *why* each piece exists and *how* it
works, using your exact architecture as the running example.

---

## 1. The Big Picture

A CNN is a funnel:

```
Wide, raw input (128×128 pixels)
        ↓  Conv Layer 1  (find edges/textures)
        ↓  Conv Layer 2  (find parts: ears, eyes)
        ↓  Conv Layer 3  (find whole shapes: face outline)
        ↓  Flatten       (turn 3D feature grid into a 1D list of numbers)
        ↓  Fully Connected Layer(s)  (make the final decision)
Narrow output (2 numbers: score for "cat", score for "dog")
```

Each convolution layer doesn't look at the *whole* image at once — it slides a
small filter over patches of the image, the same way we discussed earlier
(a 3×3 or 5×5 filter is like a tiny neuron with shared weights, reused at every
position in the image).

---

## 2. The Building Blocks

### a) Convolution (`nn.Conv2d`)
Slides a small filter (kernel) across the image, multiplying and summing pixel
values under it, producing one output value per position — exactly the
"9 pixels → 1 value" idea from before, just with 5×5 = 25 pixels per patch here.

In your model:
```python
nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
```
- `in_channels=1` → grayscale image (1 number per pixel, not 3 like RGB)
- `out_channels=16` → **16 different filters**, so 16 different feature maps come out
- `kernel_size=5` → each filter looks at a 5×5 patch
- `padding=2` → adds a 2-pixel border of zeros so the output size stays the same as
  the input (128×128 in → 128×128 out), instead of shrinking

### b) Batch Normalization (`nn.BatchNorm2d`)
Rescales the outputs of the conv layer so they have a stable, consistent range.
This makes training faster and more stable — think of it as "resetting" the
data after each layer so the next layer gets clean, well-behaved numbers.

### c) ReLU Activation (`nn.ReLU`)
A simple rule: `output = max(0, input)`. Negative values become 0, positive
values pass through unchanged. This introduces *non-linearity* — without it,
stacking conv layers would mathematically collapse into just one big linear
operation, and the network couldn't learn complex patterns (curves, shapes).

### d) Max Pooling (`nn.MaxPool2d`)
Shrinks the image by keeping only the strongest signal in each small region.
```python
nn.MaxPool2d(kernel_size=4, stride=4)
```
Takes every 4×4 block and keeps just the single largest value. This:
- Reduces computation (smaller image = fewer numbers to process)
- Makes the model less sensitive to *exact* pixel position (a cat's ear shifted
  by 2 pixels still gets detected)

### e) Dropout (`nn.Dropout2d`, `nn.Dropout`)
During training, randomly "switches off" a percentage of neurons on each pass.
```python
self.dropout_conv = nn.Dropout2d(0.25)  # kills 25% of feature maps
self.dropout_fc   = nn.Dropout(0.5)     # kills 50% of neurons before final layer
```
This is a fix for **overfitting** — without it, your model memorized the
training images (97% train accuracy) but did worse on new images (88% test
accuracy). Dropout forces the network to not rely on any single neuron too
heavily, so it learns more general, robust features.

### f) Flatten
```python
out = out.view(out.size(0), -1)
```
Conv/pool layers output a 3D block of numbers (channels × height × width).
A fully-connected layer expects a flat 1D list. This line reshapes the 3D
block into a single row of numbers per image, without changing any values.

### g) Fully Connected Layers (`nn.Linear`)
```python
self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 1024 numbers in → 128 numbers out
self.fc2 = nn.Linear(128, 2)           # 128 numbers in → 2 numbers out (cat, dog)
```
This is the classic "every input connects to every output" layer — same idea
as a plain neural network, but now working on the *features* extracted by the
conv layers, not raw pixels.

### h) LogSoftmax + NLLLoss
```python
self.logsoftmax = nn.LogSoftmax(dim=1)
...
criterion = nn.NLLLoss()
```
`LogSoftmax` turns the 2 raw output numbers into log-probabilities (how
confident the model is that the image is a cat vs a dog). `NLLLoss` (Negative
Log Likelihood Loss) then measures how wrong those probabilities were compared
to the true label, and this is exactly what gets minimized during training.

---

## 3. Your Architecture, Shape by Shape

Tracking how the image's shape changes as it flows through the network
(batch size omitted for clarity):

| Stage | Operation | Output Shape |
|---|---|---|
| Input | grayscale image | `1 × 128 × 128` |
| Conv1 + BN + ReLU | `Conv2d(1→16, k=5, pad=2)` | `16 × 128 × 128` |
| MaxPool1 | `kernel=4, stride=4` | `16 × 32 × 32` |
| Conv2 + BN + ReLU | `Conv2d(16→32, k=5, pad=2)` | `32 × 32 × 32` |
| MaxPool2 | `kernel=4, stride=4` | `32 × 8 × 8` |
| Conv3 + BN + ReLU | `Conv2d(32→64, k=5, pad=2)` | `64 × 8 × 8` |
| MaxPool3 | `kernel=2, stride=2` | `64 × 4 × 4` |
| Flatten | `view(-1)` | `1024` (= 64×4×4) |
| FC1 | `Linear(1024 → 128)` | `128` |
| FC2 | `Linear(128 → 2)` | `2` |
| LogSoftmax | probabilities | `2` (cat score, dog score) |

Notice: `padding=2` with `kernel_size=5` keeps the width/height unchanged after
each conv (128→128, 32→32, 8→8) — all the *shrinking* happens in the
MaxPool steps (128→32→8→4). This is a deliberate design: convolutions extract
features without losing information, pooling does the compression.

---

## 4. Data: Loading and Augmentation

```python
transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```
- **Resize** → every image forced to the same 128×128 size (networks need
  fixed-size input)
- **Grayscale** → drops color to 1 channel, simplifying the problem
- **ToTensor** → converts the image (pixel values 0–255) into a PyTorch tensor
  with values 0–1
- **Normalize(0.5, 0.5)** → rescales values from [0,1] to [-1,1], which
  generally helps training converge faster

For the **training set only**, extra random transforms are added:
```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)
```
This is **data augmentation**: it artificially creates variety (a flipped or
slightly rotated cat is still a cat) so the model doesn't just memorize the
exact pixels it was shown, but learns the underlying pattern. This directly
fights overfitting, alongside dropout.

The `DataLoader` then handles shuffling and batching (grouping 100 images at a
time — `batch_size=100`) so training happens efficiently on chunks of data
rather than one image at a time.

---

## 5. The Training Loop, Step by Step

```python
for images, labels in train_loader:
    optimizer.zero_grad()          # 1. clear old gradients
    outputs = model(images)        # 2. forward pass: get predictions
    loss = criterion(outputs, labels)  # 3. measure how wrong the predictions were
    loss.backward()                # 4. backward pass: compute gradients
    optimizer.step()               # 5. update the weights
```

1. **Zero gradients** — PyTorch accumulates gradients by default, so they must
   be reset before each new batch, or old and new gradients would mix.
2. **Forward pass** — the image flows through every layer defined in
   `__init__`/`forward`, producing 2 output scores.
3. **Loss** — `NLLLoss` compares the predicted scores to the true label
   (0=cat, 1=dog) and outputs a single number representing the error.
4. **Backward pass** — PyTorch's autograd automatically computes how much each
   weight in the network contributed to the error (the gradient).
5. **Optimizer step** — `Adam` (an improved version of gradient descent)
   nudges every weight slightly in the direction that reduces the error.

This repeats over the whole dataset (one full pass = one **epoch**), and the
notebook runs up to `num_epochs = 35`, stopping early if training accuracy
reaches 90%.

---

## 6. Evaluation

```python
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
```
- `model.eval()` switches off dropout and freezes batch norm statistics — you
  want the *full, stable* network when testing, not the randomly-thinned
  version used during training.
- `torch.no_grad()` tells PyTorch not to bother tracking gradients here,
  since we're not training — this saves memory and computation.
- `torch.max(outputs, 1)` picks whichever of the 2 output scores (cat/dog) is
  higher — that's the model's final prediction.

Accuracy is then just: `correct predictions ÷ total images`.

---

## 7. Why These Specific Design Choices Matter

| Symptom (without the fix) | Fix used in your notebook | Why it works |
|---|---|---|
| Model memorizes training images (train 97%, test 88%) | Dropout (0.25 / 0.5) | Forces network to not depend on single neurons |
| Model only ever sees the exact same images | RandomHorizontalFlip, RandomRotation | Creates realistic variety, improves generalization |
| Training numbers drift to unstable ranges | BatchNorm after each conv | Keeps activations in a stable, consistent range |
| Only linear operations, can't learn curves/shapes | ReLU | Introduces non-linearity |
| Huge, slow-to-process feature maps | MaxPool | Shrinks the data while keeping the strongest signals |

---

## 8. Quick Glossary

- **Feature map** — the output of one filter; a 2D grid showing where that
  filter "activated" (e.g., detected an edge)
- **Channel** — one feature map; `out_channels=16` means 16 channels come out
  of that conv layer
- **Epoch** — one full pass through the entire training dataset
- **Batch** — a small group of images processed together (100 here) before
  updating weights, instead of one-by-one
- **Overfitting** — the model does well on training data but poorly on new,
  unseen data
- **Weight sharing** — the same filter (same 25 weights, for a 5×5 kernel) is
  reused at every position in the image, and across every image in the
  dataset — this is *the* defining property of convolution

---

## 9. One-Sentence Summary

Your CNN is three stacked "pattern detectors" (conv + batchnorm + relu + pool)
that progressively shrink a 128×128 image down to 64 abstract 4×4 feature
maps, flattens those into 1024 numbers, and feeds them through a small
2-layer decision-maker (fully connected layers) that outputs a cat-vs-dog
probability — trained end-to-end with dropout and data augmentation to avoid
memorizing the training set.
