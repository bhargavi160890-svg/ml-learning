# Keras Notes — Architecture, Usage, and Comparison to PyTorch

Since you already know PyTorch, these notes are written side-by-side with what
you've learned so far — same LeNet-style network, same concepts, just showing
how Keras expresses them differently.

---

## 1. What is Keras?

Keras is a **high-level deep learning API**. It's not a competing engine to
PyTorch in the same sense — it's a friendlier interface that sits on top of a
lower-level engine (called a "backend"), which does the actual tensor math.

- Historically, Keras ran on top of TensorFlow, Theano, or CNTK.
- Today, Keras (v3) is backend-agnostic — it can run on **TensorFlow, PyTorch,
  or JAX** underneath.
- Most commonly, when people say "Keras," they mean `tf.keras` — Keras bundled
  directly inside TensorFlow, which is the version these notes focus on.

**Analogy**: if PyTorch is like driving a manual-transmission car (more control,
more steps), Keras is like an automatic — fewer decisions, faster to get moving,
but slightly less direct control under the hood.

---

## 2. Keras's Three Ways to Build a Model (its "architecture")

### a) Sequential API — simplest, for straight-line models

Layers stacked one after another, no branching. This maps almost directly to
what you've been doing with `nn.Sequential` in PyTorch.

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(6, kernel_size=5, activation=None, input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=2),
    layers.ReLU(),
    layers.Conv2D(16, kernel_size=5, activation=None),
    layers.MaxPooling2D(pool_size=2),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(120),
    layers.Dense(84),
    layers.ReLU(),
    layers.Dense(10, activation='softmax'),
])
```

### b) Functional API — for models with branching, multiple inputs/outputs

Used when a model isn't a simple straight line — e.g. skip connections
(ResNet-style), multiple inputs, or multiple outputs.

```python
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(6, kernel_size=5)(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.ReLU()(x)
x = layers.Conv2D(16, kernel_size=5)(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.ReLU()(x)
x = layers.Flatten()(x)
x = layers.Dense(120)(x)
x = layers.Dense(84)(x)
x = layers.ReLU()(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)
```

### c) Model Subclassing — most flexible, closest to PyTorch's style

You define layers in `__init__` and the forward pass in `call()` — this is the
Keras equivalent of PyTorch's `__init__` + `forward()` pattern, and will feel
the most familiar to you.

```python
class LeNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(6, kernel_size=5)
        self.pool1 = layers.MaxPooling2D(pool_size=2)
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(16, kernel_size=5)
        self.pool2 = layers.MaxPooling2D(pool_size=2)
        self.relu2 = layers.ReLU()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120)
        self.fc2 = layers.Dense(84)
        self.relu3 = layers.ReLU()
        self.fc3 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(self.fc2(x))
        return self.fc3(x)

model = LeNet()
```

---

## 3. Layer-by-Layer: PyTorch → Keras Translation

| Concept | PyTorch | Keras |
|---|---|---|
| Convolution | `nn.Conv2d(in_ch, out_ch, kernel_size)` | `layers.Conv2D(out_ch, kernel_size)` |
| Max pooling | `nn.MaxPool2d(kernel_size)` | `layers.MaxPooling2D(pool_size)` |
| Activation | `nn.ReLU()` | `layers.ReLU()` or `activation='relu'` inside a layer |
| Batch norm | `nn.BatchNorm2d(num_features)` | `layers.BatchNormalization()` |
| Dropout | `nn.Dropout(p)` | `layers.Dropout(rate)` |
| Flatten | `x.view(x.size(0), -1)` | `layers.Flatten()` |
| Fully connected | `nn.Linear(in_f, out_f)` | `layers.Dense(out_f)` |
| Softmax | `nn.LogSoftmax(dim=1)` + `NLLLoss` | `activation='softmax'` + `categorical_crossentropy` |

**One important difference**: Keras layers **don't need `in_channels` /
`in_features` specified** (except sometimes the very first layer's
`input_shape`). Keras automatically infers the input size the first time data
flows through — you only ever specify the **output** size. This removes the
"must match previous layer's out_channels" bookkeeping you had to do manually
in PyTorch.

---

## 4. Training: Keras's Biggest Structural Difference

This is where Keras diverges from PyTorch the most. Recall your PyTorch loop:

```python
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

In Keras, this entire loop is replaced by **two method calls**:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_images, train_labels,
    batch_size=100,
    epochs=35,
    validation_data=(test_images, test_labels)
)
```

- `model.compile()` sets up the optimizer, loss function, and metrics once.
- `model.fit()` handles the *entire* training loop internally — batching,
  forward pass, loss computation, backward pass, and weight updates — across
  all epochs, automatically.

Evaluation is similarly condensed:

```python
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
predictions = model.predict(new_images)
```

No manual `model.eval()` / `torch.no_grad()` — Keras handles the
train-vs-inference mode switch internally based on which method you call.

---

## 5. Advantages of Keras over PyTorch

| Advantage | Why it matters |
|---|---|
| **Less boilerplate** | `model.fit()` replaces ~10 lines of manual loop code (zero_grad, forward, loss, backward, step) |
| **Faster to prototype** | Great for beginners and quick experiments — fewer moving parts to configure correctly |
| **Automatic shape inference** | You rarely need to manually calculate `in_channels`/`in_features` — Keras infers them |
| **Built-in training utilities** | Early stopping, learning rate scheduling, checkpointing, and logging are one-line callbacks (`keras.callbacks.EarlyStopping`, etc.) instead of hand-written logic |
| **Cleaner high-level readability** | `model.summary()` gives an instant, clean table of every layer, output shape, and parameter count |
| **Deployment tooling** | TensorFlow's ecosystem (TensorFlow Lite, TensorFlow.js, TensorFlow Serving) is very mature for deploying Keras models to mobile, web, and production servers |
| **Good default behavior** | Sensible defaults reduce the number of decisions a beginner needs to make correctly to get a working model |

## 6. Where PyTorch Still Has the Edge (balance, not just Keras strengths)

| Advantage | Why it matters |
|---|---|
| **Debugging** | PyTorch's manual loop is plain Python — you can inspect any tensor mid-training with a simple `print()`. Keras's `fit()` is more of a "black box" unless you dig into callbacks. |
| **Research flexibility** | Custom, unusual training procedures (e.g. GANs with alternating updates, reinforcement learning loops) are often easier to hand-write in PyTorch than to bend Keras's `fit()` around. |
| **Dominant in research papers** | Most recent research code (especially in NLP/LLMs) ships in PyTorch, so reading cutting-edge repos is often easier with PyTorch familiarity. |
| **Explicit control** | You already saw how PyTorch made you explicitly write `optimizer.zero_grad()` and `loss.backward()` — tedious, but it means you always know exactly what's happening at each step, which builds deeper understanding (arguably why you learned the internals so thoroughly through this conversation). |

## 7. Summary: Which to reach for, when

- **Learning fundamentals / understanding what's happening under the hood** →
  PyTorch (you're already doing this — good instinct to start there).
- **Quick prototyping, beginner-friendly projects, standard architectures** →
  Keras.
- **Novel research, non-standard training loops, reading research code** →
  PyTorch.
- **Deploying a finished model to mobile/web/production quickly** → Keras/TensorFlow ecosystem.

In practice, many people end up comfortable in both — Keras for quickly
standing something up, PyTorch when they need to customize deeply or read
research code. Since you already understand the *mechanics* (shapes,
parameters, forward/backward passes) from PyTorch, picking up Keras should
mostly feel like learning a shorter syntax for concepts you already know —
not learning deep learning again from scratch.

---

## 8. Quick Reference — Full Side-by-Side Example

| Step | PyTorch | Keras |
|---|---|---|
| Define model | `class Net(nn.Module): ...` | `keras.Sequential([...])` |
| Move to GPU | `model.to(device)` | Automatic (Keras handles device placement) |
| Set up training | `optimizer = optim.Adam(...)` + `criterion = nn.NLLLoss()` | `model.compile(optimizer='adam', loss=...)` |
| Train | Manual `for` loop over epochs/batches | `model.fit(X, y, epochs=..., batch_size=...)` |
| Evaluate | Manual `with torch.no_grad(): ...` loop | `model.evaluate(X_test, y_test)` |
| Predict | `model(x)` after `model.eval()` | `model.predict(x)` |
| Inspect architecture | `print(model)` | `model.summary()` (shows shapes + param counts per layer) |
