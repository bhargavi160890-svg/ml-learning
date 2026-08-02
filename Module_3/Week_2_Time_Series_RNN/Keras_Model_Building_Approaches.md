# Keras Model-Building Approaches & Library Reference

Covers the three ways to build a Keras model (with a multi-output example),
followed by a one-liner reference table of the Keras/TensorFlow methods and
classes used along the way.

---

## 1. The Three Approaches — Core Differences

| | Sequential | Functional API | Model Subclassing |
|---|---|---|---|
| Structure | Strict linear stack — one input, one output | A graph of layers — supports branching, multiple inputs/outputs | Full Python class — `__init__` defines layers, `call()` defines the forward logic |
| Multi-output support | ❌ Not possible | ✅ Native, designed for this | ✅ Possible, but manual |
| Multi-input support | ❌ Not possible | ✅ Native | ✅ Possible, but manual |
| How you define it | Pass a list of layers | Call layers on tensors: `x = layer(x)` | Write layers in `__init__`, wire them together in `call()` |
| Readability of architecture | Very readable for simple stacks | Graph is visually clear from code layout | Less visual — must read `call()` line by line to see the flow |
| `model.summary()` / plotting | ✅ Full support | ✅ Full support, `keras.utils.plot_model()` works well | ⚠️ Limited — often shows as an opaque block until the model has been called once |
| Custom logic (conditionals, loops, dynamic behavior) | ❌ | ⚠️ Possible but awkward (graph is mostly static) | ✅ Best — full Python control flow inside `call()` |
| Closest PyTorch equivalent | `nn.Sequential` | No direct equivalent (closest: manually wiring tensors) | `nn.Module` (`__init__` + `forward()`) — most familiar to PyTorch users |
| Best for | Simple, single-path models (e.g. your original LeNet) | Multi-input/output, shared-trunk architectures | Highly custom, research-style, or dynamic-computation models |

---

## 2. Example: Shared Trunk with Two Outputs (Classification + Regression)

Only Functional API and Subclassing can express this; Sequential cannot.

### Functional API (recommended for this case)

```python
inputs = keras.Input(shape=(28, 28, 1))

# Shared trunk (feature extractor)
x = layers.Conv2D(6, kernel_size=5)(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.ReLU()(x)
x = layers.Conv2D(16, kernel_size=5)(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.ReLU()(x)
x = layers.Flatten()(x)
shared = layers.Dense(120, activation='relu')(x)

# Branch 1: classification head
class_branch = layers.Dense(84, activation='relu')(shared)
classification_output = layers.Dense(10, activation='softmax', name='classification')(class_branch)

# Branch 2: regression head
reg_branch = layers.Dense(32, activation='relu')(shared)
regression_output = layers.Dense(1, name='regression')(reg_branch)

model = keras.Model(inputs=inputs, outputs=[classification_output, regression_output])

model.compile(
    optimizer='adam',
    loss={'classification': 'sparse_categorical_crossentropy', 'regression': 'mse'},
    loss_weights={'classification': 1.0, 'regression': 0.5},
    metrics={'classification': 'accuracy', 'regression': 'mae'}
)

model.fit(
    train_images,
    {'classification': class_labels, 'regression': reg_targets},
    epochs=35,
    batch_size=100
)
```

### Model Subclassing (same capability, more manual)

```python
class MultiOutputNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(6, kernel_size=5)
        self.pool1 = layers.MaxPooling2D(pool_size=2)
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(16, kernel_size=5)
        self.pool2 = layers.MaxPooling2D(pool_size=2)
        self.relu2 = layers.ReLU()
        self.flatten = layers.Flatten()
        self.shared_dense = layers.Dense(120, activation='relu')
        self.class_dense = layers.Dense(84, activation='relu')
        self.class_out = layers.Dense(10, activation='softmax', name='classification')
        self.reg_dense = layers.Dense(32, activation='relu')
        self.reg_out = layers.Dense(1, name='regression')

    def call(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.flatten(x)
        shared = self.shared_dense(x)
        classification = self.class_out(self.class_dense(shared))
        regression = self.reg_out(self.reg_dense(shared))
        return {'classification': classification, 'regression': regression}

model = MultiOutputNet()
```

**Recommendation**: use the **Functional API** for shared-trunk/multi-head models like this — it's the standard, best-tooled choice. Reach for Subclassing only when you need something the Functional API's static graph genuinely can't express (e.g. conditional branches or loops at runtime).

---

## 3. Library & Method Reference — One-Liner Functionality

### Model-building

| Method / Class | One-line functionality |
|---|---|
| `keras.Sequential([...])` | Builds a strict linear stack of layers, one input → one output |
| `keras.Input(shape=...)` | Defines a symbolic input tensor, used as the starting point for the Functional API |
| `keras.Model(inputs=..., outputs=...)` | Wraps a graph of layers (built via Functional API) into a trainable model, supports multiple inputs/outputs |
| `class MyModel(keras.Model): ...` | Defines a fully custom model via subclassing, with `__init__` (layers) and `call()` (forward logic) |

### Layers

| Layer | One-line functionality |
|---|---|
| `layers.Conv2D(filters, kernel_size)` | Applies 2D convolution filters to extract spatial features |
| `layers.MaxPooling2D(pool_size)` | Shrinks feature maps by keeping the max value in each window |
| `layers.ReLU()` | Non-linear activation: `max(0, x)` |
| `layers.BatchNormalization()` | Normalizes activations between layers for stable, faster training |
| `layers.Dropout(rate)` | Randomly zeroes a fraction of neurons during training to prevent overfitting |
| `layers.Flatten()` | Reshapes a multi-dimensional feature map into a 1D vector per sample |
| `layers.Dense(units)` | Fully connected layer — every input connects to every output neuron |
| `activation='softmax'` (inside a layer) | Converts final raw scores into class probabilities summing to 1 |
| `activation='relu'` (inside a layer) | Shorthand for adding ReLU directly inside a `Dense`/`Conv2D` call, instead of a separate layer |

### Compiling & training

| Method | One-line functionality |
|---|---|
| `model.compile(optimizer=..., loss=..., metrics=...)` | Configures the optimizer, loss function(s), and tracked metrics before training |
| `model.fit(X, y, epochs=..., batch_size=...)` | Runs the entire training loop — batching, forward pass, loss, backward pass, weight updates — across all epochs |
| `model.evaluate(X_test, y_test)` | Computes loss/metrics on a held-out dataset, with training-specific behavior (like dropout) turned off |
| `model.predict(X_new)` | Runs inference only, returning model outputs for new data |
| `model.summary()` | Prints a table of every layer, its output shape, and parameter count |
| `keras.utils.plot_model(model)` | Renders a visual diagram of the model's architecture graph |

### Losses, optimizers, metrics

| Name | One-line functionality |
|---|---|
| `'adam'` / `keras.optimizers.Adam()` | Adaptive-learning-rate optimizer, most common default choice |
| `'sgd'` / `keras.optimizers.SGD()` | Classic stochastic gradient descent, optionally with momentum |
| `'sparse_categorical_crossentropy'` | Classification loss when labels are integers (e.g. `3` for class 3) |
| `'categorical_crossentropy'` | Classification loss when labels are one-hot encoded (e.g. `[0,0,0,1,0]`) |
| `'mse'` (Mean Squared Error) | Regression loss — penalizes squared difference between prediction and target |
| `'mae'` (Mean Absolute Error) | Regression metric/loss — average absolute difference between prediction and target |
| `loss_weights={...}` | Balances the contribution of multiple losses when a model has multiple outputs |

### Callbacks (training utilities)

| Callback | One-line functionality |
|---|---|
| `keras.callbacks.EarlyStopping()` | Stops training automatically once a monitored metric (e.g. validation loss) stops improving |
| `keras.callbacks.ModelCheckpoint()` | Saves the model's weights automatically during training (e.g. only when validation accuracy improves) |
| `keras.callbacks.ReduceLROnPlateau()` | Automatically lowers the learning rate when progress stalls |
| `keras.callbacks.TensorBoard()` | Logs training metrics for visualization in TensorBoard |

### Data handling

| Method | One-line functionality |
|---|---|
| `tf.data.Dataset.from_tensor_slices(...)` | Wraps arrays/tensors into an efficient, batchable dataset pipeline |
| `.batch(size)` | Groups dataset elements into batches of the given size |
| `.shuffle(buffer_size)` | Randomly shuffles the dataset before batching |
| `.prefetch(tf.data.AUTOTUNE)` | Overlaps data loading with model training for speed |
| `keras.preprocessing.image.ImageDataGenerator` | Loads images from folders and applies real-time data augmentation (older API; `tf.keras.utils.image_dataset_from_directory` is the modern replacement) |
| `keras.utils.image_dataset_from_directory(...)` | Loads images from a folder structure (class subfolders) directly into a `tf.data.Dataset` |

### Saving / loading

| Method | One-line functionality |
|---|---|
| `model.save('path')` | Saves the entire model (architecture + weights + optimizer state) to disk |
| `keras.models.load_model('path')` | Loads a previously saved model back into memory |
| `model.save_weights('path')` | Saves only the model's weights, not the architecture |
| `model.load_weights('path')` | Loads weights into a model with a matching architecture |

---

## 4. Quick Takeaway

- **Sequential** → simplest, single-path models only.
- **Functional API** → the standard choice once you need branching, multiple outputs, or multiple inputs (your classification + regression case).
- **Subclassing** → reach for this only when you need genuine dynamic/conditional logic in the forward pass that a static graph can't express.
- Regardless of which you choose, `compile()` + `fit()` + `evaluate()` + `predict()` is the same training/inference interface across all three — the only thing that changes is *how the model itself is defined*.
