# Voice E-commerce Ordering System — How the Notebook Works

This document walks through every stage of the notebook: what each piece of code does, why it's built that way, and how the pieces connect to the deployment files (`Order_model.py`, `Order.py`).

---

## The Big Picture

You're building a **speech digit classifier** — given a ~0.9 second audio clip of someone saying "zero" through "five," predict which digit it is. That prediction drives a voice-controlled shopping menu (digit → product category, digit → quantity).

The pipeline has four phases:

```
Audio file (.wav)
      │
      ▼
Feature extraction (get_features)      →  a 900-number vector per clip
      │
      ▼
CNN classifier (Net)                    →  6-class prediction
      │
      ▼
Train on studio data, then team data    →  a saved model file
      │
      ▼
Deploy (Order_model.py + Order.py)      →  live voice ordering
```

---

## Stage 0: Feature Extraction — `get_features()`

**Input:** a filepath to a `.wav` file
**Output:** a `(900, 1)` PyTorch tensor

```python
def get_features(filepath, sr=8000, n_mfcc=30, n_mels=128, frames=15):
    y, sr = librosa.load(filepath, sr=sr)
```

Loads the audio, forcing an 8000 Hz sample rate to match how every recording (studio and team) was captured.

```python
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    features = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)
```

Converts the raw waveform into **30 MFCC coefficients** (Mel-Frequency Cepstral Coefficients) — a compact representation of the sound's spectral shape, standard for speech tasks. This produces a `(30, num_frames)` matrix, where `num_frames` depends on the clip's exact length.

```python
    if features.shape[1] < frames:
        features = np.hstack((features, np.zeros((n_mfcc, frames - features.shape[1]))))
    elif features.shape[1] > frames:
        features = features[:, :frames]
```

Forces every clip to exactly **15 frames** — zero-padding short ones, cropping long ones — so every audio file produces a feature matrix of the identical shape `(30, 15)`, regardless of tiny timing differences between recordings.

```python
    delta1_mfcc = librosa.feature.delta(features, order=1)
    delta2_mfcc = librosa.feature.delta(features, order=2)
```

Computes two **derivative** views of the same `(30, 15)` grid:
- `delta1` = velocity — how each coefficient is changing frame-to-frame
- `delta2` = acceleration — a second derivative, computed **directly from the static MFCC** using a wider local polynomial fit (not by re-differentiating `delta1`)

Both capture how the sound evolves over time, which is often more discriminative for speech than static snapshots alone.

```python
    features = np.hstack((delta1_mfcc.flatten(), delta2_mfcc.flatten()))
    features = features.flatten()[:, np.newaxis]
    features = Variable(torch.from_numpy(features)).float()
    return features
```

Flattens both `(30,15)` matrices to `450` values each, concatenates them → **900 values total**, reshapes to `(900, 1)`, and wraps as a float tensor.

**Note:** the static (non-delta) MFCC values themselves are computed but never included in the final feature vector — only the two derivative views are kept.

---

## Stage 1a: Loading the Dataset — `load_data()`

```python
def load_data(folder_path):
    features = []
    labels = []
    for filepath in glob.glob(os.path.join(folder_path, '*.wav')):
        filename = os.path.basename(filepath)
        label = int(filename.split('_')[0])
        feat = get_features(filepath)
        features.append(feat.numpy().flatten())
        labels.append(label)
    return np.array(features), np.array(labels)
```

Scans every `.wav` in a folder, pulls the class label from the filename's first token (`0_G10_105.wav` → label `0`), extracts its 900-value feature vector, and stacks everything into `(N, 900)` features and `(N,)` labels.

## Train/Test Split & DataLoader

```python
X_train, X_test, y_train, y_test = train_test_split(
    studio_recorded_features, studio_recorded_labels,
    test_size=0.2, random_state=42, stratify=studio_recorded_labels
)
```

80/20 split, `stratify`d so every digit class is proportionally represented in both sets.

```python
X_train_tensor = torch.from_numpy(X_train).float().unsqueeze(2)   # (N, 900) -> (N, 900, 1)
```

Reshapes each 900-value feature vector into `(900, 1)` — 900 "channels," sequence length 1 — to match what the `Conv1d` layers expect as input shape `(batch, channels, length)`.

---

## Stage 1b: The CNN Architecture — `Net`

```python
self.conv1 = nn.Conv1d(in_channels=900, out_channels=400, kernel_size=1)
self.bn1 = nn.BatchNorm1d(400)
self.relu1 = nn.ReLU()
self.maxpool1 = nn.MaxPool1d(1)
self.dropout = nn.Dropout(p=0.25)
# ... conv2 (400→200), conv3 (200→100), same pattern
self.fc1 = nn.Linear(100, 6)
self.logsoftmax = nn.LogSoftmax(dim=1)
```

Three `Conv1d → BatchNorm → ReLU → MaxPool → Dropout` blocks progressively compress 900 → 400 → 200 → 100 values, then a final `Linear(100, 6)` + `LogSoftmax` produces log-probabilities over the 6 digit classes.

**Important nuance:** because the input's sequence length is 1 and every `kernel_size=1`, these `Conv1d` layers never actually "slide" anywhere — there's only one position to look at. Each layer just mixes across the 900 (then 400, then 200) input channels at that single spot. This makes it mathematically equivalent to a stack of `Linear` layers, not a layer that learns temporal patterns across multiple time steps (which is what `Conv1d` is typically used for in audio, e.g. sliding across consecutive MFCC frames). It's a valid design for compressing a fixed feature vector — just worth knowing it's not doing "real" temporal convolution.

```python
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

`NLLLoss` (Negative Log-Likelihood) pairs correctly with `LogSoftmax` output. `Adam` is a common default optimizer for fast, stable convergence.

---

## Stage 1c/d: Training & Testing Loop

```python
for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()      # clear old gradients
        output = model(data)        # forward pass
        loss = criterion(output, target)
        loss.backward()             # backward pass (compute gradients)
        optimizer.step()            # update weights
```

Standard PyTorch training loop: five steps repeated per batch, tracking running loss/accuracy per epoch.

```python
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
```

`model.eval()` disables dropout/batch-norm's training behavior; `torch.no_grad()` skips gradient tracking since we're not updating weights here — just measuring accuracy on held-out data.

---

## Stage 1e: Saving the Model

```python
model_path = 'studio_model.t7'
torch.save({'net_dict': model.state_dict()}, model_path)
```

**Important:** this saves the weights **wrapped in a dictionary** under the key `'net_dict'`, not as a bare state dict. This has to match exactly how `Order.py` loads it on the server:

```python
ckpt = torch.load(BASE_DIR + "/Hackathon-setup/speech_model.t7")
model.load_state_dict(ckpt['net_dict'])
```

If you ever save with just `torch.save(model.state_dict(), path)` (no wrapper dict), loading it this way fails — that mismatch caused one of the deployment errors along the way.

---

## Stage 2a: Team Data

```python
!wget -r -A .wav https://aiml-sandbox1.talentsprint.com/audio_recorder/<YOUR_GROUP_ID>/team_data/ ...
team_features, team_labels = load_data('./team_data')

combined_features = np.vstack((studio_recorded_features, team_features))
combined_labels = np.concatenate((studio_recorded_labels, team_labels))
```

Same `load_data()` function reused on your team's recordings, then stacked on top of the studio data — `vstack` for feature rows, `concatenate` for labels.

## Stage 2b: Fine-Tuning

```python
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt['net_dict'])       # unwrap the saved dict first
optimizer = optim.Adam(model.parameters(), lr=0.0001)   # smaller LR
```

Rather than training a fresh model from scratch on the combined data, this **loads the Stage 1 studio-trained weights** and continues training at a **10x lower learning rate**. This "fine-tuning" approach typically converges faster and generalizes better than starting over, since the model keeps what it already learned from the larger studio dataset while adapting to your team's voices.

---

## Deployment: `Order_model.py` and `Order.py`

These two files run on the server, not in this notebook — they're what actually powers the live voice-ordering demo.

**`Order_model.py`** contains:
- The exact same `Net` class (architecture must match your saved weights precisely, or `load_state_dict()` fails)
- `get_features()` — identical to the notebook version
- `record_voice()` / `play_audio()` — given, handle the microphone

**`Order.py`** contains the ordering logic:

```python
def classify_input(self, features, model):
    features = features.unsqueeze(0)          # (900,1) -> (1,900,1): add batch dim
    model.eval()
    with torch.no_grad():
        output = model(features)
        predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label
```

Takes the 900-value feature tensor from `get_features()`, adds a batch dimension (since the model always expects a batch, even of size 1), runs inference, and picks the class with the highest log-probability.

```python
def take_user_input(self, flag):
    features = Order_model.get_features(BASE_DIR + "/Hackathon-setup/1_userinput_1.wav")
    model = Order_model.Net()
    ckpt = torch.load(BASE_DIR + "/Hackathon-setup/speech_model.t7", map_location=device)
    model.load_state_dict(ckpt['net_dict'])
    model.to(device)
    digit = self.classify_input(features, model)
    digit, choice = self.confirm_input(digit, flag)
    return digit, choice
```

Records the user's voice, extracts features, loads your trained model from `speech_model.t7`, classifies the digit, then hands off to `confirm_input()` (given/complete) to interpret it as a menu choice or quantity.

---

## Key Pitfalls Solved Along the Way

| Problem | Cause | Fix |
|---|---|---|
| `load_state_dict()` shape mismatch | Saved a bare `state_dict()`, but `Order.py` expects `ckpt['net_dict']` | Save as `{'net_dict': model.state_dict()}` |
| Fine-tuning cell crashes loading Stage 1 weights | Same wrapper-dict mismatch, but on the *loading* side | Unwrap with `ckpt['net_dict']` before `load_state_dict()` |
| Server `TypeIs` import error | Server venv had `typing_extensions==4.8.0`; `torch` internally needs `TypeIs`, added in `typing_extensions>=4.10` | `pip install --upgrade "typing_extensions>=4.10"` in the venv |
| `Order.py` `NameError` on `Record_audio` | Leftover typo in starter template — imported as `Order_model` but referenced as `Record_audio` | Changed all references to `Order_model` |

---

## Quick Reference: Shapes Through the Pipeline

```
Raw audio (.wav, ~0.9s @ 8000Hz)
    → MFCC:              (30, 15)
    → delta1:             (30, 15)  →  flatten → 450
    → delta2:             (30, 15)  →  flatten → 450
    → concatenated:                     900
    → reshaped:                     (900, 1)
    → model input (batched):     (batch, 900, 1)
    → conv1 output:               (batch, 400, 1)
    → conv2 output:               (batch, 200, 1)
    → conv3 output:               (batch, 100, 1)
    → flattened:                  (batch, 100)
    → fc1 output:                 (batch, 6)     ← log-probabilities per digit
```
