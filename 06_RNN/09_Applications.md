# Applications of RNNs

RNNs are widely used in various sequence modeling tasks. Here are some of the most important applications:

## Machine Translation

RNN-based sequence-to-sequence models can translate sentences from one language to another.

### Mathematical Formulation

```math
P(y | x) = \prod_{t=1}^{T_y} P(y_t | y_1, y_2, \ldots, y_{t-1}, x)
```

Where $`x`$ is the input sentence and $`y`$ is the translated output.

**Python Example:**
```python
# This is a toy example of sequence-to-sequence translation (no training)
input_seq = [np.random.randn(3, 1) for _ in range(5)]
output_seq = [np.random.randn(2, 1) for _ in range(6)]
# Use the SimpleSeq2Seq class from previous examples
model = SimpleSeq2Seq(input_size=3, output_size=2, hidden_size=4)
enc_hs = model.encode(input_seq)
preds = model.decode(output_seq, enc_hs)
print('Predicted translation outputs:', [y.ravel() for y in preds])
```

## Speech Recognition

RNNs can map audio features to text sequences.

### Mathematical Formulation

```math
P(w | a) = \frac{P(a | w) P(w)}{P(a)}
```

Where $`a`$ is the audio input and $`w`$ is the word sequence.

**Python Example:**
```python
# Simulate audio features and word outputs
audio_features = [np.random.randn(13, 1) for _ in range(10)]  # e.g., MFCCs
word_outputs = [np.random.randn(5, 1) for _ in range(8)]
model = SimpleSeq2Seq(input_size=13, output_size=5, hidden_size=6)
enc_hs = model.encode(audio_features)
preds = model.decode(word_outputs, enc_hs)
print('Predicted speech-to-text outputs:', [y.ravel() for y in preds])
```

## Text Summarization

RNNs can generate concise summaries of longer texts.

### Mathematical Formulation

```math
y^* = \arg\max_y P(y | x)
```

Where $`x`$ is the input text and $`y^*`$ is the summary.

**Python Example:**
```python
# Simulate document and summary
doc = [np.random.randn(10, 1) for _ in range(12)]
summary = [np.random.randn(6, 1) for _ in range(3)]
model = SimpleSeq2Seq(input_size=10, output_size=6, hidden_size=8)
enc_hs = model.encode(doc)
preds = model.decode(summary, enc_hs)
print('Predicted summary outputs:', [y.ravel() for y in preds])
```

---

Next: [Evaluation Metrics](10_Evaluation.md) 