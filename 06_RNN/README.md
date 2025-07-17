# Recurrent Neural Networks

This module explores Recurrent Neural Networks (RNNs) and their variants, focusing on sequential data processing, language modeling, and sequence-to-sequence tasks.

## Basic RNN

Recurrent Neural Networks are designed to process sequential data by maintaining hidden states that capture information from previous time steps.

### Mathematical Formulation

For a sequence of inputs $x_1, x_2, \ldots, x_T$:

```math
h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
```

```math
y_t = \sigma(W_{hy} h_t + b_y)
```

where:
- $h_t \in \mathbb{R}^{d_h}$ is the hidden state at time $t$
- $x_t \in \mathbb{R}^{d_x}$ is the input at time $t$
- $y_t \in \mathbb{R}^{d_y}$ is the output at time $t$
- $W_{hh}, W_{xh}, W_{hy}$ are weight matrices
- $b_h, b_y$ are bias vectors
- $\sigma$ is the activation function

### Unrolled RNN

The RNN can be unrolled over time:

```math
h_1 = \sigma(W_{hh} h_0 + W_{xh} x_1 + b_h)
```

```math
h_2 = \sigma(W_{hh} h_1 + W_{xh} x_2 + b_h)
```

```math
\vdots
```

```math
h_T = \sigma(W_{hh} h_{T-1} + W_{xh} x_T + b_h)
```

### Loss Function

For sequence prediction:

```math
L = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)
```

where $L_t$ is the loss at time step $t$.

### Backpropagation Through Time (BPTT)

The gradient with respect to $W_{hh}$:

```math
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}
```

where:

```math
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} W_{hh}^T \text{diag}(\sigma'(h_i))
```

### Vanishing/Exploding Gradient Problem

The gradient can vanish or explode due to repeated multiplication:

```math
\|\frac{\partial h_t}{\partial h_k}\| \leq \|W_{hh}\|^{t-k} \|\sigma'\|^{t-k}
```

If $\|W_{hh}\| < 1$: gradients vanish
If $\|W_{hh}\| > 1$: gradients explode

## LSTM (Long Short-Term Memory)

LSTM addresses the vanishing gradient problem by introducing gating mechanisms and a cell state.

### LSTM Architecture

#### Cell State

```math
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
```

#### Hidden State

```math
h_t = o_t \odot \tanh(C_t)
```

#### Gates

**Forget Gate:**
```math
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
```

**Input Gate:**
```math
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
```

**Output Gate:**
```math
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
```

**Candidate Values:**
```math
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
```

### LSTM Variants

#### Peephole LSTM

Gates can also look at the cell state:

```math
f_t = \sigma(W_f \cdot [h_{t-1}, x_t, C_{t-1}] + b_f)
```

```math
i_t = \sigma(W_i \cdot [h_{t-1}, x_t, C_{t-1}] + b_i)
```

```math
o_t = \sigma(W_o \cdot [h_{t-1}, x_t, C_t] + b_o)
```

#### Coupled LSTM

Forget and input gates are coupled:

```math
f_t = 1 - i_t
```

## GRU (Gated Recurrent Unit)

GRU is a simplified version of LSTM with fewer parameters but similar performance.

### GRU Architecture

#### Update Gate

```math
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
```

#### Reset Gate

```math
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
```

#### Hidden State

```math
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
```

where:

```math
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)
```

### Comparison with LSTM

| Feature | LSTM | GRU |
|---------|------|-----|
| Parameters | 4 gates | 3 gates |
| Cell State | Yes | No |
| Memory | Long-term | Short-term |
| Complexity | Higher | Lower |

## Language Modeling

Language modeling predicts the probability of the next word given previous words.

### Mathematical Formulation

```math
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t | w_1, w_2, \ldots, w_{t-1})
```

### RNN Language Model

```math
h_t = \text{RNN}(h_{t-1}, x_t)
```

```math
P(w_t | w_1, w_2, \ldots, w_{t-1}) = \text{softmax}(W_{out} h_t + b_{out})
```

where $x_t$ is the word embedding of $w_t$.

### Loss Function

Cross-entropy loss:

```math
L = -\sum_{t=1}^{T} \log P(w_t | w_1, w_2, \ldots, w_{t-1})
```

### Perplexity

```math
\text{Perplexity} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(w_t | w_1, w_2, \ldots, w_{t-1})\right)
```

### Word Embeddings

```math
x_t = E_{w_t}
```

where $E \in \mathbb{R}^{|V| \times d}$ is the embedding matrix and $|V|$ is vocabulary size.

## Image Captioning

Image captioning generates natural language descriptions of images using CNN-RNN architectures.

### Architecture

#### Encoder (CNN)

```math
v = \text{CNN}(I)
```

where $v \in \mathbb{R}^{d_v}$ is the image feature vector.

#### Decoder (RNN)

```math
h_t = \text{RNN}(h_{t-1}, [x_t, v])
```

```math
P(w_t | w_1, w_2, \ldots, w_{t-1}, v) = \text{softmax}(W_{out} h_t + b_{out})
```

### Attention Mechanism

#### Spatial Attention

```math
\alpha_t = \text{softmax}(W_a \tanh(W_h h_{t-1} + W_v V + b_a))
```

```math
v_t = \sum_{i=1}^{K} \alpha_{t,i} V_i
```

where $V \in \mathbb{R}^{K \times d_v}$ are spatial features.

#### Loss Function

```math
L = -\sum_{t=1}^{T} \log P(w_t | w_1, w_2, \ldots, w_{t-1}, v)
```

### Training Strategies

#### Teacher Forcing

During training, use ground truth as input:

```math
h_t = \text{RNN}(h_{t-1}, [E_{w_t^*}, v])
```

where $w_t^*$ is the ground truth word.

#### Scheduled Sampling

Mix ground truth and predicted words:

```math
w_t^{input} = \begin{cases}
w_t^* & \text{with probability } p \\
\hat{w}_t & \text{with probability } 1-p
\end{cases}
```

## Sequence-to-Sequence

Sequence-to-sequence models map input sequences to output sequences of potentially different lengths.

### Encoder-Decoder Architecture

#### Encoder

```math
h_t^{(enc)} = \text{RNN}^{(enc)}(h_{t-1}^{(enc)}, x_t)
```

#### Decoder

```math
h_t^{(dec)} = \text{RNN}^{(dec)}(h_{t-1}^{(dec)}, [y_{t-1}, c_t])
```

```math
P(y_t | y_1, y_2, \ldots, y_{t-1}, x) = \text{softmax}(W_{out} h_t^{(dec)} + b_{out})
```

where $c_t$ is the context vector.

### Attention Mechanism

#### Global Attention

```math
e_{t,i} = \text{score}(h_t^{(dec)}, h_i^{(enc)})
```

```math
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}
```

```math
c_t = \sum_{i=1}^{T} \alpha_{t,i} h_i^{(enc)}
```

#### Scoring Functions

**Dot Product:**
```math
\text{score}(h_t^{(dec)}, h_i^{(enc)}) = h_t^{(dec)} \cdot h_i^{(dec)}
```

**General:**
```math
\text{score}(h_t^{(dec)}, h_i^{(enc)}) = h_t^{(dec)} W_a h_i^{(enc)}
```

**Concat:**
```math
\text{score}(h_t^{(dec)}, h_i^{(enc)}) = v_a^T \tanh(W_a [h_t^{(dec)}; h_i^{(enc)}])
```

### Training

#### Loss Function

```math
L = -\sum_{t=1}^{T_y} \log P(y_t | y_1, y_2, \ldots, y_{t-1}, x)
```

#### Beam Search

During inference, maintain top-$k$ hypotheses:

```math
\mathcal{H}_t = \text{top-k}(\{h \oplus y_t : h \in \mathcal{H}_{t-1}, y_t \in V\})
```

where $\oplus$ denotes concatenation.

## Advanced RNN Architectures

### Bidirectional RNN

Process sequence in both directions:

```math
h_t^{(f)} = \text{RNN}^{(f)}(h_{t-1}^{(f)}, x_t)
```

```math
h_t^{(b)} = \text{RNN}^{(b)}(h_{t+1}^{(b)}, x_t)
```

```math
h_t = [h_t^{(f)}; h_t^{(b)}]
```

### Deep RNN

Stack multiple RNN layers:

```math
h_t^{(l)} = \text{RNN}^{(l)}(h_{t-1}^{(l)}, h_t^{(l-1)})
```

### Attention RNN

```math
c_t = \text{Attention}(h_{t-1}, \{h_1, h_2, \ldots, h_T\})
```

```math
h_t = \text{RNN}(h_{t-1}, [x_t, c_t])
```

## Training Techniques

### Gradient Clipping

```math
\text{if } \|\nabla L\| > \tau: \quad \nabla L \leftarrow \frac{\tau}{\|\nabla L\|} \nabla L
```

### Dropout

```math
h_t = \text{dropout}(h_t, p)
```

### Layer Normalization

```math
h_t = \text{LayerNorm}(h_t)
```

## Applications

### Machine Translation

```math
P(y | x) = \prod_{t=1}^{T_y} P(y_t | y_1, y_2, \ldots, y_{t-1}, x)
```

### Speech Recognition

```math
P(w | a) = \frac{P(a | w) P(w)}{P(a)}
```

### Text Summarization

```math
y^* = \arg\max_y P(y | x)
```

## Evaluation Metrics

### BLEU Score

```math
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
```

where:
- $\text{BP}$ is the brevity penalty
- $p_n$ is the n-gram precision
- $w_n$ are weights

### ROUGE Score

```math
\text{ROUGE-N} = \frac{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}_{match}(gram_n)}{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}(gram_n)}
```

## Summary

Recurrent Neural Networks provide powerful tools for sequential data processing:

1. **Basic RNN**: Simple but suffers from vanishing gradients
2. **LSTM**: Addresses vanishing gradients with gating mechanisms
3. **GRU**: Simplified LSTM with fewer parameters
4. **Language Modeling**: Predicts next word probabilities
5. **Image Captioning**: Generates descriptions from images
6. **Sequence-to-Sequence**: Maps input to output sequences

These architectures have revolutionized natural language processing and continue to be fundamental building blocks for many AI applications. 