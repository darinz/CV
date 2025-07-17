# Applications and Evaluation

Transformers have revolutionized various domains beyond their original application in machine translation. This guide covers the major applications and evaluation metrics used to assess Transformer performance.

## Machine Translation

Machine translation was the original application of Transformers, where they achieved state-of-the-art performance on various language pairs.

### Mathematical Formulation

For source sequence $x = [x_1, x_2, \ldots, x_n]$ and target sequence $y = [y_1, y_2, \ldots, y_m]$:

$$
P(y | x) = \prod_{t=1}^{m} P(y_t | y_1, y_2, \ldots, y_{t-1}, x)
$$

The model learns to maximize the conditional probability:

$$
L = -\sum_{t=1}^{m} \log P(y_t | y_1, y_2, \ldots, y_{t-1}, x)
$$

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class TranslationTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048x_len=50pout=0.1 super(TranslationTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        # Source encoding
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        # Create source mask
        src_mask = (src.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        
        # Encode
        enc_output = self.encoder(src, src_key_padding_mask=~src_mask)
        
        # Target decoding
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        # Create target mask for causal attention
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        
        # Decode
        dec_output = self.decoder(tgt, enc_output, tgt_mask=tgt_mask, 
                                 memory_key_padding_mask=~src_mask)
        
        # Project to vocabulary
        output = self.output_projection(dec_output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(1000_model))
        
        pe[:,0torch.sin(position * div_term)
        pe[:,1torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer(pe, pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def demonstrate_translation():
emonstrate machine translation with Transformer."""
    
    # Parameters
    src_vocab_size = 1000    tgt_vocab_size = 10
    d_model = 256 for demonstration
    n_heads = 8n_layers = 4
    
    # Create model
    model = TranslationTransformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers)
    
    # Create sample data
    batch_size = 2
    src_len = 10 tgt_len = 8
    
    src = torch.randint(1 src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1 tgt_vocab_size, (batch_size, tgt_len))
    
    # Forward pass
    output = model(src, tgt)
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}) 
    return model, output

# Run demonstration
translation_model, translation_output = demonstrate_translation()
```

## Language Modeling

Language modeling predicts the next word given a sequence of previous words.

### Mathematical Formulation

For a sequence of words $w_1, w_2, \ldots, w_n$:

$$
P(w_t | w_1, w_2, \ldots, w_{t-1}) = \text{softmax}(W_{out} h_t + b_{out})
$$

where $h_t$ is the hidden state at position $t$.

### Python Implementation

```python
class LanguageModelTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512_heads=8, n_layers=6, 
                 d_ff=2048x_len=50pout=00.1):
        super(LanguageModelTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder (decoder without cross-attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=true        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Pass through transformer
        output = self.transformer(x, mask=mask)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits

def demonstrate_language_modeling():
monstrate language modeling with Transformer."""
    
    # Parameters
    vocab_size = 10
    d_model = 256   n_heads = 8n_layers = 4
    
    # Create model
    model = LanguageModelTransformer(vocab_size, d_model, n_heads, n_layers)
    
    # Create sample data
    batch_size = 4seq_len = 20
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Calculate perplexity
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(logits.view(-1ocab_size), x.view(-1)
    perplexity = torch.exp(loss)
    
    print(f"Loss: {loss.item():.4)   print(f"Perplexity: {perplexity.item():.4f}) 
    return model, logits, perplexity

# Run demonstration
lm_model, lm_output, perplexity = demonstrate_language_modeling()
```

## Text Classification

Transformers can be used for text classification by using the final hidden states.

### Mathematical Formulation

For a sequence of tokens $[t_1, t_2, \ldots, t_n]$:

$$
y = \text{softmax}(W_{out} \cdot \text{mean}(H) + b_{out})
$$

where $H = [h_1, h_2, \ldots, h_n]$ are the hidden states.

### Python Implementation

```python
class TextClassificationTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048x_len=50pout=0.1        super(TextClassificationTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=true        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create padding mask
        if attention_mask is None:
            attention_mask = (x.sum(dim=-1)
        
        # Pass through transformer
        output = self.transformer(x, src_key_padding_mask=~attention_mask)
        
        # Pooling: mean over sequence length
        mask_expanded = attention_mask.unsqueeze(-1t()
        pooled_output = (output * mask_expanded).sum(dim=1 mask_expanded.sum(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

def demonstrate_text_classification():
ext classification with Transformer."""
    
    # Parameters
    vocab_size = 100
    num_classes = 5
    d_model = 256   n_heads = 8n_layers = 4
    
    # Create model
    model = TextClassificationTransformer(vocab_size, num_classes, d_model, n_heads, n_layers)
    
    # Create sample data
    batch_size = 8seq_len = 50
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Forward pass
    logits = model(x, attention_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Calculate accuracy
    predictions = torch.argmax(logits, dim=-1)
    labels = torch.randint(0, num_classes, (batch_size,))
    accuracy = (predictions == labels).float().mean()
    
    print(f"Accuracy: [object Object]accuracy.item():.4f}) 
    return model, logits, accuracy

# Run demonstration
class_model, class_output, accuracy = demonstrate_text_classification()
```

## Named Entity Recognition (NER)

NER identifies named entities in text using token-level classification.

### Mathematical Formulation

For each token $t_i$ in the sequence:

$$
P(t_i | t_1, t_2, \ldots, t_n) = \text{softmax}(W_{out} h_i + b_{out})
$$

where $h_i$ is the hidden state for token $i$.

### Python Implementation

```python
class NERTransformer(nn.Module):
    def __init__(self, vocab_size, num_entities, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048x_len=50pout=00.1):
        super(NERTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=true        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # NER head
        self.ner_head = nn.Linear(d_model, num_entities)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create padding mask
        if attention_mask is None:
            attention_mask = (x.sum(dim=-1)
        
        # Pass through transformer
        output = self.transformer(x, src_key_padding_mask=~attention_mask)
        
        # Token-level classification
        logits = self.ner_head(output)
        
        return logits

def demonstrate_ner():
Demonstrate Named Entity Recognition with Transformer."""
    
    # Parameters
    vocab_size = 1000    num_entities = 10 # B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O, etc.
    d_model = 256   n_heads = 8n_layers = 4
    
    # Create model
    model = NERTransformer(vocab_size, num_entities, d_model, n_heads, n_layers)
    
    # Create sample data
    batch_size = 4seq_len = 30
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Forward pass
    logits = model(x, attention_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Calculate token-level accuracy
    predictions = torch.argmax(logits, dim=-1)
    labels = torch.randint(0m_entities, (batch_size, seq_len))
    
    # Mask out padding tokens
    valid_tokens = attention_mask.view(-1)
    valid_predictions = predictions.view(-1[valid_tokens]
    valid_labels = labels.view(-1alid_tokens]
    
    accuracy = (valid_predictions == valid_labels).float().mean()
    
    print(f"Token-level accuracy: [object Object]accuracy.item():.4f}) 
    return model, logits, accuracy

# Run demonstration
ner_model, ner_output, ner_accuracy = demonstrate_ner()
```

## Evaluation Metrics

### Perplexity

Perplexity measures how well a language model predicts the next word:

$$
\text{Perplexity} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(w_t | w_1, w_2, \ldots, w_{t-1})\right)
$$

### BLEU Score

BLEU (Bilingual Evaluation Understudy) measures translation quality:

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

where $p_n$ is the n-gram precision and $\text{BP}$ is the brevity penalty.

### Python Implementation

```python
def calculate_perplexity(model, data_loader, vocab_size):
    ""Calculate perplexity for a language model.
    model.eval()
    total_loss = 0    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            logits = model(x)
            
            loss = criterion(logits.view(-1ocab_size), y.view(-1))
            total_loss += loss.item()
            
            # Count non-padding tokens
            total_tokens += (y != 0).sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def calculate_bleu_score(predictions, references, n_grams=4):
   lculate BLEU score for machine translation.""
    
    def get_ngrams(text, n):
     n-grams from text.     returntuple(text[i:i+n]) for i in range(len(text)-n+1)]
    
    def calculate_precision(pred_ngrams, ref_ngrams):
  alculate precision for n-grams.       if not pred_ngrams:
            return 0  
        matches = 0
        for ngram in pred_ngrams:
            if ngram in ref_ngrams:
                matches +=1               ref_ngrams.remove(ngram)
        
        return matches / len(pred_ngrams)
    
    total_bleu =0
    num_samples = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, n_grams + 1):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            precision = calculate_precision(pred_ngrams, ref_ngrams)
            precisions.append(precision)
        
        # Calculate brevity penalty
        pred_len = len(pred)
        ref_len = len(ref)
        bp = min(1, pred_len / ref_len) if ref_len > 00        
        # Calculate BLEU
        if any(p == 0 for p in precisions):
            bleu = 0
        else:
            log_precisions = [math.log(p) for p in precisions]
            bleu = bp * math.exp(sum(log_precisions) / len(log_precisions))
        
        total_bleu += bleu
    
    return total_bleu / num_samples

def calculate_accuracy(predictions, labels, ignore_index=0):
    """Calculate accuracy for classification tasks.   correct = 0
    total = 0
    
    for pred, label in zip(predictions, labels):
        if label != ignore_index:
            if pred == label:
                correct += 1
            total +=1    return correct / total if total > 0 else 0
def demonstrate_evaluation_metrics():
emonstrate various evaluation metrics."""
    
    # Sample data for evaluation
    vocab_size = 100
    batch_size = 4seq_len = 20
    
    # Create dummy model and data
    model = LanguageModelTransformer(vocab_size, d_model=256_heads=8, n_layers=4)
    
    # Simulate data loader
    def dummy_data_loader():
        for _ in range(5):
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            y = torch.randint(0, vocab_size, (batch_size, seq_len))
            yield x, y
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, dummy_data_loader(), vocab_size)
    print(f"Perplexity: {perplexity:.4f}")
    
    # Sample translation data
    predictions =
        ['the', cat', is',on, ,mat'],
        ['i', love', 'machine,learning']
    ]
    references =
        ['the', cat,sits',on, ,mat'],
     i,lovedeep,learning']
    ]
    
    # Calculate BLEU score
    bleu_score = calculate_bleu_score(predictions, references)
    print(fBLEUScore: {bleu_score:.4f}")
    
    # Sample classification data
    pred_labels = [101
    true_labels = [1,20, 3, 2]
    
    # Calculate accuracy
    accuracy = calculate_accuracy(pred_labels, true_labels)
    print(f"Accuracy: {accuracy:.4f})    return perplexity, bleu_score, accuracy

# Run evaluation demonstration
perplexity, bleu_score, accuracy = demonstrate_evaluation_metrics()
```

## Advanced Applications

### Question Answering

Transformers can be used for extractive question answering:

```python
class QuestionAnsweringTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512_heads=8, n_layers=6    super(QuestionAnsweringTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=true        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # QA heads
        self.qa_outputs = nn.Linear(d_model, 2)  # start and end positions
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding and encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        output = self.transformer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        
        # QA logits
        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
```

### Text Generation

Transformers can generate text autoregressively:

```python
def generate_text(model, tokenizer, prompt, max_length=50temperature=1.0):
  enerate text using a Transformer model.
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            outputs = model(input_ids)
            next_token_logits = outputs[0][:, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0kip_special_tokens=True)
```

## Summary

Transformers have become the foundation for state-of-the-art models across various NLP and computer vision tasks:

1. **Machine Translation**: Sequence-to-sequence learning with encoder-decoder architecture
2. **Language Modeling**: Predicting next tokens in sequences
3. **Text Classification**: Using pooled representations for classification4Entity Recognition**: Token-level classification for entity detection
5. **Question Answering**: Extracting answers from context
6. **Text Generation**: Autoregressive text generation

Key evaluation metrics:
- **Perplexity**: Measures language model quality
- **BLEU Score**: Evaluates translation quality
- **Accuracy**: Measures classification performance
- **F1 Score**: Balanced precision and recall for NER

The versatility of Transformers stems from their ability to capture long-range dependencies and their parallel processing capabilities, making them suitable for a wide range of sequence modeling tasks. 