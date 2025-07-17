# 05 Visual Question Answering (VQA)

Visual Question Answering (VQA) is the task of answering natural language questions about images. It requires understanding both the visual content and the question.

## Problem Definition
Given an image $I$ and a question $Q$, predict an answer $A$.

## Model Components
- **Image Encoder:** Extracts features from the image (e.g., CNN, ViT).
- **Question Encoder:** Encodes the question (e.g., RNN, Transformer, BERT).
- **Fusion Module:** Combines image and question features (e.g., concatenation, attention, bilinear pooling).
- **Answer Predictor:** Outputs the answer (classification or generation).

### Mathematical Formulation
- $v = f_{\text{img}}(I)$: Image features
- $q = f_{\text{text}}(Q)$: Question features
- $A^* = \arg\max_A p(A|v, q)$: Most probable answer

## Example: Simple VQA Model (PyTorch)

```python
import torch
import torch.nn as nn

class SimpleVQAModel(nn.Module):
    def __init__(self, img_dim, ques_vocab_size, ques_embed_dim, hidden_dim, num_answers):
        super().__init__()
        self.img_fc = nn.Linear(img_dim, hidden_dim)
        self.ques_embedding = nn.Embedding(ques_vocab_size, ques_embed_dim)
        self.ques_rnn = nn.GRU(ques_embed_dim, hidden_dim, batch_first=True)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_answers)
    def forward(self, img_feat, question):
        img_emb = self.img_fc(img_feat)
        ques_emb = self.ques_embedding(question)
        _, ques_h = self.ques_rnn(ques_emb)
        ques_h = ques_h.squeeze(0)
        fused = torch.cat([img_emb, ques_h], dim=-1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits

# Dummy data
img_feat = torch.randn(3, 2048)  # 3 images
questions = torch.randint(0, 500, (3, 8))  # 3 questions, 8 tokens each
model = SimpleVQAModel(img_dim=2048, ques_vocab_size=500, ques_embed_dim=32, hidden_dim=128, num_answers=10)
logits = model(img_feat, questions)
print('Logits shape:', logits.shape)  # (3, 10)
```

### Explanation
- **Image Encoder:** Projects image features.
- **Question Encoder:** Embeds and encodes question.
- **Fusion:** Concatenates and projects features.
- **Classifier:** Predicts answer class.

## Training
- **Loss:** Cross-entropy between predicted and true answer class.
- **Datasets:** VQA v2, GQA, CLEVR.

## Real-World Models
- **MCB, BAN:** Bilinear pooling for fusion.
- **LXMERT, ViLBERT:** Multimodal transformers for VQA.

## Summary
VQA requires deep understanding and reasoning over both images and language, making it a challenging and impactful vision-language task. 