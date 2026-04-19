# i23-2627 — NLP Assignment 2: Neural NLP Pipeline

**Course:** CS-4063 Natural Language Processing — FAST NUCES  
**Student:** i23-2627  
**Framework:** PyTorch (from scratch)

---

## Repository Structure

```
i23-2627_Assignment2_DS-B/
├── i23-2627_Assignment2_DS-B.ipynb   ← Main notebook (all cells executed)
├── cleaned.txt                        ← Preprocessed BBC Urdu corpus
├── raw.txt                            ← Raw BBC Urdu corpus
├── Metadata.json                      ← Article metadata & topic labels
├── embeddings/
│   ├── tfidf_matrix.npy               ← TF-IDF matrix (V × N)
│   ├── ppmi_matrix.npy                ← PPMI co-occurrence matrix
│   ├── embeddings_w2v.npy             ← Skip-gram ½(V+U) embeddings
│   └── word2idx.json                  ← Token → index mapping
├── models/
│   ├── bilstm_pos.pt                  ← BiLSTM POS tagger checkpoint
│   ├── bilstm_ner.pt                  ← BiLSTM NER tagger (CRF) checkpoint
│   └── transformer_cls.pt             ← Transformer classifier checkpoint
├── data/
│   ├── pos_train.conll
│   ├── pos_test.conll
│   ├── ner_train.conll
│   └── ner_test.conll
└── figures/
    ├── tsne_ppmi.png
    ├── w2v_loss_curve.png
    ├── pos_training_curves.png
    ├── pos_confusion_matrix.png
    ├── ner_training_curves.png
    ├── transformer_training_curves.png
    ├── transformer_confusion_matrix.png
    └── attn_heatmap_article{1,2,3}.png
```

---

## Requirements

```bash
pip install torch numpy scikit-learn matplotlib seaborn tqdm
```

Python ≥ 3.9, PyTorch ≥ 2.0 recommended.  
GPU optional but recommended for faster training.

---

## How to Reproduce

### 1. Clone and set up

```bash
git clone https://github.com/YAYAKHAN107/nlp-a2.git
cd i23-2627-NLP-Assignment2
pip install torch numpy scikit-learn matplotlib seaborn
```

### 2. Place corpus files

Copy `cleaned.txt`, `raw.txt`, and `Metadata.json` into the root directory (same folder as the notebook).

### 3. Run the notebook

```bash
jupyter notebook i23-2627_Assignment2_DS-B.ipynb
```

Run all cells top to bottom (**Kernel → Restart & Run All**).  
Expected total runtime: ~30–60 min on CPU, ~10–15 min with GPU.

---

## Part Summary

| Part | Task | Key Output |
|------|------|-----------|
| 1.1 | TF-IDF matrix | `embeddings/tfidf_matrix.npy` |
| 1.2 | PPMI + t-SNE | `embeddings/ppmi_matrix.npy`, `figures/tsne_ppmi.png` |
| 2.1 | Skip-gram Word2Vec (5 epochs) | `embeddings/embeddings_w2v.npy` |
| 2.2 | NN eval + analogy + 4-condition MRR | Printed in notebook |
| 3 | POS & NER annotation (500 sents) | `data/*.conll` |
| 4 | 2-layer BiLSTM + CRF | `models/bilstm_pos.pt`, `models/bilstm_ner.pt` |
| 5 | Eval + ablation study | Confusion matrices, F1 scores |
| 6–7 | Transformer encoder (4 blocks) | `models/transformer_cls.pt` |
| 8 | Attention heatmaps + comparison | `figures/attn_heatmap_*.png` |

---

## Implementation Notes

- **No pretrained models** used. All models implemented from scratch in PyTorch.
- `nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoder` are **not used**.
- CRF uses custom forward algorithm (log-space) + Viterbi decoding.
- Sinusoidal PE stored as a non-learned `register_buffer`.
- Transformer uses Pre-Layer Normalisation (Pre-LN) residual connections.
- AdamW + cosine LR schedule with 50 warmup steps for Transformer training.
