# GPT-2 Reproduction: From Scratch

**A faithful implementation of the GPT-2 (124M) language model architecture, built for educational depth and technical transparency.**

---

## Navigation Guide

* **Code Quality:** Check `src/model.py` to see clean, documented implementation of the Transformer architecture.
* **Project Management:** See `notes/progress_log.md` (or similar) to view my step-by-step learning process and debugging history.
* **Implementation Details:** The `GPT` class in `src/model.py` strictly follows the GPT-2 specifications (LayerNorm placement, causal masking, weight tying).
* **Experiments:** See the `experiments/` directory for isolated tests on attention mechanisms and positional encodings.
* **Reproducibility:** The training loop in `src/train.py` includes gradient accumulation and mixed-precision training support.

---

## Project Overview

This repository contains a from-scratch implementation of the GPT-2 language model, following the architecture described in the original OpenAI paper and the educational methodology of Andrej Karpathy.

### Motivation
I undertook this project to gain a "first principles" understanding of:
1.  **The Transformer Architecture:** How Self-Attention and MLPs interact to process information.
2.  **Tensor Shapes & Broadcasting:** Managing high-dimensional data flow without errors.
3.  **Optimization:** Implementing backpropagation, learning rate scheduling, and efficient data loading.

---

## Technical Stack

* **Language:** Python 3.10+
* **Framework:** PyTorch
* **Tokenizer:** TikToken (BPE)
* **Utilities:** NumPy, TQDM

---

## Repository Structure

```text
gpt2-reproduction/
├── src/
│   ├── __init__.py
│   ├── config.py       # Hyperparameters and model configuration
│   ├── model.py        # The GPT-2 Architecture (Attention, MLP, Block)
│   └── train.py        # Training loop and optimization logic
├── experiments/        # Isolated scripts for testing concepts (e.g., attention.py)
├── notes/              # Learning journal and theoretical derivation notes
├── data/               # (Ignored) Dataset storage
├── .gitignore          # Best-practices gitignore for Python/ML
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---
## Credits & References

* **Primary Reference:** [Andrej Karpathy: Let's build GPT](https://youtu.be/l8pRSuU81PU)
* **Paper:** [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* **Original repo:** [nanoGPT](https://github.com/karpathy/nanoGPT)
---

Author: Sacha DEGOIX
