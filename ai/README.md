# Vontex AI — from-scratch starter

This folder contains the first educational language-model experiment for Vontex. It initializes a small decoder-only Transformer with **random weights** and trains it on text you provide. It does not call ChatGPT, Gemini, or another hosted language-model API.

## Hardware target

The starter configuration is intentionally small for a Windows PC with 8 GB RAM, an Intel Core i5 CPU, and integrated graphics. It runs on CPU. Training will be slow, and this tiny model is a learning prototype—not a ChatGPT-level assistant.

## 1. Create a Python environment (Windows PowerShell)

From the repository folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ai/requirements.txt
```

If Python 3.11 is not installed, install it from the official Python website, then reopen PowerShell.

## 2. Add training text

Put plain-text material that you have permission to use in `ai/data.txt`. Start with a few hundred KB of clean, varied text. The starter uses a UTF-8 byte tokenizer, so it needs no downloaded vocabulary.

Do not put private messages, passwords, personal records, or copyrighted books/articles into a public repository. Keep your real training dataset outside GitHub if it is private or you do not have redistribution rights.

## 3. Train

```powershell
python ai/train.py
```

The model checkpoint is saved locally to `ai/checkpoints/vontex.pt`. Checkpoints are ignored by Git.

## 4. Generate a sample

```powershell
python ai/generate.py --prompt "Vontex is"
```

## What this prototype does and does not do

- It trains a tiny Transformer from randomly initialized weights.
- It learns to predict the next UTF-8 byte from your training text.
- It does not yet have reliable instruction-following, factual knowledge, long-term memory, file understanding, or safe conversational behavior.
- A useful assistant requires much more data, compute, evaluation, and later a separate inference service.
- The model runs locally on your PC in this first stage. The GitHub Pages website cannot run Python/PyTorch directly.

## Next stages

1. Improve data preparation and evaluation.
2. Add a proper subword tokenizer and larger training corpus.
3. Measure model quality before increasing model size.
4. Build a local API and connect the Vontex web interface.
5. Add explicit user-controlled memory and file processing.
