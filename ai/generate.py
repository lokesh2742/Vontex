"""Generate text using a locally trained VontexGPT checkpoint."""
import argparse
from pathlib import Path
import torch
from model import VontexGPT

BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE / "checkpoints" / "vontex.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Vontex is")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        raise SystemExit("No trained model found. Run: python ai/train.py")

    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = VontexGPT(context=saved["context"], vocab_size=saved["vocab_size"])
    model.load_state_dict(saved["model_state"])
    model.eval()

    # UTF-8 byte tokenizer: every byte is represented by an integer 0..255.
    prompt_bytes = args.prompt.encode("utf-8") or b" "
    tokens = torch.tensor([list(prompt_bytes)], dtype=torch.long)
    output = model.generate(tokens, max_new_tokens=args.tokens, temperature=args.temperature)
    print(bytes(output[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
