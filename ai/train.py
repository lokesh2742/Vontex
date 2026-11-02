"""Train VontexGPT from random initialization on ai/data.txt."""
from pathlib import Path
import torch
from model import VontexGPT

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data.txt"
CHECKPOINT_DIR = BASE / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "vontex.pt"

# Conservative defaults for an 8 GB RAM CPU-only computer.
BATCH_SIZE = 8
CONTEXT = 128
MAX_STEPS = 2000
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 200
DEVICE = "cpu"


def get_batch(data: torch.Tensor):
    if len(data) <= CONTEXT + 1:
        raise ValueError("Training text is too short. Add more text to ai/data.txt.")
    starts = torch.randint(len(data) - CONTEXT - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + CONTEXT] for i in starts])
    y = torch.stack([data[i + 1:i + CONTEXT + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


def main():
    if not DATA_FILE.exists():
        DATA_FILE.write_text(
            "Vontex is a personal AI project.\n"
            "This is starter text only. Add your own permitted training text here.\n",
            encoding="utf-8",
        )
        print("Created ai/data.txt. Add a larger text collection, then run training again.")
        return

    raw = DATA_FILE.read_bytes()
    if len(raw) < 2048:
        print("Warning: your dataset is very small. The model will mostly memorize it.")
    if len(raw) < CONTEXT + 2:
        raise SystemExit("Add more text to ai/data.txt before training.")

    data = torch.tensor(list(raw), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    if len(val_data) <= CONTEXT + 1:
        val_data = train_data

    torch.manual_seed(42)
    model = VontexGPT(context=CONTEXT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Training on {DEVICE}; bytes={len(data):,}; parameters={sum(p.numel() for p in model.parameters()):,}")
    print("This is a tiny learning experiment; CPU training may take a while.")

    for step in range(1, MAX_STEPS + 1):
        model.train()
        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data)
                _, val_loss = model(vx, vy)
            print(f"step {step:4d}/{MAX_STEPS} | train loss {loss.item():.3f} | val loss {val_loss.item():.3f}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "context": CONTEXT, "vocab_size": 256},
        CHECKPOINT_FILE,
    )
    print(f"Saved model checkpoint to: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
