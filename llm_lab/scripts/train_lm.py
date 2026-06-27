import argparse
from pathlib import Path

import numpy as np

from llm_lab.model.layers import TransformerLM
from llm_lab.training.checkpoint import load_checkpoint, save_checkpoint
from llm_lab.training.loop import estimate_loss, get_batch
from llm_lab.training.losses import cross_entropy
from llm_lab.training.optimizer import AdamW, gradient_clipping


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")

    return parser.parse_args()


def checkpoint_output_path(checkpoint_path, step):
    path = Path(checkpoint_path)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    path.mkdir(parents=True, exist_ok=True)
    return path / f"checkpoint_step_{step:06d}.pt"


def main():
    args = parse_args()

    # load memmap data
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="c")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="c")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )

    # setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(
            args.resume_from, model, optimizer, map_location=args.device) + 1
        print(f"resumed from {args.resume_from} at step {start_step}")

    for step in range(start_step, args.max_steps):
        inputs, targets = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            args.device,
        )

        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        completed_step = step + 1

        if completed_step % args.log_interval == 0:
            print(f"step {completed_step}: train loss {loss.item():.4f}")

        if completed_step % args.eval_interval == 0:
            val_loss = estimate_loss(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                args.device,
                args.eval_iters,
            )
            print(f"step {completed_step}: val loss {val_loss:.4f}")

        if args.checkpoint_path is not None and completed_step % args.save_interval == 0:
            out = checkpoint_output_path(args.checkpoint_path, completed_step)
            save_checkpoint(model, optimizer, completed_step, out, config=vars(args))
            print(f"step {completed_step}: saved checkpoint to {out}")


if __name__ == "__main__":
    main()
