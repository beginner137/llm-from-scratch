import argparse
import torch

from llm_lab.model.layers import TransformerLM
from llm_lab.model.functional import softmax
from llm_lab.training.checkpoint import load_checkpoint_dict
from llm_lab.tokenizer.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Transformer LM")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--merges", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits

    if top_k <= 0:
        raise ValueError("top_k must be positive or None")

    top_k = min(top_k, logits.shape[-1])
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold = values[..., -1, None]

    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None:
        return logits

    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be in (0, 1] or None")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    remove = torch.zeros_like(sorted_remove)
    remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


def main():
    args = parse_args()
    checkpoint = load_checkpoint_dict(
        args.checkpoint, map_location=args.device)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(
            "Checkpoint does not contain config. Resume training from this checkpoint "
            "with the updated training script and save a new checkpoint first."
        )

    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens=["<|endoftext|>"]
    )

    # TODO: construct TransformerLM with the model hyperparameters from config.
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=args.device,
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokens = tokenizer.encode(args.prompt)
    eos_id = tokenizer._token_to_id[b"<|endoftext|>"]
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            input_window = tokens[-config["context_length"]:]
            input_tensor = torch.tensor(
                input_window,
                dtype=torch.long,
                device=args.device
            ).unsqueeze(0)
            logits = model(input_tensor)  # [1, sequence_length, vocab_size]
            next_logits = logits[:, -1, :]
            next_logits = apply_top_k(next_logits, args.top_k)
            next_logits = apply_top_p(next_logits, args.top_p)
            probabilities = softmax(next_logits, dim=-1,
                                    temperature=args.temperature)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token_id = next_token.item()
            if next_token_id == eos_id:
                break
            tokens.append(next_token.item())

    print(tokenizer.decode(tokens))
    return


if __name__ == "__main__":
    main()
