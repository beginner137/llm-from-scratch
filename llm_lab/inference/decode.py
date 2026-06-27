import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--merges", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    # TODO: load tokenizer from args.vocab and args.merges.
    # TODO: load checkpoint dict and read checkpoint["config"].
    # TODO: construct TransformerLM with the model hyperparameters from config.
    # TODO: load checkpoint["model"] into the model.
    # TODO: encode args.prompt into token ids.
    # TODO: generate args.max_new_tokens using logits from the final position.
    # TODO: decode generated token ids back into text and print.
    print(args)


if __name__ == "__main__":
    main()
