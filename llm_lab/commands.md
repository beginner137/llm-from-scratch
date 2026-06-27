# LLM Lab Commands

Run these from the repository root.

## Download Data

```bash
mkdir -p llm_lab/data
cd llm_lab/data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip -f owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip -f owt_valid.txt.gz

cd ../..
```

## Train Tokenizers

TinyStories BPE:

```bash
caffeinate -s uv run python -m llm_lab.tokenizer.bpe_optimized \
  llm_lab/data/TinyStoriesV2-GPT4-train.txt \
  --output llm_lab/outputs/tinystories_bpe.pkl \
  --output-vocab llm_lab/outputs/tinystories_vocab.json \
  --output-merges llm_lab/outputs/tinystories_merges.txt \
  --max-memory-gb 3.0 \
  --vocab-size 10000
```

OpenWebText BPE:

```bash
caffeinate -s uv run python -m llm_lab.tokenizer.bpe_optimized \
  llm_lab/data/owt_train.txt \
  --output llm_lab/outputs/owt_bpe.pkl \
  --output-vocab llm_lab/outputs/owt_vocab.json \
  --output-merges llm_lab/outputs/owt_merges.txt \
  --max-memory-gb 8.0 \
  --vocab-size 32000
```

## Build Token ID Bins

These commands tokenize raw text into `uint16` `.bin` files for `np.memmap`.

TinyStories train:

```bash
uv run python -m llm_lab.scripts.tokenize_to_bin \
  --input llm_lab/data/TinyStoriesV2-GPT4-train.txt \
  --output llm_lab/data/tinystories_train_uint16.bin \
  --vocab llm_lab/outputs/tinystories_vocab.json \
  --merges llm_lab/outputs/tinystories_merges.txt \
  --special-token "<|endoftext|>" \
  --num-workers 8 \
  --chunk-bytes 8000000 \
  --max-memory-gb 2
```

TinyStories valid:

```bash
uv run python -m llm_lab.scripts.tokenize_to_bin \
  --input llm_lab/data/TinyStoriesV2-GPT4-valid.txt \
  --output llm_lab/data/tinystories_valid_uint16.bin \
  --vocab llm_lab/outputs/tinystories_vocab.json \
  --merges llm_lab/outputs/tinystories_merges.txt \
  --special-token "<|endoftext|>" \
  --num-workers 8 \
  --chunk-bytes 8000000 \
  --max-memory-gb 2
```

OpenWebText train:

```bash
uv run python -m llm_lab.scripts.tokenize_to_bin \
  --input llm_lab/data/owt_train.txt \
  --output llm_lab/data/owt_train_uint16.bin \
  --vocab llm_lab/outputs/owt_vocab.json \
  --merges llm_lab/outputs/owt_merges.txt \
  --special-token "<|endoftext|>" \
  --num-workers 8 \
  --chunk-bytes 8000000 \
  --max-memory-gb 2
```

OpenWebText valid:

```bash
uv run python -m llm_lab.scripts.tokenize_to_bin \
  --input llm_lab/data/owt_valid.txt \
  --output llm_lab/data/owt_valid_uint16.bin \
  --vocab llm_lab/outputs/owt_vocab.json \
  --merges llm_lab/outputs/owt_merges.txt \
  --special-token "<|endoftext|>" \
  --num-workers 8 \
  --chunk-bytes 8000000 \
  --max-memory-gb 2
```

## Train TinyStories

Small smoke test:

```bash
uv run python -m llm_lab.scripts.train_lm \
  --train-data llm_lab/data/tinystories_train_uint16.bin \
  --val-data llm_lab/data/tinystories_valid_uint16.bin \
  --vocab-size 10000 \
  --context-length 128 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --d-ff 341 \
  --batch-size 16 \
  --max-steps 5000 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --log-interval 20 \
  --eval-interval 200 \
  --eval-iters 20 \
  --save-interval 1000 \
  --checkpoint-path checkpoints/tinystories-eos \
  --device mps
```

Overnight MPS run:

```bash
uv run python -m llm_lab.scripts.train_lm \
  --train-data llm_lab/data/tinystories_train_uint16.bin \
  --val-data llm_lab/data/tinystories_valid_uint16.bin \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 256 \
  --num-layers 6 \
  --num-heads 8 \
  --d-ff 682 \
  --batch-size 8 \
  --max-steps 50000 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --log-interval 50 \
  --eval-interval 500 \
  --eval-iters 20 \
  --save-interval 2500 \
  --checkpoint-path checkpoints/tinystories-eos-256 \
  --device mps
```

Resume training:

```bash
uv run python -m llm_lab.scripts.train_lm \
  --train-data llm_lab/data/tinystories_train_uint16.bin \
  --val-data llm_lab/data/tinystories_valid_uint16.bin \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 256 \
  --num-layers 6 \
  --num-heads 8 \
  --d-ff 682 \
  --batch-size 8 \
  --max-steps 75000 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --log-interval 50 \
  --eval-interval 500 \
  --eval-iters 20 \
  --save-interval 2500 \
  --checkpoint-path checkpoints/tinystories-eos-256 \
  --resume-from checkpoints/tinystories-eos-256/checkpoint_step_050000.pt \
  --device mps
```

## Decode

TinyStories generation:

```bash
uv run python -m llm_lab.inference.decode \
  --checkpoint checkpoints/tinystories-eos-256/checkpoint_step_050000.pt \
  --vocab llm_lab/outputs/tinystories_vocab.json \
  --merges llm_lab/outputs/tinystories_merges.txt \
  --prompt "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 0.9 \
  --top-p 0.95 \
  --device mps
```

More deterministic generation:

```bash
uv run python -m llm_lab.inference.decode \
  --checkpoint checkpoints/tinystories-eos-256/checkpoint_step_050000.pt \
  --vocab llm_lab/outputs/tinystories_vocab.json \
  --merges llm_lab/outputs/tinystories_merges.txt \
  --prompt "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 0.7 \
  --top-k 20 \
  --device mps
```
