
### Download Training & Validation dataset

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

### Train TinyStories BPE 
```
caffeinate -s uv run python -m llm_lab.tokenizer.bpe_optimized llm_lab/data/TinyStoriesV2-GPT4-train.txt \
  --output outputs/tinystories_bpe.pkl \
  --output-vocab outputs/tinystories_vocab.json \
  --output-merges outputs/tinystories_merges.txt \
  --max-memory-gb 3.0 \
  --vocab-size 10000
```


### Train OpenWebText BPE
```
caffeinate -s uv run python -m llm_lab.tokenizer.bpe_optimized llm_lab/data/owt_train.txt \
  --output llm_lab/outputs/owt_bpe.pkl \
  --output-vocab llm_lab/outputs/owt_vocab.json \
  --output-merges llm_lab/outputs/owt_merges.txt \
  --max-memory-gb 12.0 \
  --vocab-size 32000
```