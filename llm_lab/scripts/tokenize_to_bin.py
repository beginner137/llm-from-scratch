import argparse
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from llm_lab.tokenizer.tokenizer import Tokenizer

_TOKENIZER = None


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize a text file into a uint16 token-id .bin file")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--merges", type=str, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--chunk-bytes", type=int, default=8_000_000)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--max-memory-gb", type=float, default=2.0)
    return parser.parse_args()


def _init_worker(vocab_path, merges_path, special_tokens):
    # Each process has its own Python interpreter, so load one tokenizer per worker.
    # This avoids re-reading vocab/merges for every chunk.
    global _TOKENIZER
    _TOKENIZER = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=special_tokens,
    )


def _tokenize_lines(lines):
    # Runs inside a worker process. Return raw uint16 bytes so the parent can
    # write directly to the output file without building one huge token list.
    ids = list(_TOKENIZER.encode_iterable(lines))
    if ids and max(ids) > np.iinfo(np.uint16).max:
        raise ValueError(f"Token id {max(ids)} does not fit in uint16")
    return np.asarray(ids, dtype=np.uint16).tobytes(), len(ids)


def _iter_line_chunks(input_file, chunk_bytes):
    # Keep line boundaries intact while producing roughly chunk_bytes of text.
    # The chunks are independent tokenization jobs.
    lines = []
    current_bytes = 0
    for line in input_file:
        lines.append(line)
        current_bytes += len(line.encode("utf-8"))
        if current_bytes >= chunk_bytes:
            yield lines, current_bytes
            lines = []
            current_bytes = 0

    if lines:
        yield lines, current_bytes


def main():
    args = parse_args()
    # Bound the number of raw text chunks submitted to workers. This keeps
    # multiprocessing from queueing the whole dataset in memory.
    max_in_flight = max(1, int(args.max_memory_gb * (1024**3) // args.chunk_bytes))
    max_in_flight = min(max_in_flight, max(1, args.num_workers * 4))

    total_tokens = 0
    total_bytes_read = 0
    # Chunks may finish out of order, so every submitted chunk gets an id.
    next_chunk_id = 0
    # The next chunk id that is allowed to be written. This preserves dataset order.
    next_write_id = 0
    # Future -> (chunk id, raw byte count) for chunks currently running.
    pending = {}
    # chunk id -> tokenized bytes for chunks finished but waiting for earlier chunks.
    completed = {}
    with open(args.input, "r", encoding="utf-8") as input_file, open(args.output, "wb") as output_file:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(args.vocab, args.merges, args.special_token),
        ) as executor:
            chunks = _iter_line_chunks(input_file, args.chunk_bytes)

            while True:
                # Fill the worker queue up to the memory-bounded in-flight limit.
                while len(pending) < max_in_flight:
                    try:
                        lines, raw_bytes = next(chunks)
                    except StopIteration:
                        break
                    future = executor.submit(_tokenize_lines, lines)
                    pending[future] = (next_chunk_id, raw_bytes)
                    next_chunk_id += 1
                    total_bytes_read += raw_bytes

                if not pending:
                    break

                # Wait until at least one worker finishes, then collect all finished jobs.
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    chunk_id, _ = pending.pop(future)
                    completed[chunk_id] = future.result()

                # Workers finish in arbitrary order. Write only the next expected
                # chunk, then keep writing consecutive completed chunks if available.
                while next_write_id in completed:
                    token_bytes, token_count = completed.pop(next_write_id)
                    output_file.write(token_bytes)
                    total_tokens += token_count
                    next_write_id += 1
                    print(
                        f"chunks={next_write_id} raw_gb={total_bytes_read / 1e9:.2f} "
                        f"tokens={total_tokens}",
                        flush=True,
                    )

    print(f"Wrote {total_tokens} tokens to {args.output}")


if __name__ == "__main__":
    main()
