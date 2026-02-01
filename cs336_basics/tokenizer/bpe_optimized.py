"""
Optimized BPE Training for Apple Silicon (M3 Pro)
==================================================

Key optimizations:
1. Numba JIT compilation for hot loops
2. Memory-mapped file I/O
3. Spawn multiprocessing context (required for macOS)
4. Tuned process count for performance/efficiency cores
5. Batched processing with memory limits
6. Efficient shared state management
"""

import regex as re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, BinaryIO
import os
import mmap
import multiprocessing as mp
from multiprocessing import get_context
import functools
from tqdm import tqdm

# Try to import numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BPEConfig:
    """Configuration for BPE training optimized for Apple Silicon."""

    # Core parameters
    vocab_size: int = 32000
    special_tokens: List[str] = field(default_factory=list)

    # Resource limits
    max_memory_gb: float = 8.0          # M3 Pro has 18-36GB unified memory
    num_processes: Optional[int] = None  # None = auto-detect

    # Performance tuning
    use_mmap: bool = True               # Memory-mapped file I/O
    use_numba: bool = True              # JIT compilation
    chunk_size_mb: int = 64             # Size per parallel chunk

    # Progress & logging
    show_progress: bool = True

    def __post_init__(self):
        # M3 Pro: Use ~8-10 processes (performance cores + some efficiency)
        # Too many processes = overhead from context switching
        if self.num_processes is None:
            cpu_count = mp.cpu_count()
            # For M3 Pro (12 cores): use 8-10
            # For other Macs: use 75% of cores
            self.num_processes = min(cpu_count, max(4, int(cpu_count * 0.75)))

        # Validate
        if self.vocab_size <= 256:
            raise ValueError("vocab_size must be > 256")
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        if not 1 <= self.num_processes <= 64:
            raise ValueError("num_processes out of reasonable range")

        # Disable numba if not available
        if self.use_numba and not NUMBA_AVAILABLE:
            print("Warning: numba not installed, falling back to pure Python")
            self.use_numba = False

    @classmethod
    def for_m3_pro(cls, vocab_size: int = 32000, ram_gb: float = 18.0) -> "BPEConfig":
        """Optimized preset for M3 Pro."""
        return cls(
            vocab_size=vocab_size,
            max_memory_gb=ram_gb * 0.6,  # Use 60% of available
            num_processes=8,              # 6 perf + 2 efficiency
            use_mmap=True,
            use_numba=True,
            chunk_size_mb=128,            # Larger chunks for unified memory
        )

    def __repr__(self):
        return (f"BPEConfig(vocab_size={self.vocab_size}, "
                f"num_processes={self.num_processes}, "
                f"max_memory_gb={self.max_memory_gb}, "
                f"use_numba={self.use_numba})")


# =============================================================================
# CORE FUNCTIONS (with optional Numba JIT)
# =============================================================================

def merge_pure_python(indices: list, pair: tuple, new_index: int) -> list:
    """Pure Python merge - fallback when numba unavailable."""
    new_indices = []
    i = 0
    n = len(indices)
    while i < n:
        if i + 1 < n and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def merge_numba(indices, pair_0, pair_1, new_index):
        """Numba-accelerated merge - compiles to native ARM64."""
        # Pre-allocate (worst case: no merges)
        result = []
        i = 0
        n = len(indices)
        while i < n:
            if i + 1 < n and indices[i] == pair_0 and indices[i + 1] == pair_1:
                result.append(new_index)
                i += 2
            else:
                result.append(indices[i])
                i += 1
        return result


def merge(indices: list, pair: tuple, new_index: int, use_numba: bool = False) -> list:
    """Merge wrapper - uses numba if available and enabled."""
    if use_numba and NUMBA_AVAILABLE:
        return merge_numba(indices, pair[0], pair[1], new_index)
    return merge_pure_python(indices, pair, new_index)


# =============================================================================
# FILE I/O
# =============================================================================

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """Find chunk boundaries aligned to special token positions."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0, 0]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 1024 * 1024  # 1MB search window

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def read_chunk_mmap(input_path: str, start: int, end: int) -> str:
    """Memory-mapped file reading - efficient for Apple's unified memory."""
    with open(input_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            chunk_bytes = mm[start:end]
            return chunk_bytes.decode("utf-8", errors="ignore")


def read_chunk_standard(input_path: str, start: int, end: int) -> str:
    """Standard file reading - fallback."""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        return chunk_bytes.decode("utf-8", errors="ignore")


# =============================================================================
# PRE-TOKENIZATION
# =============================================================================

# Compile regex once at module level
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pre_tokenize_into_words(string: str) -> List[List[int]]:
    """Split string into words, each as a list of UTF-8 bytes."""
    return [list(m.group(0).encode("utf-8")) for m in GPT2_PAT.finditer(string)]


def parallel_pre_tokenize_and_count(
    text_chunk: str,
    special_tokens: List[str]
) -> Dict[tuple, int]:
    """Tokenize a chunk and count word frequencies."""
    local_word_freq = defaultdict(int)

    # Split by special tokens
    sub_chunks = [text_chunk]
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        sub_chunks = re.split(pattern, text_chunk)

    # Tokenize and count
    for chunk in filter(None, sub_chunks):
        for word in pre_tokenize_into_words(chunk):
            local_word_freq[tuple(word)] += 1

    return dict(local_word_freq)  # Convert for pickling


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_bpe(
    input_path: str,
    config: Optional[BPEConfig] = None,
    # Legacy interface (for backwards compatibility)
    vocab_size: Optional[int] = None,
    special_tokens: Optional[List[str]] = None,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train BPE tokenizer optimized for Apple Silicon.

    Args:
        input_path: Path to training text file
        config: BPEConfig object (recommended)
        vocab_size: Legacy param, use config instead
        special_tokens: Legacy param, use config instead

    Returns:
        vocab: {token_id: bytes}
        merges: [(bytes1, bytes2), ...] in priority order
    """
    # Handle legacy interface
    if config is None:
        config = BPEConfig(
            vocab_size=vocab_size or 32000,
            special_tokens=special_tokens or []
        )

    if config.show_progress:
        print(f"Config: {config}")

    # Use spawn context (required for macOS)
    ctx = get_context("spawn")

    # ==========================================================================
    # PHASE 1: Find chunk boundaries
    # ==========================================================================
    with open(input_path, "rb") as f:
        split_token = b"<|endoftext|>" if "<|endoftext|>" in config.special_tokens else b"\n\n"
        boundaries = find_chunk_boundaries(f, config.num_processes * 2, split_token)

    chunk_boundaries = list(zip(boundaries[:-1], boundaries[1:]))

    if config.show_progress:
        print(f"File split into {len(chunk_boundaries)} chunks")

    # ==========================================================================
    # PHASE 2: Parallel pre-tokenization with batching
    # ==========================================================================

    # Calculate batch sizes based on memory limit
    max_memory_bytes = config.max_memory_gb * (1024 ** 3)
    batches = []
    current_batch = []
    current_size = 0

    for start, end in chunk_boundaries:
        chunk_size = end - start
        # Estimate memory: ~3x for text + tokenized + overhead
        estimated_mem = chunk_size * 3

        if current_batch and current_size + estimated_mem > max_memory_bytes:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append((start, end))
        current_size += estimated_mem

    if current_batch:
        batches.append(current_batch)

    if config.show_progress:
        print(f"Processing in {len(batches)} batch(es) to fit memory limit")

    # Initialize vocabulary
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    special_tokens_sorted = []

    if config.special_tokens:
        special_tokens_sorted = sorted(config.special_tokens, key=len, reverse=True)
        for i, token_str in enumerate(special_tokens_sorted):
            vocab[256 + i] = token_str.encode("utf-8")

    # Choose read function
    read_func = read_chunk_mmap if config.use_mmap else read_chunk_standard

    # Aggregate word frequencies across all batches
    word_freqs: Dict[tuple, int] = defaultdict(int)

    with ctx.Pool(processes=config.num_processes) as pool:
        batch_iter = tqdm(batches, desc="Batches") if config.show_progress else batches

        for batch in batch_iter:
            # Read chunks in parallel
            read_partial = functools.partial(read_func, input_path)
            text_chunks = pool.starmap(read_partial, batch)

            # Tokenize in parallel
            tokenize_partial = functools.partial(
                parallel_pre_tokenize_and_count,
                special_tokens=special_tokens_sorted
            )

            chunk_iter = pool.imap_unordered(tokenize_partial, text_chunks)
            if config.show_progress:
                chunk_iter = tqdm(chunk_iter, total=len(batch),
                                  desc="Tokenizing", leave=False)

            for local_dict in chunk_iter:
                for word, freq in local_dict.items():
                    word_freqs[word] += freq

            # Explicit cleanup
            del text_chunks

    if config.show_progress:
        print(f"Unique words: {len(word_freqs):,}")

    # ==========================================================================
    # PHASE 3: Build initial pair counts and index
    # ==========================================================================

    # Convert to mutable list for efficient updates
    words: List[Tuple[list, int]] = [
        (list(word), freq) for word, freq in word_freqs.items() if freq > 0
    ]
    del word_freqs  # Free memory

    # Build pair counts and reverse index
    pair_counts: Dict[tuple, int] = defaultdict(int)
    pair_to_word_indices: Dict[tuple, set] = defaultdict(set)

    for wi, (word_list, freq) in enumerate(words):
        if len(word_list) < 2:
            continue
        for pair in zip(word_list, word_list[1:]):
            pair_counts[pair] += freq
            pair_to_word_indices[pair].add(wi)

    if config.show_progress:
        print(f"Initial pairs: {len(pair_counts):,}")

    # ==========================================================================
    # PHASE 4: BPE merge loop
    # ==========================================================================

    merge_order: List[Tuple[bytes, bytes]] = []
    next_id = len(vocab)
    num_merges = config.vocab_size - next_id

    merge_iter = range(num_merges)
    if config.show_progress:
        merge_iter = tqdm(merge_iter, desc="BPE Merges")

    for _ in merge_iter:
        if not pair_counts:
            if config.show_progress:
                print("No more pairs to merge")
            break

        # Find best pair (max frequency, tiebreak by lexicographic order)
        best_pair = max(
            pair_counts.keys(),
            key=lambda p: (pair_counts[p], vocab.get(p[0], b""), vocab.get(p[1], b""))
        )

        # Create new token
        new_index = next_id
        next_id += 1

        merge_order.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_index] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # Update affected words
        affected_indices = list(pair_to_word_indices[best_pair])

        for wi in affected_indices:
            word_list, freq = words[wi]

            # Remove old pair counts
            old_pairs = list(zip(word_list, word_list[1:]))
            for pair in old_pairs:
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_word_indices[pair].discard(wi)
                if not pair_to_word_indices[pair]:
                    del pair_to_word_indices[pair]

            # Apply merge
            word_list[:] = merge(word_list, best_pair, new_index, config.use_numba)

            # Add new pair counts
            new_pairs = list(zip(word_list, word_list[1:]))
            for pair in new_pairs:
                pair_counts[pair] += freq
                pair_to_word_indices[pair].add(wi)

    if config.show_progress:
        print(f"Final vocab size: {len(vocab):,}")

    return vocab, merge_order


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BPE tokenizer (M3 Pro optimized)")
    parser.add_argument("input_path", help="Path to training text file")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--max-memory-gb", type=float, default=8.0)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--output", help="Output path for vocab/merges")

    args = parser.parse_args()

    config = BPEConfig(
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        max_memory_gb=args.max_memory_gb,
        num_processes=args.num_processes,
        use_numba=not args.no_numba,
        use_mmap=not args.no_mmap,
    )

    vocab, merges = train_bpe(args.input_path, config)

    print(f"\nTrained {len(vocab)} tokens with {len(merges)} merges")

    if args.output:
        import pickle
        with open(args.output, "wb") as f:
            pickle.dump({"vocab": vocab, "merges": merges}, f)
        print(f"Saved to {args.output}")
