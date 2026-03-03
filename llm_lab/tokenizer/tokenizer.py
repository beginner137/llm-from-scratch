from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
import heapq
import json
from typing import Dict, List, Tuple

import regex as re

from .utils import pre_tokenize_into_words


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self._token_to_id = {
            token: token_id for token_id, token in vocab.items()}
        self._bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self._special_tokens = special_tokens or []
        self._special_token_bytes = {tok: tok.encode(
            "utf-8") for tok in self._special_tokens}

        if self._special_tokens:
            ordered = sorted(self._special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in ordered)
            self._special_regex = re.compile(pattern)
        else:
            self._special_regex = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: Dict[int, bytes] = {}
        for token_id_str, token_val in raw_vocab.items():
            token_id = int(token_id_str)
            if isinstance(token_val, str) and _looks_like_hex(token_val):
                token_bytes = bytes.fromhex(token_val)
            elif isinstance(token_val, str):
                token_bytes = token_val.encode("utf-8")
            else:
                raise ValueError(
                    f"Unsupported vocab value type: {type(token_val)}")
            vocab[token_id] = token_bytes

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = line.strip()
                if not cleaned or cleaned.startswith("#"):
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                a, b = parts
                if _looks_like_hex(a) and _looks_like_hex(b):
                    merges.append((bytes.fromhex(a), bytes.fromhex(b)))
                else:
                    merges.append((a.encode("utf-8"), b.encode("utf-8")))

        if special_tokens:
            for special in special_tokens:
                special_bytes = special.encode("utf-8")
                if special_bytes not in vocab.values():
                    vocab[len(vocab)] = special_bytes

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        if not text:
            return []

        ids: List[int] = []
        for kind, chunk in self._split_by_special_tokens(text):
            if kind == "special":
                token_bytes = self._special_token_bytes[chunk]
                token_id = self._token_to_id[token_bytes]
                ids.append(token_id)
                continue

            for word in pre_tokenize_into_words(chunk):
                word_bytes = bytes(word)
                for bpe_token in self._bpe(word_bytes):
                    token_id = self._token_to_id.get(bpe_token)
                    if token_id is None:
                        raise KeyError(f"Token not in vocab: {bpe_token!r}")
                    ids.append(token_id)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        pieces = [self.vocab[token_id] for token_id in ids]
        # Individual byte-level BPE tokens may contain partial UTF-8 sequences.
        # Use replacement mode for robust per-token decoding.
        return b"".join(pieces).decode("utf-8", errors="replace")

    def _split_by_special_tokens(self, text: str) -> Iterator[Tuple[str, str]]:
        if not self._special_regex:
            yield "text", text
            return

        last = 0
        for match in self._special_regex.finditer(text):
            start, end = match.span()
            if start > last:
                yield "text", text[last:start]
            yield "special", match.group(0)
            last = end
        if last < len(text):
            yield "text", text[last:]

    @lru_cache(maxsize=65536)
    def _bpe(self, token_bytes: bytes) -> Tuple[bytes, ...]:
        if len(token_bytes) <= 1:
            return (token_bytes,)

        # Linked-list representation of the current token sequence.
        symbols: List[bytes] = [bytes([b]) for b in token_bytes]
        n = len(symbols)
        prev = [-1] + list(range(n - 1))
        nxt = list(range(1, n)) + [-1]
        alive = [True] * n

        # Min-heap of candidate merges: (rank, left_node_index, version).
        # `version[left]` invalidates stale heap entries after local updates.
        version = [0] * n
        heap: List[Tuple[int, int, int]] = []

        def refresh_pair(left: int) -> None:
            if left < 0 or not alive[left]:
                return
            version[left] += 1
            right = nxt[left]
            if right == -1 or not alive[right]:
                return
            rank = self._bpe_ranks.get((symbols[left], symbols[right]))
            if rank is not None:
                heapq.heappush(heap, (rank, left, version[left]))

        for left in range(n - 1):
            refresh_pair(left)

        head = 0
        while heap:
            selected_pair: Tuple[bytes, bytes] | None = None
            while heap:
                rank, left, seen_version = heapq.heappop(heap)
                if not alive[left] or seen_version != version[left]:
                    continue
                right = nxt[left]
                if right == -1 or not alive[right]:
                    continue
                pair = (symbols[left], symbols[right])
                current_rank = self._bpe_ranks.get(pair)
                if current_rank is None or current_rank != rank:
                    continue
                selected_pair = pair
                break

            if selected_pair is None:
                break

            a, b = selected_pair
            cur = head
            while cur != -1:
                right = nxt[cur]
                if (
                    right != -1
                    and alive[cur]
                    and alive[right]
                    and symbols[cur] == a
                    and symbols[right] == b
                ):
                    # Merge right into cur, then update only affected adjacent pairs.
                    symbols[cur] = symbols[cur] + symbols[right]
                    alive[right] = False
                    nxt[cur] = nxt[right]
                    if nxt[right] != -1:
                        prev[nxt[right]] = cur

                    refresh_pair(prev[cur])
                    refresh_pair(cur)
                    # Skip the merged token to keep merges non-overlapping.
                    cur = nxt[cur]
                else:
                    cur = right

        result: List[bytes] = []
        cur = head
        while cur != -1:
            if alive[cur]:
                result.append(symbols[cur])
            cur = nxt[cur]
        return tuple(result)


def _looks_like_hex(value: str) -> bool:
    if len(value) % 2 != 0:
        return False
    for ch in value:
        if ch not in "0123456789abcdefABCDEF":
            return False
    return True
