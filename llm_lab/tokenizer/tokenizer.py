from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
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
        self._token_to_id = {token: token_id for token_id, token in vocab.items()}
        self._bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self._special_tokens = special_tokens or []
        self._special_token_bytes = {tok: tok.encode("utf-8") for tok in self._special_tokens}

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
                raise ValueError(f"Unsupported vocab value type: {type(token_val)}")
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
        return b"".join(pieces).decode("utf-8", errors="strict")

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

        word: List[bytes] = [bytes([b]) for b in token_bytes]
        while True:
            best_pair = None
            best_rank = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self._bpe_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_word: List[bytes] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return tuple(word)


def _looks_like_hex(value: str) -> bool:
    if len(value) % 2 != 0:
        return False
    for ch in value:
        if ch not in "0123456789abcdefABCDEF":
            return False
    return True
