from __future__ import annotations

from typing import List

import regex as re

GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pre_tokenize_into_words(string: str) -> List[List[int]]:
    """Split string into words, each as a list of UTF-8 bytes."""
    return [list(m.group(0).encode("utf-8")) for m in GPT2_PAT.finditer(string)]
