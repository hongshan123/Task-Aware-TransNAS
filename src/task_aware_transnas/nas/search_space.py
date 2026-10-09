"""Validated, hardware-friendly Transformer candidate definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ArchitectureCandidate:
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.hidden_size <= 0:
            raise ValueError("num_layers and hidden_size must be positive")
        if self.num_attention_heads <= 0 or self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.intermediate_ratio <= 0:
            raise ValueError("intermediate_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def intermediate_size(self) -> int:
        return int(self.hidden_size * self.intermediate_ratio)


@dataclass(frozen=True)
class SearchSpace:
    num_layers: Sequence[int] = (2, 4, 8, 12)
    hidden_sizes: Sequence[int] = (512, 768, 1024)
    attention_heads: Sequence[int] = (8, 12, 16)
    intermediate_ratios: Sequence[float] = (2.0, 3.0, 4.0)
    dropouts: Sequence[float] = (0.0, 0.05, 0.1)

    def validate(self) -> None:
        for hidden_size in self.hidden_sizes:
            if not any(hidden_size % heads == 0 for heads in self.attention_heads):
                raise ValueError(f"no configured head count divides hidden size {hidden_size}")
