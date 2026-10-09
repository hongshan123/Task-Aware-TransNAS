"""Small reservoir-sampling buffer for future continual-learning experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReplayItem:
    task_id: str
    payload: Any


class ReplayBuffer:
    """Fixed-capacity reservoir buffer; this module performs no training."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._seen = 0
        self._items: list[ReplayItem] = []

    def add(self, task_id: str, payload: Any) -> None:
        item = ReplayItem(task_id=task_id, payload=payload)
        self._seen += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        index = self._rng.randrange(self._seen)
        if index < self.capacity:
            self._items[index] = item

    def sample(self, count: int) -> list[ReplayItem]:
        if count < 0:
            raise ValueError("count cannot be negative")
        return self._rng.sample(self._items, min(count, len(self._items)))

    def __len__(self) -> int:
        return len(self._items)
