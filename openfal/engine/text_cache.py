"""
Persistent + In-Memory LRU Cache for Qwen3-VL-32B Layer-50 Multimodal Hidden States.

Because MiniMax-H3 uses the 50th-layer representations of Qwen3-VL-32B to condition the
single-stream 33B DiT, caching the resulting embeddings in Tier 2 (RAM) and Tier 3 (NVMe)
allows openfal to keep Qwen3-VL completely unloaded from GPU VRAM during diffusion!
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class TextEncoderCache:
    def __init__(self, cache_dir: Path, max_ram_entries: int = 64):
        self.cache_dir = cache_dir / "qwen3_vl_layer50"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_ram_entries = max_ram_entries
        self._ram_lru: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def compute_key(
        prompt: str,
        reference_urls: Optional[List[str]] = None,
        task: str = "fl2va",
        expansion_mode: str = "balanced",
    ) -> str:
        payload = json.dumps(
            {
                "prompt": prompt.strip(),
                "refs": sorted(reference_urls or []),
                "task": task,
                "expansion": expansion_mode,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def get(self, key: str) -> Tuple[Optional[np.ndarray], str]:
        if key in self._ram_lru:
            self._ram_lru.move_to_end(key)
            self.hits += 1
            return self._ram_lru[key], "tier2_ram"

        disk_path = self.cache_dir / f"{key}.npy"
        if disk_path.exists():
            arr = np.load(disk_path)
            self._put_ram(key, arr)
            self.hits += 1
            return arr, "tier3_nvme"

        self.misses += 1
        return None, "miss"

    def put(self, key: str, embedding: np.ndarray) -> None:
        self._put_ram(key, embedding)
        disk_path = self.cache_dir / f"{key}.npy"
        np.save(disk_path, embedding)

    def _put_ram(self, key: str, embedding: np.ndarray) -> None:
        self._ram_lru[key] = embedding
        self._ram_lru.move_to_end(key)
        while len(self._ram_lru) > self.max_ram_entries:
            self._ram_lru.popitem(last=False)

    def stats(self) -> dict:
        return {
            "ram_entries": len(self._ram_lru),
            "nvme_entries": len(list(self.cache_dir.glob("*.npy"))),
            "hits": self.hits,
            "misses": self.misses,
        }
