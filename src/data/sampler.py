"""Batch sampler enforcing format diversity. implementation.md §6.2.

Shuffle alone gives the adversary no guaranteed within-batch format
variation. Each batch is partitioned into >=8 groups; each group draws from
one (format, embodiment) pair, formats sampled without replacement per
batch, embodiment sampled per group so every per-embodiment head keeps
receiving gradient.
"""
import random

from torch.utils.data import Sampler


class FormatBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size: int, num_groups: int = 8, num_batches_per_epoch: int | None = None):
        if batch_size % num_groups != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by num_groups={num_groups}")

        self.group_size = batch_size // num_groups
        self.num_groups = num_groups
        self.num_batches = num_batches_per_epoch or (len(dataset) // batch_size)

        self.index_by_key: dict[tuple[str, str], list[int]] = {}
        self.format_to_embodiments: dict[str, list[str]] = {}
        for i, (episode_id, _t) in enumerate(dataset.index):
            meta = dataset._metas[episode_id]
            key = (meta["format_id"], meta["embodiment_id"])
            self.index_by_key.setdefault(key, []).append(i)
            embs = self.format_to_embodiments.setdefault(meta["format_id"], [])
            if meta["embodiment_id"] not in embs:
                embs.append(meta["embodiment_id"])

        if len(self.format_to_embodiments) < num_groups:
            raise ValueError(
                f"only {len(self.format_to_embodiments)} distinct formats in split, "
                f"need >= {num_groups}"
            )

    def __iter__(self):
        formats = list(self.format_to_embodiments.keys())
        for _ in range(self.num_batches):
            chosen_formats = random.sample(formats, self.num_groups)
            batch = []
            for fmt in chosen_formats:
                emb = random.choice(self.format_to_embodiments[fmt])
                pool = self.index_by_key[(fmt, emb)]
                if len(pool) >= self.group_size:
                    batch.extend(random.sample(pool, self.group_size))
                else:
                    batch.extend(random.choice(pool) for _ in range(self.group_size))
            yield batch

    def __len__(self):
        return self.num_batches
