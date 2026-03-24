import json
import os
from typing import Iterable, List

import torch
from torch.utils.data import Dataset, Sampler


def _normalize_selected_indices(selected_indices: Iterable[int]) -> List[int]:
    if torch.is_tensor(selected_indices):
        selected_indices = selected_indices.tolist()
    return [int(x) for x in selected_indices]


def save_selected_indices(file_path, selected_indices):
    selected_indices = _normalize_selected_indices(selected_indices)
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {"selected_indices": selected_indices}
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_selected_indices(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload["selected_indices"]
    return _normalize_selected_indices(payload)


class PairSubsetDataset(Dataset):
    """Recover a real subset from pair-level sample indices."""

    def __init__(self, dataset, selected_indices, return_sample_idx=None):
        if not hasattr(dataset, "get_pair_metadata"):
            raise TypeError("dataset must provide get_pair_metadata(sample_idx)")

        self.dataset = dataset
        self.selected_indices = _normalize_selected_indices(selected_indices)
        if return_sample_idx is None:
            return_sample_idx = getattr(dataset, "return_sample_idx", False)
        self.return_sample_idx = bool(return_sample_idx)

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        sample_idx = self.selected_indices[index]
        if hasattr(self.dataset, "get_sample"):
            return self.dataset.get_sample(sample_idx, return_sample_idx=self.return_sample_idx)
        return self.dataset[sample_idx]

    def get_pair_metadata(self, index):
        sample_idx = self.selected_indices[index]
        return self.dataset.get_pair_metadata(sample_idx)

    def get_selected_indices(self):
        return list(self.selected_indices)


class SelectedIndicesSampler(Sampler[int]):
    """Sample a dataset by global pair indices without materializing a subset dataset."""

    def __init__(self, selected_indices, shuffle=False, generator=None):
        self.selected_indices = _normalize_selected_indices(selected_indices)
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        if not self.shuffle:
            return iter(self.selected_indices)

        perm = torch.randperm(len(self.selected_indices), generator=self.generator).tolist()
        shuffled = [self.selected_indices[idx] for idx in perm]
        return iter(shuffled)

    def __len__(self):
        return len(self.selected_indices)
