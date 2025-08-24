# data/epoch_wrap.py
from torch.utils.data import Dataset

class EpochMultiply(Dataset):
    """
    Makes an epoch effectively k times longer without increasing memory.
    __len__ = base_len * k, and __getitem__ returns base[idx % base_len].
    Your base dataset should already pick a random crop each __getitem__.
    """
    def __init__(self, base: Dataset, k: int = 16):
        assert k >= 1
        self.base = base
        self.k = int(k)

    def __len__(self):
        return len(self.base) * self.k

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]
