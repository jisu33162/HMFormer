"""Array-backed minibatch loading."""
import numpy as np
from torch.utils.data import DataLoader, Dataset


class Rows(Dataset):
    def __init__(self, x, y=None, rows=None):
        self.x, self.y = x, y
        self.rows = np.arange(len(x)) if rows is None else np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        return np.array(self.x[row], dtype=np.float32, copy=True), (int(self.y[row]) if self.y is not None else -1), row


def loader(x, y, batch, generator=None, rows=None):
    return DataLoader(Rows(x, y, rows), batch_size=batch, shuffle=generator is not None,
                      generator=generator, num_workers=0, pin_memory=False, drop_last=False)
