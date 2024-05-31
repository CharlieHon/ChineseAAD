import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    1D:
        x_train (3872, 128, 32); y_train (3872, 1)
        x_valid (480, 128, 32);  y_valid (480, 1)
    """
    def __init__(self, eeg, label):
        self.eeg = eeg
        self.label = label

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, index):
        x = self.eeg[index]
        x = torch.tensor(x, dtype=torch.float32)
        y = self.label[index][0]
        y = torch.tensor(y, dtype=torch.long)
        return x, y
