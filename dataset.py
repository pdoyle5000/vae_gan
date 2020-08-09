from torch.utils.data import Dataset


def _import_data(path: str):
    return ""


def _transforms():
    return


class VaeganDataset(Dataset):
    def __init__(self):
        self.data = _import_data("path/to/data")
        self.transforms = _transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # wrap in transform
        return self.data[idx]
