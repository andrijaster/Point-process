import numpy as np
from torch.utils.data import Dataset, random_split

class point_process_dataset(Dataset):

    import pandas as pd

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        near = idx
        if array[idx] > value:
            near = idx-1
        return near

    def __init__(self, dataset=None):

        if dataset is None:
            self.samples = np.abs(100*np.random.rand(100))
            self.samples = np.sort(self.samples).reshape(1,-1)
        else:
            self.samples = dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = point_process_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    b = next(iter(dataloader))
    print(b)