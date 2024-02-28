import numpy as np
import torch
from torch.utils.data import Dataset

class MNISTSequencesDataset(Dataset):

    def __init__(self, file_path, LIF_linear_features, ingest_numpy_dtype, ingest_torch_dtype, transform=None):
        """
        Arguments:
            file_path (string): Path to the npy file with sequences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.load(file_path, mmap_mode='r')
        self.transform = transform
        self.LIF_linear_features = LIF_linear_features
        self.ingest_numpy_dtype = ingest_numpy_dtype
        self.ingest_torch_dtype = ingest_torch_dtype

    def perform_conversion(self, data):
        if self.LIF_linear_features == 1024: # Unsure how well this will work with passive memmap loading
            # Zero padding data to 32x32
            data_aug = np.zeros((data.shape[0], data.shape[1], 32, 32), dtype=self.ingest_numpy_dtype)
            data_aug[:,:,:28,:28] = data
            data = data_aug.copy()
            del data_aug
        # data = data.reshape((data.shape[0], data.shape[1], data.shape[2]*data.shape[3]))
        data = data.reshape((-1, data.shape[-2]*data.shape[-1]))
        data = torch.tensor(data, dtype=self.ingest_torch_dtype)
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = self.perform_conversion(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

class NormaliseToZeroOneRange():
    def __init__(self, divide_by=255, dtype=torch.float16):
        self.divide_by = divide_by
        self.dtype = dtype
    def __call__(self, tensor):
        return (tensor / self.divide_by).to(self.dtype)