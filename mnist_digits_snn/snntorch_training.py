import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import snntorch as snn

def mkdir(path):
    if not os.path.exists(path): os.makedirs(path)
    return path

def save_state(path, epoch, net, optimizer, loss_hist):
    weights_save_file = f'snntorch_model_state_epoch_{epoch}_{datetime.datetime.now().strftime("%d-%m-%YT%H-%M")}.pth'
    state = {
        'epochs': num_epochs,
        'model_state': net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss_hist': loss_hist
    }
    torch.save(state, mkdir(path)/weights_save_file)

class NormaliseToZeroOneRange():
    def __init__(self, divide_by=255, dtype=torch.float16):
        self.divide_by = divide_by
        self.dtype = dtype
    def __call__(self, tensor):
        return (tensor / self.divide_by).to(self.dtype)

class MNISTSequencesDataset(Dataset):

    def __init__(self, file_path, transform=None):
        """
        Arguments:
            file_path (string): Path to the npy file with sequences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.load(file_path, mmap_mode='r')
        self.transform = transform

    def perform_conversion(self, data):
        if LIF_linear_features == 1024: # Unsure how well this will work with passive memmap loading
            # Zero padding data to 32x32
            data_aug = np.zeros((data.shape[0], data.shape[1], 32, 32), dtype=ingest_numpy_dtype)
            data_aug[:,:,:28,:28] = data
            data = data_aug.copy()
            del data_aug
        # data = data.reshape((data.shape[0], data.shape[1], data.shape[2]*data.shape[3]))
        data = data.reshape((-1, data.shape[-2]*data.shape[-1]))
        data = torch.tensor(data, dtype=ingest_torch_dtype)
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

class Net(nn.Module):
    def __init__(self, num_steps):
        super().__init__()

        self.num_steps = num_steps

        # self.fc1 = nn.Linear(28*28, 28*28)
        self.lif1 = snn.RLeaky(beta=beta, linear_features=LIF_linear_features, reset_mechanism=reset_mechanism) # also experiment with all_to_all and V (weights) parameters

    def forward(self, x):
        spk1, mem1 = self.lif1.init_rleaky()
        spk1, mem1 = spk1.to(dtype), mem1.to(dtype)

        spk1_rec = []
        mem1_rec = []

        for step in range(self.num_steps):
            # x = self.fc1(x)
            spk1, mem1 = self.lif1(x[:,step,:], spk1, mem1)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

        # convert lists to tensors
        spk1_rec = torch.stack(spk1_rec)
        mem1_rec = torch.stack(mem1_rec)
        spk1_rec = torch.swapaxes(spk1_rec, 0, 1)
        mem1_rec = torch.swapaxes(mem1_rec, 0, 1)

        return spk1_rec, mem1_rec

def run():

    normalise_transform = NormaliseToZeroOneRange(dtype=dtype)
    mnist_dataset = MNISTSequencesDataset(dataset_path/dataset_file, transform=normalise_transform)
    train_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    batch_data_shape = next(iter(train_loader)).shape

    net = Net(num_steps=batch_data_shape[1]).to(device).to(dtype)


    if init_type == 'pretrained' and LIF_linear_features == 28*28:
        net.lif1.recurrent.weight.data = torch.Tensor(np.load(Path.home()/'RANCZLAB-NAS/iakov/produced/mnist_classification_weights_matrix.npy')).to(device).to(dtype)
    elif init_type == 'pretrained' and LIF_linear_features == 1024:
        class_weights_784 = torch.Tensor(np.load(Path.home()/'RANCZLAB-NAS/iakov/produced/mnist_classification_weights_matrix.npy')).to(device).to(dtype)
        net.lif1.recurrent.weight.data[:class_weights_784.shape[0], :class_weights_784.shape[1]] = class_weights_784

    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    if scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_epoch_step, gamma=0.25)
        
    max_grad_norm = 1.0  # Define the maximum gradient norm threshold

    # Save training config to the folder
    with open(mkdir(weights_save_path)/'training_config.json', 'w') as f:
        json.dump(training_config, f)
    
    # TRAINING LOOP

    loss_hist = []
    counter = 0

    batch_loss = 0

    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data in tqdm(train_batch, f'Epoch {epoch} Loss {batch_loss}'):

            data = data.to(device)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data)

            # initialise the loss and sum over time
            # loss_val = torch.zeros((1), dtype=dtype, device=device) # TRY THIS APPROACH AFTER
            # for step in range(num_steps):
            #     loss_val += loss(mem_rec[step], targets)
            batch_loss = loss(mem_rec, torch.zeros_like(mem_rec)) # preactivation loss
            # batch_loss = loss(spk_rec, torch.zeros_like(mem_rec)) # postactivation loss

            # gradient calculation and weight update
            optimizer.zero_grad()
            batch_loss.backward()
            if grad_clipping == "True": nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            loss_hist.append(batch_loss.item())
        if scheduler_type != 'none': scheduler.step()
        save_state(weights_save_path, epoch, net, optimizer, loss_hist)
    print('Training completed.')

if __name__ == '__main__':
    
    # Setting random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    with open("training_config.json", "r") as read_file:
        training_config = json.load(read_file)
    print('Training configuration successfully read.')

    dataset_sampling_frequency = training_config['dataset_sampling_frequency']
    dataset_path = Path(training_config['dataset_path'])
    dataset_file = Path(training_config['dataset_file'])
    batch_size = training_config['batch_size']
    num_epochs = training_config['num_epochs']
    LIF_linear_features = training_config['LIF_linear_features']
    # LIF_linear_features = 1024
    reset_mechanism = training_config['reset_mechanism']
    init_type = training_config['init_type']
    num_workers = training_config['num_workers']
    learning_rate = training_config['learning_rate']
    scheduler_type = training_config['scheduler_type']
    scheduler_epoch_step = training_config['scheduler_epoch_step']
    grad_clipping = training_config['grad_clipping']
    
    ingest_torch_dtype = torch.uint8
    ingest_numpy_dtype = np.uint8

    tao_mem = 0.01 # 10ms membrane time constant
    timestep = 1/dataset_sampling_frequency
    beta = np.exp(-timestep / tao_mem)

    dtype = torch.float32
    device = torch.device('cuda')

    weights_save_path=Path.home()/f'RANCZLAB-NAS/iakov/produced/mnist_sequence_checkpoints'/f'sequences_{dataset_sampling_frequency}hz_{int(LIF_linear_features)}x{int(LIF_linear_features)}_matrix_class_{init_type}_{reset_mechanism}_reset_beta_{beta:.2f}_scheduler_{scheduler_type}_epochstep_{scheduler_epoch_step}_gradclipp_{grad_clipping}'
    
    run()