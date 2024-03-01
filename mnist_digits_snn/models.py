import torch
import torch.nn as nn
import snntorch as snn

class SNN1(nn.Module):
    def __init__(self, num_steps, beta, LIF_linear_features, reset_mechanism, dtype):
        super().__init__()

        self.num_steps = num_steps
        self.dtype = dtype

        # self.fc1 = nn.Linear(28*28, 28*28)
        self.lif1 = snn.RLeaky(beta=beta, linear_features=LIF_linear_features, reset_mechanism=reset_mechanism) # also experiment with all_to_all and V (weights) parameters

    def forward(self, x):
        spk1, mem1 = self.lif1.init_rleaky()
        spk1, mem1 = spk1.to(self.dtype), mem1.to(self.dtype)

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