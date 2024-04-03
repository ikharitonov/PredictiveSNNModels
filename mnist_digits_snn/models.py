import torch
import torch.nn as nn
import snntorch as snn
import math

def get_model(model_name):
    model_dict = {
        "SNN1": SNN1,
        "SNN1syn": SNN1syn,
        "SNN2": SNN2,
        "SNN2syn": SNN2syn,
        "SNN1_Supervised_Extension": SNN1_Supervised_Extension,
        # "RNN1": RNN1
    }
    return model_dict[model_name]

class SNN1(nn.Module):
    # Without the first Linear layer
    def __init__(self, num_steps, beta, alpha, LIF_linear_features, reset_mechanism, weight_init, dtype):
        super().__init__()

        self.num_steps = num_steps
        self.dtype = dtype

        self.lif1 = snn.RLeaky(beta=beta, linear_features=LIF_linear_features, reset_mechanism=reset_mechanism) # also experiment with all_to_all and V (weights) parameters
        
        if weight_init == 'normal':
            stdv = 1. / math.sqrt(LIF_linear_features)
            self.lif1.recurrent.weight.data.normal_(0,stdv)
        

    def forward(self, x):
        spk1, mem1 = self.lif1.init_rleaky()
        spk1, mem1 = spk1.to(self.dtype), mem1.to(self.dtype)

        spk1_rec = []
        mem1_rec = []

        for step in range(self.num_steps):
            spk1, mem1 = self.lif1(x[:,step,:], spk1, mem1)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

        # convert lists to tensors
        spk1_rec = torch.stack(spk1_rec)
        mem1_rec = torch.stack(mem1_rec)
        spk1_rec = torch.swapaxes(spk1_rec, 0, 1)
        mem1_rec = torch.swapaxes(mem1_rec, 0, 1)

        return spk1_rec, mem1_rec
    
class SNN1syn(nn.Module):
    # Without the first Linear layer
    def __init__(self, num_steps, beta, alpha, LIF_linear_features, reset_mechanism, weight_init, dtype):
        super().__init__()

        self.num_steps = num_steps
        self.dtype = dtype

        self.lif1 = snn.RSynaptic(alpha=alpha, beta=beta, linear_features=LIF_linear_features, reset_mechanism=reset_mechanism)

    def forward(self, x):
        spk1, syn1, mem1 = self.lif1.init_rsynaptic()
        spk1, syn1, mem1 = spk1.to(self.dtype), syn1.to(self.dtype), mem1.to(self.dtype)

        spk1_rec = []
        syn1_rec = []
        mem1_rec = []

        for step in range(self.num_steps):
            spk1, syn1, mem1 = self.lif1(x[:,step,:], spk1, syn1, mem1)

            spk1_rec.append(spk1)
            syn1_rec.append(syn1)
            mem1_rec.append(mem1)

        # convert lists to tensors
        spk1_rec = torch.stack(spk1_rec)
        syn1_rec = torch.stack(syn1_rec)
        mem1_rec = torch.stack(mem1_rec)
        spk1_rec = torch.swapaxes(spk1_rec, 0, 1)
        syn1_rec = torch.swapaxes(syn1_rec, 0, 1)
        mem1_rec = torch.swapaxes(mem1_rec, 0, 1)

        return spk1_rec, mem1_rec
    
class SNN2(nn.Module):
    # With the first Linear layer
    def __init__(self, num_steps, beta, alpha, LIF_linear_features, reset_mechanism, weight_init, dtype):
        super().__init__()

        self.num_steps = num_steps
        self.dtype = dtype

        self.fc1 = nn.Linear(LIF_linear_features, LIF_linear_features, bias=True)
        self.lif1 = snn.RLeaky(beta=beta, linear_features=LIF_linear_features, reset_mechanism=reset_mechanism) # also experiment with all_to_all and V (weights) parameters

    def forward(self, x):
        spk1, mem1 = self.lif1.init_rleaky()
        spk1, mem1 = spk1.to(self.dtype), mem1.to(self.dtype)

        spk1_rec = []
        mem1_rec = []

        for step in range(self.num_steps):
            y = self.fc1(x[:,step,:])
            spk1, mem1 = self.lif1(y, spk1, mem1)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

        # convert lists to tensors
        spk1_rec = torch.stack(spk1_rec)
        mem1_rec = torch.stack(mem1_rec)
        spk1_rec = torch.swapaxes(spk1_rec, 0, 1)
        mem1_rec = torch.swapaxes(mem1_rec, 0, 1)

        return spk1_rec, mem1_rec
    
class SNN2syn(nn.Module):
    # With the first Linear layer
    def __init__(self, num_steps, beta, alpha, LIF_linear_features, reset_mechanism, weight_init, dtype):
        super().__init__()

        self.num_steps = num_steps
        self.dtype = dtype

        self.fc1 = nn.Linear(LIF_linear_features, LIF_linear_features, bias=True)
        self.lif1 = snn.RSynaptic(alpha=alpha, beta=beta, linear_features=LIF_linear_features, reset_mechanism=reset_mechanism)

    def forward(self, x):
        spk1, syn1, mem1 = self.lif1.init_rsynaptic()
        spk1, syn1, mem1 = spk1.to(self.dtype), syn1.to(self.dtype), mem1.to(self.dtype)

        spk1_rec = []
        syn1_rec = []
        mem1_rec = []

        for step in range(self.num_steps):
            y = self.fc1(x[:,step,:])
            spk1, syn1, mem1 = self.lif1(y, spk1, syn1, mem1)

            spk1_rec.append(spk1)
            syn1_rec.append(syn1)
            mem1_rec.append(mem1)

        # convert lists to tensors
        spk1_rec = torch.stack(spk1_rec)
        syn1_rec = torch.stack(syn1_rec)
        mem1_rec = torch.stack(mem1_rec)
        spk1_rec = torch.swapaxes(spk1_rec, 0, 1)
        syn1_rec = torch.swapaxes(syn1_rec, 0, 1)
        mem1_rec = torch.swapaxes(mem1_rec, 0, 1)

        return spk1_rec, mem1_rec


class SNN1_Supervised_Extension(nn.Module):
    # Without the first Linear layer
    def __init__(self, num_steps, beta, alpha, LIF_linear_features, reset_mechanism, weight_init, dtype):
        super().__init__()

        self.num_steps = num_steps
        self.dtype = dtype

        self.lif1 = snn.RLeaky(beta=beta, linear_features=LIF_linear_features, reset_mechanism=reset_mechanism) # also experiment with all_to_all and V (weights) parameters
        self.fc2 = nn.Linear(LIF_linear_features, LIF_linear_features, bias=True)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc_out = nn.Linear(LIF_linear_features, 10, bias=False)
        self.lif_out = snn.Leaky(beta=beta)
        
        if weight_init == 'normal':
            stdv = 1. / math.sqrt(LIF_linear_features)
            self.lif1.recurrent.weight.data.normal_(0,stdv)
        

    def convert_to_tensor(self, lst):
        lst = torch.stack(lst)
        return torch.swapaxes(lst, 0, 1)
    
    def forward(self, x):
        spk1, mem1 = self.lif1.init_rleaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        spk1, mem1 = spk1.to(self.dtype), mem1.to(self.dtype)
        mem2 = mem2.to(self.dtype)
        mem_out = mem_out.to(self.dtype)

        spk1_rec = []
        mem1_rec = []
        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            spk1, mem1 = self.lif1(x[:,step,:], spk1, mem1)
            fc2_out = self.fc2(spk1)
            spk2, mem2 = self.lif2(fc2_out, mem2)
            fc_out_out = self.fc_out(spk2)
            spk_out, mem_out = self.lif_out(fc_out_out, mem_out)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        # convert lists to tensors
        spk1_rec = convert_to_tensor(spk1_rec)
        mem1_rec = convert_to_tensor(mem1_rec)
        spk_out_rec = convert_to_tensor(spk_out_rec)
        mem_out_rec = convert_to_tensor(mem_out_rec)

        return spk_out_rec, mem_out_rec, spk1_rec, mem1_rec
    
# RNN from EmergentPredictiveCoding repository
# class Network(torch.nn.Module):
#     """
#     Recurrent Neural Network class containing parameters of the network
#     and computes the forward pass.
#     Returns hidden state of the network and preactivations of the units. 
#     """
#     def __init__(self, input_size: int, hidden_size: int, activation_func, weights_init=functions.init_params, prevbatch=False, conv=False, device=None):
#         super(Network, self).__init__()

#         self.input_size = input_size
#         if conv:
#             self.conv = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3)
#         self.is_conv= conv
#         self.hidden_size = hidden_size
#         self.activation_func = activation_func
#         self.W = torch.nn.Parameter(weights_init(hidden_size, hidden_size))
#         self.prevbatch = prevbatch
#         self.device = device

#     def forward(self, x, state=None, synap_trans=False, mask=None):



#         for i in range(sequence_length):
#             h, l_a = self.model(batch[i], state=h) # l_a is now a list of potential loss terms 
            
#             loss = loss + self.loss(l_a, loss_fn) 
#         state = h
#         return loss, loss.detach(), state


#         if state is None:
#             state = self.init_state(x.shape[0])
#         h = state
#         h = h.to(self.device)
#         x = x.to(self.device)

#         a = h @ self.W + x

#         h = self.activation_func(a)
#         # return state vector and list of losses 
#         return h, [a, h, self.W]

#     def init_state(self, batch_size):
#         return torch.zeros((batch_size, self.hidden_size))