import os
import numpy as np
import torch
from IPython.display import clear_output
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path): os.makedirs(path)
    return path

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad

    
    
class SNNModel():
    def __init__(self, batch_size, hidden_units, num_timesteps, step_length, device, dtype):
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.num_timesteps = num_timesteps
        self.step_length = step_length
        self.device = device
        self.dtype = dtype
        
        self.spike_fn = SurrGradSpike.apply # overwrite native nonlinearity by the surrogate gradient
    
    def init_parameters(self):
        tau_mem = 10e-3
        tau_syn = 5e-3

        self.alpha = float(np.exp(-self.step_length/tau_syn))
        self.beta = float(np.exp(-self.step_length/tau_mem))
        
        self.mem_spike_threshold = 1
        
        weight_scale = 0.2
        # weight_scale = 10
        
        self.v1 = torch.empty((self.hidden_units, self.hidden_units), device=self.device, dtype=self.dtype, requires_grad=True) # recurrent connections matrix
        torch.nn.init.normal_(self.v1, mean=0.0, std=weight_scale/np.sqrt(self.hidden_units))

    def load_weights(self, weights_matrix):
        self.v1 = torch.Tensor(weights_matrix)
        
    def run_snn(self, inputs):
        syn = torch.zeros((self.batch_size, self.hidden_units), device=self.device, dtype=self.dtype)
        mem = torch.zeros((self.batch_size, self.hidden_units), device=self.device, dtype=self.dtype)

        mem_rec = []
        spk_rec = []

        # Compute hidden layer activity
        out = torch.zeros((self.batch_size, self.hidden_units), device=self.device, dtype=self.dtype)
        # h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))
        for t in range(self.num_timesteps):
            thresholded_mem = mem - self.mem_spike_threshold
            out = self.spike_fn(thresholded_mem) # surrogate heaviside

            new_syn = self.alpha * syn + inputs[:,t,:] + torch.einsum("ab,bc->ac", (out, self.v1)) # input are unweighted
            new_mem = (self.beta * mem + syn) * (self.mem_spike_threshold - out.detach()) # .detach() because we don't want to backprop through the reset

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec,dim=1)
        spk_rec = torch.stack(spk_rec,dim=1)

        return mem_rec, spk_rec

    def L1_loss_function(self, x):
        return torch.mean(torch.abs(x))

    def save_weights(self, epoch_num, path):
        np.save(f'{mkdir(path)}/epoch{epoch_num}_v1_matrix.npy', self.v1.cpu().detach().numpy())
    
    def train(self, input_tensor, lr=1e-3, num_epochs=10, save_weights_path=None):
        
        input_tensor = input_tensor.to(self.device)
        self.num_batches = input_tensor.shape[0]//self.batch_size

        params = [self.v1]
        optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))

        loss_hist = []
        for e in range(num_epochs):

            # epoch_loss = torch.zeros(1, dtype=self.dtype, requires_grad=True).to(self.device)
            epoch_loss = 0

            for batch_i in tqdm(range(self.num_batches), f'epoch {e+1}'):
                
                batch_loss = torch.zeros(1, dtype=self.dtype, requires_grad=True).to(self.device)
                
                mem_rec, spk_rec = self.run_snn(input_tensor[batch_i*self.batch_size:(batch_i+1)*self.batch_size,:,:])

                batch_loss = self.L1_loss_function(mem_rec)
                epoch_loss = epoch_loss + batch_loss.item()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            epoch_loss = epoch_loss / self.num_batches
            loss_hist.append(epoch_loss)
            # live_plot(loss_hist)
            if save_weights_path: self.save_weights(e, save_weights_path)
            print("Epoch %i: loss=%.5f"%(e+1, epoch_loss))

        return loss_hist

def live_plot(loss):
    if len(loss) == 1:
        return
    clear_output(wait=True)
    ax = plt.figure(figsize=(3,2), dpi=150).gca()
    ax.plot(range(1, len(loss) + 1), loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.get_major_locator().set_params(integer=True)
    # sns.despine()
    plt.show()