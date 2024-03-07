import numpy as np
import matplotlib.pyplot as plt
import torch
import snntorch as snn

inp = torch.zeros((100,50))
inp[10:20,:] = 0.9
inp[50:60,:] = 0.99
inp[70:80,:] = 1

beta = 0.875 # from tau = 0.03
lif = snn.RLeaky(beta=beta, linear_features=50)
spk_rec, mem_rec = [], []; spk, mem = lif.init_rleaky()
# lif.recurrent.weight.data = torch.zeros_like(lif.recurrent.weight.data)
# lif.recurrent.bias.data = torch.zeros_like(lif.recurrent.bias.data)
for i in range(100): spk, mem = lif(inp[i,:], spk, mem); spk_rec.append(spk); mem_rec.append(mem); print(lif.recurrent.weight.data.mean())
mem_rec = torch.stack(mem_rec).detach().cpu().numpy(); spk_rec = torch.stack(spk_rec).detach().cpu().numpy()
print(len(np.where(spk_rec==1)[0]))
plt.plot(mem_rec[:,:]); plt.show()