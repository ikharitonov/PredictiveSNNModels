import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import get_model
from datasets import MNISTSequencesSupervisedDataset, NormaliseToZeroOneRange

from SNNCustomConfig import SNNCustomConfig

def run():

    device = torch.device(config.params["torch_device"])

    tau_syn = config.params["synaptic_time_constant_tao"]
    tao_mem = config.params["membrane_time_constant_tao"] # 10ms membrane time constant
    timestep = 1/config.params["dataset_sampling_frequency"]
    beta = np.exp(-timestep / tao_mem)
    alpha = np.exp(-timestep / tau_syn)

    # Data ingest and network initialisation
    targets_type = config.params["training_targets_type"]
    normalise_transform = NormaliseToZeroOneRange(dtype=dtype)
    mnist_dataset = MNISTSequencesSupervisedDataset(config.dirs["training"], config.dirs[targets_type], config.params["LIF_linear_features"], ingest_numpy_dtype, ingest_torch_dtype, transform=normalise_transform)
    train_loader = DataLoader(mnist_dataset, batch_size=config.params["batch_size"], shuffle=True, num_workers=config.params["dataloader_num_workers"])

    config.data_shape = next(iter(train_loader)).shape

    model_class = get_model(config.model_name)
    model = model_class(num_steps=config.data_shape[1], beta=beta, alpha=alpha, LIF_linear_features=config.params["LIF_linear_features"], reset_mechanism=config.params["reset_mechanism"], weight_init=config.params["weight_init"], dtype=dtype).to(device).to(dtype)
    
    # Loading weights from the classification task
    fc2_path = Path.home() / 'RANCZLAB-NAS/iakov/produced/classification_task_weights/fc2.npy'
    fc_out_path = Path.home() / 'RANCZLAB-NAS/iakov/produced/classification_task_weights/fc_out.npy'
    model.fc2.weight.data = torch.Tensor(np.load(fc2_path)).to(device).to(dtype)
    model.fc_out.weight.data = torch.Tensor(np.load(fc_out_path)).to(device).to(dtype)
    
    # Freezing weights of the deep part of the network
    model.fc2.weight.requires_grad_(False)
    model.fc_out.weight.requires_grad_(False)

    loss = nn.L1Loss()
    optimizer = config.get_optimizer(model)
    scheduler = config.get_scheduler(optimizer)
        
    max_grad_norm = config.params["grad_clipping_max_norm"]  # Define the maximum gradient norm threshold


    # TRAINING LOOP
    model, optimizer, checkpoint = config.iteration_begin_step(model, optimizer)

    loss_hist = []
    batch_loss = 0

    # Define rheobase and input multipliers
    rheobase_multiplier = 1
    if config.params["rheobase_apply"] == "true": 
        rheobase_multiplier = config.params["rheobase_multiplier"]
        print(f"Applying rheobase multiplier = {rheobase_multiplier}")
    input_multiplier = config.params["input_multiplier"]
    print(f"Applying input multiplier = {input_multiplier}")
    print(f"Combined multiplier = {rheobase_multiplier * input_multiplier}")

    for epoch in range(config.start_epoch, config.params["epochs"]):
        
        config.epoch_begin_step()
        model.train(True)

        # Minibatch training loop
        for batch_i, batch in enumerate(tqdm(train_loader, f'Epoch {epoch} Loss {batch_loss}')):

            data = batch[0].to(device)
            targets = batch[1].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            model_outputs = model(data * rheobase_multiplier * input_multiplier)
            spk_out_rec, mem_out_rec, spk1_rec, mem1_rec = model_outputs

            batch_loss = loss(spk_out_rec, targets)

            # gradient calculation and weight update
            batch_loss.backward()
            if config.params["grad_clipping"] == "true": nn.utils.clip_grad_value_(model.parameters(), max_grad_norm)
            optimizer.step()

            loss_hist.append(batch_loss.item())
            config.batch_end_step(epoch, batch_i, batch_loss, optimizer, scheduler)
        config.epoch_end_step(epoch, batch_loss, optimizer, scheduler, model, loss_hist)
    config.iteration_end_step()

if __name__ == '__main__':
    
    # Setting random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    ingest_torch_dtype = torch.uint8
    ingest_numpy_dtype = np.uint8

    dtype = torch.float32
    
    config = SNNCustomConfig(cli_args=sys.argv)
    # config = SNNCustomConfig(model_name="SNN1", dataset_name="mnist_simplest_sequence", configuration_name="config1_rheo1", configuration_file="test.txt", continue_training=True)

    run()