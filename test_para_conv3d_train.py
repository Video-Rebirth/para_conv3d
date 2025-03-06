#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys, time, traceback
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from parallel_conv3d import ParaConv3D
from copy import deepcopy
import numpy as np
import random

import logging, warnings
warnings.filterwarnings('ignore')

# set format
logging.basicConfig(
                    level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    handlers=[logging.StreamHandler(stream=sys.stdout)])

# set seed for Python random module
random.seed(42)
# set seed for NumPy
np.random.seed(42) # for debugging bp errors

torch.manual_seed(3)
torch.cuda.manual_seed(33)
torch.cuda.manual_seed_all(333)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
device = f"cuda:{local_rank}"

torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

try:
    dist.init_process_group(backend='nccl', 
                                init_method="env://")  
except RuntimeError as e:
    logging.error(f"Process group initialization failed: {e}\n")

torch.cuda.set_device(device) 

group = dist.group.WORLD # ***
group_size = dist.get_world_size(group)
logging.info(f"\nworld_size: {world_size},  rank: {rank},  local_rank: {local_rank},  device: {device}") 

# example Conv3D net
in_channels = 3
out_channels = 256
kernel_size = (3, 3, 5)
stride = (2, 4, 4)
padding = (5, 3, 1)
param_async = False

dhw_shape = (128, 256, 256)
para_dim_index = 3

lr = 1e-4
num_epochs = 10  # Number of epochs for training

batch_size = 1
num_classes = 10
# dtype = torch.float64
# dtype = torch.float32
dtype = torch.bfloat16

samples_list = []
for epoch in range(num_epochs):
    torch.manual_seed(epoch+3)
    inputs = torch.randn(batch_size, in_channels, dhw_shape[0], dhw_shape[1], dhw_shape[2], 
                            dtype=dtype, requires_grad=True).to(device, non_blocking=True)
    if sum(padding) > 0:
        ### add paddings
        p3d = (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
        inputs = torch.nn.functional.pad(inputs, p3d, "constant", 0)
    
    labels = torch.randint(0, num_classes, (batch_size, ), dtype=torch.long).to(device)
    
    samples_list.append([inputs, labels])

# reference
# https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
dim2_after_conv = (dhw_shape[0] + 2*padding[0] - kernel_size[0]) // stride[0] + 1
dim3_after_conv = (dhw_shape[1] + 2*padding[1] - kernel_size[1]) // stride[1] + 1
dim4_after_conv = (dhw_shape[2] + 2*padding[2] - kernel_size[2]) // stride[2] + 1

dim2_after_conv = (dim2_after_conv - kernel_size[0]) // stride[0] + 1
dim3_after_conv = (dim3_after_conv - kernel_size[1]) // stride[1] + 1
dim4_after_conv = (dim4_after_conv - kernel_size[2]) // stride[2] + 1

fc_dim = out_channels * dim2_after_conv * dim3_after_conv * dim4_after_conv

class ParaConv3DNet(nn.Module):
    def __init__(self, pg: dist.ProcessGroup = None, para_deg=1, para_dim_index=4):
        super(ParaConv3DNet, self).__init__()
        self.conv1 = ParaConv3D(pg=group, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                bias=False, dtype=dtype,
                                para_deg=para_deg, para_dim_index=para_dim_index, param_async=param_async)
        
        self.conv2 = ParaConv3D(pg=group, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                bias=False, dtype=dtype,
                                para_deg=para_deg, para_dim_index=para_dim_index, param_async=param_async)
        
        self.fc1 = nn.Linear(fc_dim, num_classes, bias=False, dtype=dtype)  # Adjust input size according to the output of conv layers
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1, fc_dim)
        x = self.fc1(x)

        return x


def main():
    ### no parallelism, benchmark
    vanilla_conv3d = ParaConv3DNet(pg=group, para_deg=1, para_dim_index=para_dim_index).to(device).to(device)
    vanilla_conv3d.train()

    ### parallel conv3d
    para_conv3d = ParaConv3DNet(pg=group, para_deg=world_size, para_dim_index=para_dim_index).to(device)
    para_conv3d.conv1.conv3d = deepcopy(vanilla_conv3d.conv1.conv3d) # to use the same initialization
    para_conv3d.conv2.conv3d = deepcopy(vanilla_conv3d.conv2.conv3d)

    para_conv3d.fc1 = deepcopy(vanilla_conv3d.fc1)
    para_conv3d.train()

    vanilla_outputs = []
    vanilla_loss = []
    vanilla_gradients = []

    para_outputs = []
    para_loss = []
    para_gradients = []
        
    ### parallel Conv3D
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(para_conv3d.parameters(), lr=lr)  # Adam optimizer
    for epoch in tqdm(range(num_epochs)):  # Loop over the dataset multiple times
        inputs, labels = samples_list[epoch]

        # zero out gradients
        optimizer.zero_grad()

        # forward
        outputs = para_conv3d(inputs)
        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()
        
        para_outputs.append(outputs.detach().cpu())
        para_loss.append(loss.item())

        for name, param in para_conv3d.named_parameters():
            if name == "conv1.conv3d.weight":
                para_gradients.append(param.grad)
    
    # for benchmarking   
    if rank == 0:
        logging.info(f"\nrank, {rank},  para_loss: {para_loss}\n")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        ### vanilla Conv3D, no parallelism
        criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
        optimizer = optim.Adam(vanilla_conv3d.parameters(), lr=lr)  # Adam optimizer
        for epoch in tqdm(range(num_epochs)):  # Loop over the dataset multiple times
            inputs, labels = samples_list[epoch]

            optimizer.zero_grad()
            outputs = vanilla_conv3d(inputs)

            with torch.autocast(device_type="cuda", dtype=torch.float64):
                # outputs = outputs.to(torch.float32)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            vanilla_outputs.append(outputs.detach().cpu())
            vanilla_loss.append(loss.item())

            for name, param in vanilla_conv3d.named_parameters():
                if name == "conv1.conv3d.weight":
                    # logging.info(f"rank: {rank},  vanilla conv3d,  param.grad.shape: {param.grad.shape},  torch.sum(torch.abs(param.grad)): {torch.sum(torch.abs(param.grad))}")
                    vanilla_gradients.append(param.grad)

        logging.info(f"\nrank, {rank},  vanilla_loss: {vanilla_loss}")

        # are the results the same?
        rtol=1e-4
        atol=1e-7

        vanilla_loss = torch.tensor(vanilla_loss)
        para_loss = torch.tensor(para_loss)
        try:
            torch.testing.assert_close(vanilla_loss, para_loss, rtol=rtol, atol=atol)
            logging.info("\nThe losses from vanilla Conv3D Net and the losses from ParaConv3D Net are allclose.\n")
        except AssertionError:
            traceback.print_exc()
            print()

        vanilla_gradients = torch.cat(vanilla_gradients, dim=0)
        para_gradients = torch.cat(para_gradients, dim=0)
        try:
            torch.testing.assert_close(vanilla_gradients, para_gradients, rtol=rtol, atol=atol)
            logging.info("The gradients from vanilla Conv3D Net and the gradients from ParaConv3D Net are allclose.\n")
        except AssertionError:
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()  
    logging.info(f"\nrank: {rank}, finished")
