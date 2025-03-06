# -*- coding: UTF-8 -*-

import os, sys
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple

import logging, warnings
warnings.filterwarnings('ignore')

# set format
logging.basicConfig(
                    level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    handlers=[logging.StreamHandler(stream=sys.stdout)])

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

RANK = int(os.environ['RANK'])

# find the indices for spliting the input
def get_split_indices(group_size:int = 0, org_dim_size:int = 0, kernel_size:int = 0, stride:int = 0):
    stride_index = [i for i in range(0, org_dim_size - (kernel_size - 1), stride)] 
    conv_out_size = (org_dim_size - kernel_size) // stride + 1
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    
    try:
        assert conv_out_size == len(stride_index)
    except AssertionError:
        logging.error(f"\nout_size {out_size} not equal len(stride_index) {len(stride_index)}\n")
        sys.exit(1)
    
    input_split_indices = []
    output_split_indices = []
    temp_idx = 0
    conv_out_idx = 0
    out_size_per_rank, rmder = divmod(conv_out_size, group_size)
    for rk in range(group_size):
        start_idx = stride_index[temp_idx]
        temp_idx += out_size_per_rank
        last_idx = stride_index[temp_idx - 1] # take the out_size_per_rank elements
        if (rmder > 0) and (rk < group_size - 1):
            conv_out_size -= out_size_per_rank
            out_size_per_rank, rmder = divmod(conv_out_size, group_size - 1 - rk)

        temp_offset = kernel_size if kernel_size >= stride else stride
        end_idx = last_idx + temp_offset if rk < group_size - 1 else org_dim_size

        input_split_indices.append([start_idx, end_idx])

        size_after_conv = (end_idx - start_idx - kernel_size) // stride + 1
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        output_split_indices.append([conv_out_idx, conv_out_idx + size_after_conv]) # after convolution
        conv_out_idx += size_after_conv
    
    # logging.info(f"rank: {RANK},  input_split_indices: {input_split_indices},  output_split_indices: {output_split_indices}")
    return input_split_indices, output_split_indices

def _split(input_ = None, para_dim_index: int = 4, split_indices: List = None):
    global RANK
    input_ = input_.contiguous()
    start_idx, end_idx = split_indices[RANK]
    if para_dim_index == 2:
        split_results = input_[:, :, start_idx:end_idx, :, :].contiguous() 
    elif para_dim_index == 3:
        split_results = input_[:, :, :, start_idx:end_idx, :].contiguous() 
    elif para_dim_index == 4:
        split_results = input_[:, :, :, :, start_idx:end_idx].contiguous()
    else:
        logging.error(f"\nERROR: wrong para_dim_index: {para_dim_index}\n")
        sys.exit(1)

    return split_results

def _gather(input_ = None, pg: dist.ProcessGroup = None, para_dim_index: int = 4, split_indices: List = None, is_reshape=False):
    input_ = input_.contiguous()
    # logging.info(f"rank: {RANK},  input_.shape: {input_.shape}")
    # group_size = dist.get_world_size(pg)
    # split_indices = get_split_indices(group_size, org_total_size, kernel_size, stride)
    tensor_list = []
    # collect the results from multiple GPU
    # build the shapes after convolution
    for start_idx, end_idx in split_indices:
        temp_span = end_idx - start_idx
        if para_dim_index == 2:
            tensor_list.append(torch.empty_like(input_[:, :, 0:1, :, :].expand(-1, -1, temp_span, -1, -1)))
        elif para_dim_index == 3:
            tensor_list.append(torch.empty_like(input_[:, :, :, 0:1, :].expand(-1, -1, -1, temp_span, -1)))
        elif para_dim_index == 4:
            tensor_list.append(torch.empty_like(input_[:, :, :, :, 0:1].expand(-1, -1, -1, -1, temp_span)))
        else:
            logging.error(f"\nERROR: wrong para_dim_index: {para_dim_index}\n")
            sys.exit(1)
    
    # torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)
    # dist.all_gather(tensor_list, input_, group=pg, async_op=True)  ### error with async_op=True
    # dist.barrier()
    dist.all_gather(tensor_list, input_, group=pg, async_op=False)  ### note the difference between NCCL and GLOO

    # put everything into one tensor
    output = torch.cat(tensor_list, dim=para_dim_index).contiguous()

    """
    # TODO: gather only at rank = 0
    # torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False), 
    # This function requires all tensors to be the same size on each process.
    if RANK == 0:
        dist.gather(input_, tensor_list, group=pg) 
        output = torch.cat(tensor_list, dim=dim).contiguous()
    else:
        dist.gather(input_, group=pg) 
        output = input_
    """ 
    org_total_size = split_indices[-1][-1]
    if is_reshape:
        # deal with overlapping with convolution/kernel_size
        if para_dim_index == 2:
            real_output = torch.zeros_like(input_[:, :, 0:1, :, :].expand(-1, -1, org_total_size, -1, -1))
            for temp_tensor, idx in zip(tensor_list, split_indices):
                start_idx, end_idx = idx
                for i in range(start_idx, end_idx):
                    j = i - start_idx
                    real_output[:, :, i, :, :] = real_output[:, :, i, :, :] + temp_tensor[:, :, j, :, :]
        elif para_dim_index == 3:
            real_output = torch.zeros_like(input_[:, :, :, 0:1, :].expand(-1, -1, -1, org_total_size, -1))
            for temp_tensor, idx in zip(tensor_list, split_indices):
                start_idx, end_idx = idx
                for i in range(start_idx, end_idx):
                    j = i - start_idx
                    real_output[:, :, :, i, :] = real_output[:, :, :, i, :] + temp_tensor[:, :, :, j, :]
        elif para_dim_index == 4:
            real_output = torch.zeros_like(input_[:, :, :, :, 0:1].expand(-1, -1, -1, -1, org_total_size))
            for temp_tensor, idx in zip(tensor_list, split_indices):
                start_idx, end_idx = idx
                for i in range(start_idx, end_idx):
                    j = i - start_idx
                    real_output[:, :, :, :, i] = real_output[:, :, :, :, i] + temp_tensor[:, :, :, :, j]
        else:
            logging.error(f"\nwrong para_dim_index: {para_dim_index}\n")
            sys.exit(1)
 
        """
        if RANK == 0:
            logging.info(f"rank: {RANK},  output.shape: {output.shape},  torch.sum(torch.abs(output)): {torch.sum(torch.abs(output))}")
            logging.info(f"rank: {RANK},  real_output.shape: {real_output.shape},  torch.sum(torch.abs(real_output)): {torch.sum(torch.abs(real_output))}")
        """
        output = real_output

    return output

class AllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, conv3d_module, param_async, grad_reduce_handles):
        ctx.grad_reduce_handles = grad_reduce_handles
        ctx.param_async = param_async
        ctx.conv3d = conv3d_module
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        for param in ctx.conv3d.parameters():
            if param.grad is not None:
                if ctx.param_async:
                    # logging.info(f"Please remember to call work.wait() to ensure async operations complete.")
                    handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True) 
                    ctx.grad_reduce_handles.append(handle)
                else:
                    # dist.barrier()
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM) 
                    # dist.all_reduce(param, op=dist.ReduceOp.AVG)
                    # logging.info(f"\nrank: {RANK}, I am doing all-reduce of gradients.  param.grad.shape: {param.grad.shape},  torch.sum(torch.abs(param.grad)): {torch.sum(torch.abs(param.grad))}\n")
        return grad_output, None, None, None, None, None, None

# split input tensor x
class _ConvSplitForwardGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_ = None, process_group: dist.ProcessGroup = None, para_dim_index: int = 4, split_indices: List = None, is_reshape=False):
        ctx.pg = process_group
        ctx.para_dim_idx = para_dim_index
        ctx.split_indices = split_indices
        ctx.is_reshape = is_reshape
        output = _split(input_, para_dim_index, split_indices) # split inputs x
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = _gather(grad_output, ctx.pg, ctx.para_dim_idx, ctx.split_indices, is_reshape=ctx.is_reshape) # gather gradients of x
        return output, None, None, None, None, None, None


# gather convolution outputs
class _ConvGatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_ = None, process_group: dist.ProcessGroup = None, para_dim_index: int = 4, split_indices: List = None):
        ctx.pg = process_group
        ctx.para_dim_idx = para_dim_index
        ctx.split_indices = split_indices
        output = _gather(input_, process_group, para_dim_index, split_indices) # gather convolution outputs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # logging.info(f"rank: {RANK},  ctx.split_indices: {ctx.split_indices}")
        output = _split(grad_output, ctx.para_dim_idx, ctx.split_indices) # split gradients of convolution outputs
        return output, None, None, None, None, None, None


class ParaConv3D(nn.Module):
    def __init__(self,
                 pg: dist.ProcessGroup,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple = (1, 1, 1),
                 stride: Tuple = (1, 1, 1),
                 bias=True,
                 dtype=torch.float32,
                 para_deg: int = 1,
                 para_dim_index: int = 4,
                 param_async=False):
        super(ParaConv3D, self).__init__()

        self.pg = pg

        self.para_dim_index = para_dim_index

        self.para_kernel_size = kernel_size[self.para_dim_index - 2] # in the parallel computing dimension
        self.para_stride = stride[self.para_dim_index - 2]
 
        self.para_deg = para_deg
        self.group_size = dist.get_world_size(pg)

        """
        try:
            assert self.para_deg == self.group_size
        except AssertionError:
            logging.error(f"\nself.para_deg {self.para_deg} and self.group_size {self.group_size} are not equal.\n")
        """

        self.input_split_indices = []
        self.output_split_indices = []

        self.is_reshape = self.para_kernel_size > self.para_stride ### any overlapping

        self.param_async = param_async
        self.grad_reduce_handles = []

        # reuse torch.nn.Conv3D
        self.conv3d = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0, # no longer need to consider padding here; paddings have already been added to inputs
            bias=bias, 
            dtype=dtype
        )

    def forward(self, x):  
        if self.para_deg > 1:
            para_dim_size = x.shape[self.para_dim_index] # [batch_size, in_channels, depth, height, width]
    
            size_after_conv = (para_dim_size - self.para_kernel_size) // self.para_stride + 1
            size_per_rank = size_after_conv // self.group_size
            if size_per_rank < 1:
                logging.error(f"\nsize_per_rank after convolution is 0. size_after_conv: {size_after_conv}, size_per_rank after convolution: {size_per_rank}\n")
                sys.exit(1)
            
            self.input_split_indices, self.output_split_indices = get_split_indices(self.group_size, para_dim_size, self.para_kernel_size, self.para_stride)

            x = AllReduceFunction.apply(x, self.conv3d, self.param_async, self.grad_reduce_handles)

            x = _ConvSplitForwardGatherBackward.apply(x, self.pg, self.para_dim_index, self.input_split_indices, self.is_reshape)
            # Conv3D with split x
        
        x = self.conv3d(x) 

        if self.para_deg > 1:
            x = _ConvGatherForwardSplitBackward.apply(x, self.pg, self.para_dim_index, self.output_split_indices)
        
        return x

    def get_param_grad_reduce_handles(self):
        return self.grad_reduce_handles

    # Synchronization: Always call work.wait() to ensure async operations complete.
    def wait_param_grad_reduce_handles(self):
        for handle in self.grad_reduce_handles:
            handle.wait()
        self.grad_reduce_handles = []



"""
References:
https://gitee.com/ascend/MindSpeed/blob/core_r0.8.0/mindspeed/multi_modal/conv3d/conv3d_depth_parallel.py
https://gitee.com/ascend/MindSpeed/blob/7b8f3c19d0efea459c3b438063d81c0183ebdd8c/docs/features/conv3d_sequence_parallel.md
https://gitee.com/ascend/MindSpeed/pulls/1237.diff?skip_mobile=true
"""