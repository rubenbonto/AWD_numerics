import numpy as np
import torch
import torch.optim as optim
import ot  # Optimal transport library
from nets import DQN
from utils import Memory, optimize_model

import os
import sys

import time

CD_path = os.path.abspath('/Users/rubenbontorno/Documents/Master_Thesis/Code/AWD_numerics/Conditional_density')

if CD_path not in sys.path:
    sys.path.append(CD_path)

from CD_knn_NerualNet import train_conditional_density, evaluate_conditional_density

"""
This code is adapted from the paper:

"Fitted Value Iteration Methods for Bicausal Optimal Transport"
by Erhan Bayraktar and Bingyan Han (2023).

Reference: https://arxiv.org/abs/2306.12658

The original code can be found on GitHub: https://github.com/hanbingyan/FVIOT.

Modifications were made to remove the assumption of knowing the conditional density.
"""



def train_dqn_instance(x_dim, y_dim, time_horizon, samplepath_x, samplepath_y,
                       n_opt, in_sample_size,
                       device, discount=1, mem_size=3000, trunc_flag=True):
    """
    Trains a single instance of a DQN for conditional density estimation.
    
    Parameters:
        x_dim (int): Dimension of x.
        y_dim (int): Dimension of y.
        time_horizon (int): Total number of time steps.
        samplepath_x (torch.Tensor): Tensor of shape (smp_size, time_horizon+1, x_dim) containing x sample paths.
        samplepath_y (torch.Tensor): Tensor of shape (smp_size, time_horizon+1, y_dim) containing y sample paths.
        n_opt (int): Number of gradient descent steps to perform per time step.
        in_sample_size (int): Sample size for empirical optimal transport estimation.
        cond_density_x_func (callable): A function that takes (current_x, sample_size) and returns a tensor of shape (sample_size, x_dim).
        cond_density_y_func (callable): A function that takes (current_y, sample_size) and returns a tensor of shape (sample_size, y_dim).
        device (torch.device): The device to run on.
        discount (float): Discount factor.
        mem_size (int): Memory size for the replay memory (default 1000).
        trunc_flag (bool): If True, clip network parameters after each optimization step.
        
    Returns:
        final_value (float): The estimated value at time 0.
        val_hist (np.ndarray): Array of estimated values for each time step.
        loss_hist (np.ndarray): Array of average losses for each time step.
    """
    # Initialize replay memory and networks
    memory = Memory(mem_size)
    policy_net = DQN(x_dim, y_dim, time_horizon).to(device)
    target_net = DQN(x_dim, y_dim, time_horizon).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
    
    smp_size = samplepath_x.shape[0]
    val_hist = np.zeros(time_horizon + 1)
    loss_hist = np.zeros(time_horizon + 1)
    
    # Loop backward in time
    for t in range(time_horizon, -1, -1):
        # Example: use t=1 as X and t=2 as Y (this is a simplification)
        X = np.expand_dims(samplepath_x[:, t-1], axis=1)  # shape: (num_paths, 1)
        Y = np.expand_dims(samplepath_x[:, t], axis=1)  # shape: (num_paths, 1)
        d_X = 1
        d_Y = 1
        k = 55

        # Concatenate to form a data tensor: first column(s) for X and the remaining for Y.
        data = np.concatenate([X, Y], axis=1)  # shape: (num_paths, 2)

        # Convert to a PyTorch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        x_estimator, x_loss_hist, x_n_nan = train_conditional_density(data_tensor, d_X=d_X, d_Y=d_Y, k=60,
                                                             n_iter=650, n_batch=50, lr=1e-3, nns_type=' ')

        # Example: use t=1 as X and t=2 as Y (this is a simplification)
        X = np.expand_dims(samplepath_y[:, t-1], axis=1) # shape: (num_paths, 1)
        Y = np.expand_dims(samplepath_y[:, t], axis=1)  # shape: (num_paths, 1)
        d_X = 1
        d_Y = 1
        k = 55

        # Concatenate to form a data tensor: first column(s) for X and the remaining for Y.
        data = np.concatenate([X, Y], axis=1)  # shape: (num_paths, 2)

        # Convert to a PyTorch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        y_estimator, y_loss_hist, y_n_nan = train_conditional_density(data_tensor, d_X=d_X, d_Y=d_Y, k=60,
                                                             n_iter=640, n_batch=50, lr=1e-3, nns_type=' ')      


        for smp_id in range(smp_size):
            
            # Construct batches for OT computation.
            # x_batch is obtained by repeating next_x in each row,
            # y_batch is created by tiling next_y down the rows.
            # Get the evaluated density (assumes it's a NumPy array)
            x_vals = evaluate_conditional_density(x_estimator, samplepath_x[smp_id, t-1], B=in_sample_size)
            y_vals = evaluate_conditional_density(y_estimator, samplepath_y[smp_id, t-1], B=in_sample_size)

            # Convert to PyTorch tensor and ensure correct shape [50, 1]
            next_x = torch.tensor(x_vals, dtype=torch.float32).reshape(-1, 1)
            next_y = torch.tensor(y_vals, dtype=torch.float32).reshape(-1, 1)

            x_batch = torch.repeat_interleave(next_x, repeats=in_sample_size, dim=0)
            y_batch = torch.tile(next_y, (in_sample_size, 1))
            l2_mat = torch.sum((x_batch - y_batch)**2, dim=1) 


            
            if t == time_horizon:
                expected_v = 0.0
            elif t == time_horizon - 1:
                # Compute the empirical OT cost between next states.
                min_obj = l2_mat.reshape(in_sample_size, in_sample_size)
                expected_v = ot.emd2(np.ones(in_sample_size)/in_sample_size,
                                     np.ones(in_sample_size)/in_sample_size,
                                     min_obj.detach().cpu().numpy())
            else:
                # Evaluate the target network for the next time step.
                time_tensor = torch.ones(x_batch.shape[0], 1, device=device) * (t + 1.0)
                val = target_net(time_tensor, x_batch, y_batch).reshape(-1)
                min_obj = (l2_mat + discount * val).reshape(in_sample_size, in_sample_size)
                expected_v = ot.emd2(np.ones(in_sample_size)/in_sample_size,
                                     np.ones(in_sample_size)/in_sample_size,
                                     min_obj.detach().cpu().numpy())
            
            # Push the transition into memory.
            memory.push(torch.tensor([t], dtype=torch.float32, device=device),
                        samplepath_x[smp_id, t],
                        samplepath_y[smp_id, t],
                        torch.tensor([expected_v], device=device))
            


        
        # Optimize the policy network for n_opt steps.
        for _ in range(n_opt):
            loss = optimize_model(policy_net, memory, optimizer, trunc_flag)
            if trunc_flag:
                with torch.no_grad():
                    for param in policy_net.parameters():
                        param.clamp_(-1.0, 1.0)
            if loss is not None:
                loss_hist[t] += loss.detach().cpu().item()
        loss_hist[t] /= n_opt

        
        # Update the target network.
        target_net.load_state_dict(policy_net.state_dict())
        
        x_input = torch.tensor(samplepath_x[0, 0], dtype=torch.float32).reshape(1, x_dim)
        y_input = torch.tensor(samplepath_y[0, 0], dtype=torch.float32).reshape(1, y_dim)

        # Test the value at time 0 using the first sample.
        val = target_net(torch.ones(1, 1, device=device) * 0.0, x_input, y_input).reshape(-1)
        val_hist[t] = val.item()

        # Free GPU memory used by the x_estimator
        del x_estimator
        torch.cuda.empty_cache()

        del y_estimator
        torch.cuda.empty_cache()
        
        
        memory.clear()
        print('Time step', t, 'Loss:', loss_hist[t])
    
    print('Final value at time 0:', val_hist[0])
    return val_hist[0], val_hist, loss_hist