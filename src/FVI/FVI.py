import numpy as np
import torch
import torch.optim as optim
import ot  # Optimal transport library

from sklearn.cluster import KMeans
import time


from FVI.nets import DQN
from FVI.utils import Memory, optimize_model

from Conditional_density.CD_knn_NerualNet import train_conditional_density
from Conditional_density.CD_nonparam import (
    estimate_conditional_density_one_step,
    estimate_conditional_density_two_step,
)


"""
This code is adapted from the paper:

"Fitted Value Iteration Methods for Bicausal Optimal Transport"
by Erhan Bayraktar and Bingyan Han (2023).

Reference: https://arxiv.org/abs/2306.12658

The original code can be found on GitHub: https://github.com/hanbingyan/FVIOT.

Modifications were made to remove the assumption of knowing the conditional density.
"""


# Use NN from the knn approach for CDE
def train_dqn_instance(
    x_dim,
    y_dim,
    time_horizon,
    samplepath_x,
    samplepath_y,
    n_opt,
    in_sample_size,
    device,
    discount=1,
    mem_size=3000,
    trunc_flag=True,
    n_iter=1500,
):
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
        if t < time_horizon and t > 0:

            ############ Markovian assumption for know train only on the previous step not the all past!!!!!!!!!!!!!

            X = np.expand_dims(samplepath_x[:, t], axis=1)
            Y = np.expand_dims(samplepath_x[:, t + 1], axis=1)
            d_X = 1
            d_Y = 1

            data = np.concatenate([X, Y], axis=1)

            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            x_estimator, x_loss_hist, x_n_nan = train_conditional_density(
                data_tensor,
                d_X=d_X,
                d_Y=d_Y,
                k=in_sample_size,
                n_iter=n_iter,
                n_batch=50,
                lr=1e-3,
                nns_type=" ",
                Lip=True,
            )

            ############ Markovian assumption for know train only on the previous step not the all past!!!!!!!!!!!!!

            X = np.expand_dims(samplepath_y[:, t], axis=1)
            Y = np.expand_dims(samplepath_y[:, t + 1], axis=1)
            d_X = 1
            d_Y = 1

            data = np.concatenate([X, Y], axis=1)

            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            y_estimator, y_loss_hist, y_n_nan = train_conditional_density(
                data_tensor,
                d_X=d_X,
                d_Y=d_Y,
                k=in_sample_size,
                n_iter=n_iter,
                n_batch=50,
                lr=1e-3,
                nns_type=" ",
                Lip=True,
            )

        for smp_id in range(smp_size):

            if t < time_horizon and t > 0:

                x_estimator.atomnet.to(device)
                x_estimator.atomnet.eval()
                y_estimator.atomnet.to(device)
                y_estimator.atomnet.eval()

                with torch.no_grad():
                    x0_tensor = torch.tensor(
                        [[samplepath_x[smp_id, t]]], dtype=torch.float32, device=device
                    )
                    y0_tensor = torch.tensor(
                        [[samplepath_y[smp_id, t]]], dtype=torch.float32, device=device
                    )
                    x_est = x_estimator.atomnet(x0_tensor)
                    y_est = y_estimator.atomnet(y0_tensor)

                next_x = torch.tensor(x_est, dtype=torch.float32).reshape(-1, 1)
                next_y = torch.tensor(y_est, dtype=torch.float32).reshape(-1, 1)

                x_batch = torch.repeat_interleave(next_x, repeats=in_sample_size, dim=0)
                y_batch = torch.tile(next_y, (in_sample_size, 1))
                l2_mat = torch.sum((x_batch - y_batch) ** 2, dim=1)

            if t == 0:
                indices_x = np.random.choice(
                    samplepath_x.shape[0], size=in_sample_size, replace=False
                )
                indices_y = np.random.choice(
                    samplepath_x.shape[0], size=in_sample_size, replace=False
                )
                samples_x = samplepath_x[indices_x, t + 1].reshape(
                    -1, 1
                )  # adjust reshape if x_dim > 1
                samples_y = samplepath_y[indices_y, t + 1].reshape(
                    -1, 1
                )  # adjust reshape if y_dim > 1

                # Convert centers to torch tensors
                next_x = torch.tensor(samples_x, device=device, dtype=torch.float32)
                next_y = torch.tensor(samples_y, device=device, dtype=torch.float32)

                x_batch = torch.repeat_interleave(next_x, repeats=in_sample_size, dim=0)
                y_batch = torch.tile(next_y, (in_sample_size, 1))
                l2_mat = torch.sum((x_batch - y_batch) ** 2, dim=1)

            if t == time_horizon:
                # NO CDE for this step!!!
                expected_v = 0.0
            elif t == time_horizon - 1:
                min_obj = l2_mat.reshape(in_sample_size, in_sample_size)
                expected_v = ot.emd2(
                    np.ones(in_sample_size) / in_sample_size,
                    np.ones(in_sample_size) / in_sample_size,
                    min_obj.detach().cpu().numpy(),
                )
            else:
                time_tensor = torch.ones(x_batch.shape[0], 1, device=device) * (t + 1.0)
                val = target_net(time_tensor, x_batch, y_batch).reshape(-1)
                min_obj = (l2_mat + discount * val).reshape(
                    in_sample_size, in_sample_size
                )
                expected_v = ot.emd2(
                    np.ones(in_sample_size) / in_sample_size,
                    np.ones(in_sample_size) / in_sample_size,
                    min_obj.detach().cpu().numpy(),
                )

            # Push the transition into memory.
            memory.push(
                torch.tensor([t], dtype=torch.float32, device=device),
                samplepath_x[smp_id, t],
                samplepath_y[smp_id, t],
                torch.tensor([expected_v], device=device),
            )

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

        x_input = torch.tensor(samplepath_x[0, 0], dtype=torch.float32).reshape(
            1, x_dim
        )
        y_input = torch.tensor(samplepath_y[0, 0], dtype=torch.float32).reshape(
            1, y_dim
        )

        # Test the value at time 0 using the first sample.
        val = target_net(
            torch.ones(1, 1, device=device) * 0.0, x_input, y_input
        ).reshape(-1)
        val_hist[t] = val.item()

        # Free memory (otherwise kernel crash in notebook)
        if t < time_horizon and t > 0:
            del x_estimator
            torch.cuda.empty_cache()

            del y_estimator
            torch.cuda.empty_cache()

        memory.clear()
        print("Time step", t, "Loss:", loss_hist[t])

    print("Final value at time 0:", val_hist[0])
    return val_hist[0], val_hist, loss_hist


# Use NN from the knn approach for CDE
def train_dqn_instance_mult(
    x_dim,
    y_dim,
    time_horizon,
    samplepath_x,
    samplepath_y,
    n_opt,
    in_sample_size,
    device,
    discount=1,
    mem_size=3000,
    trunc_flag=True,
    n_iter=1500,
):
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
        if t < time_horizon and t > 0:

            ############ Markovian assumption for know train only on the previous step not the all past!!!!!!!!!!!!!

            X = np.expand_dims(samplepath_x[:, 1 : t + 1], axis=1)
            Y = np.expand_dims(samplepath_x[:, t + 1], axis=1)
            d_X = t
            d_Y = 1

            # Flatten X if it has more than 2 dimensions
            if X.ndim > 2:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X

            # Ensure Y is 2D
            Y_flat = Y.reshape(Y.shape[0], -1)

            # Concatenate along the feature axis
            data = np.concatenate([X_flat, Y_flat], axis=1)

            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            x_estimator, x_loss_hist, x_n_nan = train_conditional_density(
                data_tensor,
                d_X=d_X,
                d_Y=d_Y,
                k=in_sample_size,
                n_iter=n_iter,
                n_batch=50,
                lr=1e-3,
                nns_type=" ",
                Lip=True,
                Print_res=True,
            )

            ############ Markovian assumption for know train only on the previous step not the all past!!!!!!!!!!!!!

            X = np.expand_dims(samplepath_y[:, 1 : t + 1], axis=1)
            Y = np.expand_dims(samplepath_y[:, t + 1], axis=1)
            d_X = t
            d_Y = 1

            # Flatten X if it has more than 2 dimensions
            if X.ndim > 2:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X

            # Ensure Y is 2D
            Y_flat = Y.reshape(Y.shape[0], -1)

            # Concatenate along the feature axis
            data = np.concatenate([X_flat, Y_flat], axis=1)

            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            y_estimator, y_loss_hist, y_n_nan = train_conditional_density(
                data_tensor,
                d_X=d_X,
                d_Y=d_Y,
                k=in_sample_size,
                n_iter=n_iter,
                n_batch=50,
                lr=1e-3,
                nns_type=" ",
                Lip=True,
                Print_res=True,
            )

        for smp_id in range(smp_size):

            if t < time_horizon and t > 0:

                x_estimator.atomnet.to(device)
                x_estimator.atomnet.eval()
                y_estimator.atomnet.to(device)
                y_estimator.atomnet.eval()

                with torch.no_grad():
                    x0_tensor = torch.tensor(
                        [[samplepath_x[smp_id, 1 : t + 1]]],
                        dtype=torch.float32,
                        device=device,
                    )
                    y0_tensor = torch.tensor(
                        [[samplepath_y[smp_id, 1 : t + 1]]],
                        dtype=torch.float32,
                        device=device,
                    )
                    x_est = x_estimator.atomnet(x0_tensor)
                    y_est = y_estimator.atomnet(y0_tensor)

                next_x = torch.tensor(x_est, dtype=torch.float32).reshape(-1, 1)
                next_y = torch.tensor(y_est, dtype=torch.float32).reshape(-1, 1)

                x_batch = torch.repeat_interleave(next_x, repeats=in_sample_size, dim=0)
                y_batch = torch.tile(next_y, (in_sample_size, 1))
                l2_mat = torch.sum((x_batch - y_batch) ** 2, dim=1)

            if t == 0:
                indices_x = np.random.choice(
                    samplepath_x.shape[0], size=in_sample_size, replace=False
                )
                indices_y = np.random.choice(
                    samplepath_x.shape[0], size=in_sample_size, replace=False
                )
                samples_x = samplepath_x[indices_x, t + 1].reshape(
                    -1, 1
                )  # adjust reshape if x_dim > 1
                samples_y = samplepath_y[indices_y, t + 1].reshape(
                    -1, 1
                )  # adjust reshape if y_dim > 1

                # Convert centers to torch tensors
                next_x = torch.tensor(samples_x, device=device, dtype=torch.float32)
                next_y = torch.tensor(samples_y, device=device, dtype=torch.float32)

                x_batch = torch.repeat_interleave(next_x, repeats=in_sample_size, dim=0)
                y_batch = torch.tile(next_y, (in_sample_size, 1))
                l2_mat = torch.sum((x_batch - y_batch) ** 2, dim=1)

            if t == time_horizon:
                # NO CDE for this step!!!
                expected_v = 0.0
            elif t == time_horizon - 1:
                min_obj = l2_mat.reshape(in_sample_size, in_sample_size)
                expected_v = ot.emd2(
                    np.ones(in_sample_size) / in_sample_size,
                    np.ones(in_sample_size) / in_sample_size,
                    min_obj.detach().cpu().numpy(),
                )
            else:
                time_tensor = torch.ones(x_batch.shape[0], 1, device=device) * (t + 1.0)
                val = target_net(time_tensor, x_batch, y_batch).reshape(-1)
                min_obj = (l2_mat + discount * val).reshape(
                    in_sample_size, in_sample_size
                )
                expected_v = ot.emd2(
                    np.ones(in_sample_size) / in_sample_size,
                    np.ones(in_sample_size) / in_sample_size,
                    min_obj.detach().cpu().numpy(),
                )

            # Push the transition into memory.
            memory.push(
                torch.tensor([t], dtype=torch.float32, device=device),
                samplepath_x[smp_id, t],
                samplepath_y[smp_id, t],
                torch.tensor([expected_v], device=device),
            )

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

        x_input = torch.tensor(samplepath_x[0, 0], dtype=torch.float32).reshape(
            1, x_dim
        )
        y_input = torch.tensor(samplepath_y[0, 0], dtype=torch.float32).reshape(
            1, y_dim
        )

        # Test the value at time 0 using the first sample.
        val = target_net(
            torch.ones(1, 1, device=device) * 0.0, x_input, y_input
        ).reshape(-1)
        val_hist[t] = val.item()

        # Free memory (otherwise kernel crash in notebook)
        if t < time_horizon and t > 0:
            del x_estimator
            torch.cuda.empty_cache()

            del y_estimator
            torch.cuda.empty_cache()

        memory.clear()
        print("Time step", t, "Loss:", loss_hist[t])

    print("Final value at time 0:", val_hist[0])
    return val_hist[0], val_hist, loss_hist


# Same as above but avoiding CDE for outlier values!!
def train_dqn_instance_no_outlier(
    x_dim,
    y_dim,
    time_horizon,
    samplepath_x,
    samplepath_y,
    n_opt,
    in_sample_size,
    device,
    discount=1,
    mem_size=3000,
    trunc_flag=True,
    n_iter=1500,
):
    """
    Trains a single instance of a DQN for conditional density estimation.

    Parameters:
        x_dim (int): Dimension of x.
        y_dim (int): Dimension of y.
        time_horizon (int): Total number of time steps.
        samplepath_x (np.ndarray or torch.Tensor): Array of shape (smp_size, time_horizon+1, x_dim) containing x sample paths.
        samplepath_y (np.ndarray or torch.Tensor): Array of shape (smp_size, time_horizon+1, y_dim) containing y sample paths.
        n_opt (int): Number of gradient descent steps to perform per time step.
        in_sample_size (int): Sample size for empirical optimal transport estimation.
        device (torch.device): The device to run on.
        discount (float): Discount factor.
        mem_size (int): Memory size for the replay memory.
        trunc_flag (bool): If True, clip network parameters after each optimization step.
        n_iter (int): Number of iterations for training the conditional density estimators.

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

    # ----------------------------
    # Outlier removal: For each time step (except t=0),
    # find the indices of the 10 smallest and 10 largest observations
    # in both samplepath_x and samplepath_y. Then, remove these sample paths.
    outlier_indices = set()
    for t in range(1, time_horizon + 1):
        # For samplepath_x
        values_x = samplepath_x[:, t].flatten()  # shape: (smp_size,)
        sorted_indices_x = np.argsort(values_x)
        # smallest 10 and largest 10 indices for x
        outlier_indices.update(sorted_indices_x[:10].tolist())
        outlier_indices.update(sorted_indices_x[-10:].tolist())

        # For samplepath_y
        values_y = samplepath_y[:, t].flatten()  # shape: (smp_size,)
        sorted_indices_y = np.argsort(values_y)
        # smallest 10 and largest 10 indices for y
        outlier_indices.update(sorted_indices_y[:10].tolist())
        outlier_indices.update(sorted_indices_y[-10:].tolist())

    # Create list of non-outlier indices
    non_outlier_indices = [i for i in range(smp_size) if i not in outlier_indices]

    # Create new sample paths without outliers
    samplepath_x_no_outlier = samplepath_x[non_outlier_indices, :]
    samplepath_y_no_outlier = samplepath_y[non_outlier_indices, :]
    smp_size = samplepath_x_no_outlier.shape[0]  # update sample size
    # ----------------------------

    val_hist = np.zeros(time_horizon + 1)
    loss_hist = np.zeros(time_horizon + 1)

    # Loop backward in time
    for t in range(time_horizon, -1, -1):
        if t < time_horizon:
            # Train conditional density estimators using samplepath_x and samplepath_y
            # (Note: These estimators are trained on the full paths.)
            # For x
            X = np.expand_dims(samplepath_x[:, t], axis=1)  # shape: (new_num_paths, 1)
            Y = np.expand_dims(
                samplepath_x[:, t + 1], axis=1
            )  # shape: (new_num_paths, 1)
            d_X = 1
            d_Y = 1

            data = np.concatenate([X, Y], axis=1)  # shape: (new_num_paths, 2)
            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            x_estimator, x_loss_hist, x_n_nan = train_conditional_density(
                data_tensor,
                d_X=d_X,
                d_Y=d_Y,
                k=in_sample_size,
                n_iter=n_iter,
                n_batch=50,
                lr=1e-3,
                nns_type=" ",
                Lip=True,
            )

            # For y
            X = np.expand_dims(samplepath_y[:, t], axis=1)  # shape: (new_num_paths, 1)
            Y = np.expand_dims(
                samplepath_y[:, t + 1], axis=1
            )  # shape: (new_num_paths, 1)
            d_X = 1
            d_Y = 1

            data = np.concatenate([X, Y], axis=1)  # shape: (new_num_paths, 2)
            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            y_estimator, y_loss_hist, y_n_nan = train_conditional_density(
                data_tensor,
                d_X=d_X,
                d_Y=d_Y,
                k=in_sample_size,
                n_iter=n_iter,
                n_batch=50,
                lr=1e-3,
                nns_type=" ",
                Lip=True,
            )

        # Use the outlier-filtered paths for OT computation
        for smp_id in range(smp_size):
            if t < time_horizon:
                # Set networks to evaluation mode and send to device
                x_estimator.atomnet.to(device)
                x_estimator.atomnet.eval()
                y_estimator.atomnet.to(device)
                y_estimator.atomnet.eval()

                with torch.no_grad():
                    x0_tensor = torch.tensor(
                        [[samplepath_x_no_outlier[smp_id, t]]],
                        dtype=torch.float32,
                        device=device,
                    )
                    y0_tensor = torch.tensor(
                        [[samplepath_y_no_outlier[smp_id, t]]],
                        dtype=torch.float32,
                        device=device,
                    )
                    x_est = x_estimator.atomnet(
                        x0_tensor
                    )  # assume output shape is [1, k]
                    y_est = y_estimator.atomnet(
                        y0_tensor
                    )  # assume output shape is [1, k]

                # Ensure the output is in shape [k, 1]
                next_x = torch.tensor(x_est, dtype=torch.float32).reshape(-1, 1)
                next_y = torch.tensor(y_est, dtype=torch.float32).reshape(-1, 1)

                x_batch = torch.repeat_interleave(next_x, repeats=in_sample_size, dim=0)
                y_batch = torch.tile(next_y, (in_sample_size, 1))
                l2_mat = torch.sum((x_batch - y_batch) ** 2, dim=1)

            if t == time_horizon:
                expected_v = 0.0
            elif t == time_horizon - 1:
                min_obj = l2_mat.reshape(in_sample_size, in_sample_size)
                expected_v = ot.emd2(
                    np.ones(in_sample_size) / in_sample_size,
                    np.ones(in_sample_size) / in_sample_size,
                    min_obj.detach().cpu().numpy(),
                )
            else:
                time_tensor = torch.ones(x_batch.shape[0], 1, device=device) * (t + 1.0)
                val = target_net(time_tensor, x_batch, y_batch).reshape(-1)
                min_obj = (l2_mat + discount * val).reshape(
                    in_sample_size, in_sample_size
                )
                expected_v = ot.emd2(
                    np.ones(in_sample_size) / in_sample_size,
                    np.ones(in_sample_size) / in_sample_size,
                    min_obj.detach().cpu().numpy(),
                )

            memory.push(
                torch.tensor([t], dtype=torch.float32, device=device),
                samplepath_x_no_outlier[smp_id, t],
                samplepath_y_no_outlier[smp_id, t],
                torch.tensor([expected_v], device=device),
            )

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

        x_input = torch.tensor(
            samplepath_x_no_outlier[0, 0], dtype=torch.float32
        ).reshape(1, x_dim)
        y_input = torch.tensor(
            samplepath_y_no_outlier[0, 0], dtype=torch.float32
        ).reshape(1, y_dim)
        val = target_net(
            torch.ones(1, 1, device=device) * 0.0, x_input, y_input
        ).reshape(-1)
        val_hist[t] = val.item()

        if t < time_horizon:
            del x_estimator
            torch.cuda.empty_cache()
            del y_estimator
            torch.cuda.empty_cache()

        memory.clear()
        print("Time step", t, "Loss:", loss_hist[t])

    print("Final value at time 0:", val_hist[0])
    return val_hist[0], val_hist, loss_hist


###### TO MODIFY FOR TIME STEP DO IT ONE TIME TOO MUCH ## and sampling!!


# training with the non parametirc CDE (assuming markov)
def train_dqn_instance_nonparam(
    x_dim,
    y_dim,
    time_horizon,
    samplepath_x,
    samplepath_y,
    n_opt,
    in_sample_size,
    device,
    discount=1,
    mem_size=3000,
    trunc_flag=True,
):
    """
    Trains a single instance of a DQN for conditional density estimation using `estimate_conditional_density`.

    Parameters:
        x_dim (int): Dimension of x.
        y_dim (int): Dimension of y.
        time_horizon (int): Total number of time steps.
        samplepath_x (torch.Tensor): Tensor of shape (smp_size, time_horizon+1, x_dim) containing x sample paths.
        samplepath_y (torch.Tensor): Tensor of shape (smp_size, time_horizon+1, y_dim) containing y sample paths.
        n_opt (int): Number of gradient descent steps to perform per time step.
        in_sample_size (int): Sample size for empirical optimal transport estimation.
        device (torch.device): The device to run on.
        discount (float): Discount factor.
        mem_size (int): Memory size for the replay memory.
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
        for smp_id in range(smp_size):
            # Example: use t=1 as X and t=2 as Y (this is a simplification)
            X = np.expand_dims(samplepath_x[:, t - 1], axis=1)  # shape: (num_paths, 1)
            Y = np.expand_dims(samplepath_x[:, t], axis=1)  # shape: (num_paths, 1)
            data = np.concatenate([X, Y], axis=1)

            # Define a grid of y values
            y_min = np.min(Y) - 3
            y_max = np.max(Y) + 3
            y_grid = np.linspace(y_min, y_max, 200)
            dy = y_grid[1] - y_grid[0]  # grid spacing

            # Set bandwidth parameters for the two-step estimator
            bandwidth_x = 0.5  # for local regression
            bandwidth_e = 0.5  # for residual density

            # Estimate conditional densities
            f_cond_x = estimate_conditional_density_one_step(
                data, samplepath_x[smp_id, t - 1], bandwidth_x, bandwidth_e, y_grid
            )

            # Convert continuous density to discrete probabilities over the grid
            # We multiply by the grid spacing to approximate the integral
            pmf = f_cond_x * dy
            pmf /= pmf.sum()  # normalize so it sums to 1

            samples_x = np.random.choice(y_grid, size=in_sample_size, p=pmf)

            # Example: use t=1 as X and t=2 as Y (this is a simplification)
            X = np.expand_dims(samplepath_y[:, t - 1], axis=1)  # shape: (num_paths, 1)
            Y = np.expand_dims(samplepath_y[:, t], axis=1)  # shape: (num_paths, 1)
            data = np.concatenate([X, Y], axis=1)

            # Define a grid of y values
            y_min = np.min(Y) - 3
            y_max = np.max(Y) + 3
            y_grid = np.linspace(y_min, y_max, 200)
            dy = y_grid[1] - y_grid[0]  # grid spacing

            # Set bandwidth parameters for the two-step estimator
            bandwidth_x = 0.5  # for local regression
            bandwidth_e = 0.5  # for residual density

            f_cond_y = estimate_conditional_density_one_step(
                data, samplepath_y[smp_id, t - 1], bandwidth_x, bandwidth_e, y_grid
            )

            # Convert continuous density to discrete probabilities over the grid
            # We multiply by the grid spacing to approximate the integral
            pmf = f_cond_y * dy
            pmf /= pmf.sum()  # normalize so it sums to 1

            samples_y = np.random.choice(y_grid, size=in_sample_size, p=pmf)

            # Convert to PyTorch tensor and ensure correct shape [50, 1]
            next_x = torch.tensor(samples_x, dtype=torch.float32).reshape(-1, 1)
            next_y = torch.tensor(samples_y, dtype=torch.float32).reshape(-1, 1)

            x_batch = torch.repeat_interleave(next_x, repeats=in_sample_size, dim=0)
            y_batch = torch.tile(next_y, (in_sample_size, 1))
            l2_mat = torch.sum((x_batch - y_batch) ** 2, dim=1)

            if t == time_horizon:
                expected_v = 0.0
            elif t == time_horizon - 1:
                # Compute the empirical OT cost between next states.
                min_obj = l2_mat.reshape(in_sample_size, in_sample_size)
                expected_v = ot.emd2(
                    np.ones(in_sample_size) / in_sample_size,
                    np.ones(in_sample_size) / in_sample_size,
                    min_obj.detach().cpu().numpy(),
                )
            else:
                # Evaluate the target network for the next time step.
                time_tensor = torch.ones(x_batch.shape[0], 1, device=device) * (t + 1.0)
                val = target_net(time_tensor, x_batch, y_batch).reshape(-1)
                min_obj = (l2_mat + discount * val).reshape(
                    in_sample_size, in_sample_size
                )
                expected_v = ot.emd2(
                    np.ones(in_sample_size) / in_sample_size,
                    np.ones(in_sample_size) / in_sample_size,
                    min_obj.detach().cpu().numpy(),
                )

            # Push the transition into memory.
            memory.push(
                torch.tensor([t], dtype=torch.float32, device=device),
                samplepath_x[smp_id, t],
                samplepath_y[smp_id, t],
                torch.tensor([expected_v], device=device),
            )

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

        x_input = torch.tensor(samplepath_x[0, 0], dtype=torch.float32).reshape(
            1, x_dim
        )
        y_input = torch.tensor(samplepath_y[0, 0], dtype=torch.float32).reshape(
            1, y_dim
        )

        # Test the value at time 0 using the first sample.
        val = target_net(
            torch.ones(1, 1, device=device) * 0.0, x_input, y_input
        ).reshape(-1)
        val_hist[t] = val.item()

        memory.clear()
        print("Time step", t, "Loss:", loss_hist[t])

    print("Final value at time 0:", val_hist[0])
    return val_hist[0], val_hist, loss_hist
