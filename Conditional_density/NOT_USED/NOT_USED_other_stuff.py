import numpy as np
from scipy.stats import logistic, norm
import matplotlib.pyplot as plt


def kernel_scenarios(data, kernelDistribution=logistic, Markovian=True):
    """
    Constructs a closure for conditional density (scenario) generation based on kernel smoothing.

    Args:
        data (np.ndarray): An array of shape (N, T) where N is the number of trajectories and T is the number of stages.
        kernelDistribution: A distribution from scipy.stats to be used as the kernel (default is logistic).
        Markovian (bool): If True, weights are reinitialized at each stage (for scenario lattices).
                          If False, weights are multiplied over stages (for scenario trees).

    Returns:
        closure: A function that, when called, generates one sample trajectory (a vector of length T).
    """

    def closure():
        N, T = data.shape
        d = 1  # dimensionality of the kernel (univariate in our case)
        w = np.ones(N, dtype=np.float64)  # initialize weights
        x = np.empty(T, dtype=np.float64)  # to store the generated trajectory

        for t in range(T):
            # Normalize weights
            w = w / np.sum(w)
            # Compute effective sample size
            Nt = (np.sum(w) ** 2) / np.sum(w**2)
            # Compute the weighted mean and effective standard deviation at stage t
            mean_t = np.sum(w * data[:, t])
            sigma_t = np.sqrt(np.sum(w * ((data[:, t] - mean_t) ** 2)))
            # Bandwidth at stage t: note exponent is -1/(d+4)
            ht = sigma_t * (Nt ** (-1 / (d + 4))) + np.finfo(float).eps

            # Sample an index using the weights (composition method)
            u = np.random.uniform(0, 1)
            cum_w = np.cumsum(w)
            jstar = np.searchsorted(cum_w, u)

            # Sample a new value from the kernel distribution centered at data[jstar, t] with scale ht.
            dist = kernelDistribution(loc=data[jstar, t], scale=ht)
            x[t] = dist.rvs()

            # Update weights for the next stage.
            if t < T - 1:
                if Markovian:
                    # For Markovian scenarios, reinitialize weights based solely on the current stage.
                    w = np.array(
                        [
                            kernelDistribution(loc=data[j, t], scale=ht).pdf(x[t])
                            for j in range(N)
                        ]
                    )
                else:
                    # For non-Markovian (scenario trees), multiply the weights.
                    w = w * np.array(
                        [
                            kernelDistribution(loc=data[j, t], scale=ht).pdf(x[t])
                            for j in range(N)
                        ]
                    )
        return x

    return closure


def train_conditional_density_kernel(data, kernelDistribution=logistic, Markovian=True):
    """
    "Trains" the conditional density estimator by wrapping the data into a closure.

    Args:
        data (np.ndarray): An (N, T) array where each row is a trajectory.
        kernelDistribution: The kernel to use (default: logistic).
        Markovian (bool): Flag to indicate if the weighting is Markovian.

    Returns:
        estimator: A closure function that, when called, generates a sample trajectory.
    """
    data_float = data.astype(np.float64)
    estimator = kernel_scenarios(
        data_float, kernelDistribution=kernelDistribution, Markovian=Markovian
    )
    return estimator


def evaluate_conditional_density_kernel(estimator, stage, n_samples=1):
    """
    Evaluates the estimator by drawing multiple scenarios and extracting the value at the specified stage.

    Args:
        estimator: The closure returned by train_conditional_density_kernel.
        stage (int): The stage (1-indexed) at which to evaluate the conditional density.
        n_samples (int): Number of scenarios (samples) to draw.

    Returns:
        samples (np.ndarray): An array of sampled values at the specified stage.
    """
    stage_index = stage - 1  # convert 1-indexed stage to 0-index
    samples = []
    for _ in range(n_samples):
        traj = estimator()  # generate one full trajectory
        samples.append(traj[stage_index])
    return np.array(samples)
