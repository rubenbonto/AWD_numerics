import numpy as np
from sklearn.neighbors import KernelDensity




def estimate_conditional_density_one_step(data, x0, bandwidth_joint, bandwidth_marginal, y_grid):
    """
    Estimates the conditional density f(y|x0) via kernel density estimation.
    
    Args:
        data (np.ndarray): 2D array with shape (n_samples, 2) where the first column is x and the second is y.
        x0 (float): The fixed value of x at which the conditional density is evaluated.
        bandwidth_joint (float): Bandwidth used for estimating the joint density f(x,y).
        bandwidth_marginal (float): Bandwidth used for estimating the marginal density f(x).
        y_grid (np.ndarray): 1D array of y values at which to evaluate the conditional density.
        
    Returns:
        f_cond (np.ndarray): The estimated conditional density values evaluated at the given y_grid.
    """
    # Fit the joint density estimator on the (x,y) data
    kde_joint = KernelDensity(kernel='gaussian', bandwidth=bandwidth_joint)
    kde_joint.fit(data)
    
    # For each y in the grid, evaluate the joint density at (x0, y)
    X_eval = np.column_stack([np.full(len(y_grid), x0), y_grid])
    log_f_joint = kde_joint.score_samples(X_eval)
    f_joint = np.exp(log_f_joint)
    
    # Fit the marginal density estimator on the x-values only
    kde_x = KernelDensity(kernel='gaussian', bandwidth=bandwidth_marginal)
    kde_x.fit(data[:, 0].reshape(-1, 1))
    log_f_x = kde_x.score_samples([[x0]])
    f_x = np.exp(log_f_x)[0]
    
    # Compute the conditional density
    f_cond = f_joint / f_x
    return f_cond

import numpy as np

def gaussian_kernel(u):
    """Standard Gaussian kernel K(u) = (1 / sqrt(2*pi)) * exp(-u^2/2)."""
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

def estimate_conditional_density_two_step(
    data, x0, bandwidth_x, bandwidth_e, y_grid
):
    """
    Two-step estimator of f(y | x0):
    
    1) Estimate m(x) = E[Y|X=x] via Nadaraya-Watson.
    2) Compute residuals e_i = Y_i - m_hat(X_i).
    3) Estimate g(e|x0) locally, then shift by m_hat(x0).
       i.e. f_hat(y|x0) = g_hat(y - m_hat(x0) | x0).

    Args:
        data (np.ndarray): 2D array of shape (n_samples, 2) 
                           where data[:,0] are X's and data[:,1] are Y's.
        x0 (float): The point at which we want f(y|x0).
        bandwidth_x (float): Bandwidth for the local regression in X.
        bandwidth_e (float): Bandwidth for the residual density in e.
        y_grid (np.ndarray): 1D array of y-values at which to evaluate f(y|x0).

    Returns:
        f_cond (np.ndarray): Estimated conditional density f(y|x0) at each y in y_grid.
    """
    X = data[:, 0]
    Y = data[:, 1]
    n = len(X)
    
    # 1) Estimate m_hat(X_i) for each sample point (Nadaraya-Watson).
    m_hat = np.zeros(n)
    for i in range(n):
        # Weights for local regression around X[i]
        w = gaussian_kernel((X[i] - X) / bandwidth_x) / bandwidth_x
        w_sum = np.sum(w)
        if w_sum > 0:
            m_hat[i] = np.sum(w * Y) / w_sum
        else:
            # Fallback if all weights are 0 (very unlikely unless bandwidth_x is tiny)
            m_hat[i] = np.mean(Y)
    
    # 2) Residuals e_i
    e = Y - m_hat
    
    # 3) Estimate m_hat(x0)
    w0 = gaussian_kernel((x0 - X) / bandwidth_x) / bandwidth_x
    w0_sum = np.sum(w0)
    if w0_sum > 1e-12:
        m_x0 = np.sum(w0 * Y) / w0_sum
    else:
        m_x0 = np.mean(Y)
    
    # 4) For each y in y_grid, compute e_ = y - m_x0 and then estimate g_hat(e_|x0)
    #    Finally, f_hat(y|x0) = g_hat(y - m_x0 | x0).
    f_cond = np.zeros_like(y_grid, dtype=float)
    
    for j, y_val in enumerate(y_grid):
        e_val = y_val - m_x0
        
        # Numerator for g_hat(e_val | x0) = sum_i [ w0[i] * K((e_val - e[i]) / bandwidth_e ) / bandwidth_e ]
        # Denominator = sum_i w0[i]
        ke = gaussian_kernel((e_val - e) / bandwidth_e) / bandwidth_e
        numerator = np.sum(w0 * ke)
        
        # Local normalization by sum(w0)
        f_cond[j] = numerator / w0_sum  # = g_hat(e_val|x0)
    
    return f_cond
