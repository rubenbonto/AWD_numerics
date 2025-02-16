from scipy.optimize import linprog
import ot
import numpy as np
from scipy.special import logsumexp

"""
This module provides multiple methods to solve discrete optimal transport (OT) problems.
Each function implements a different approach to compute the OT plan, typically used
for solving sub-OT problems that occur in backward induction.

Methods included:
- `solver_pot`: Uses the Earth Mover's Distance (EMD) from the POT library.
- `solver_lp`: Uses linear programming (LP) to find the OT plan.
- `Sinkhorn_iteration`: Implements the Sinkhorn algorithm for entropic regularized OT.
"""

# This method is the fastest and most accurate among the three.
def solver_pot(distance_matrix_subset, pi_ratios, pi_tilde_ratios):
    """
    Solve for the optimal transport plan using the POT library's EMD solver.

    Parameters:
    - distance_matrix_subset (np.ndarray): A 2D cost matrix.
    - pi_ratios (np.ndarray): 1D source distribution (row marginals).
    - pi_tilde_ratios (np.ndarray): 1D target distribution (column marginals).

    Returns:
    - np.ndarray: The optimal transport plan (probability matrix).
    """
    if not np.isclose(np.sum(pi_ratios), np.sum(pi_tilde_ratios)):
        raise ValueError("The total mass of the source and target distributions must be equal.")
    
    return ot.emd(pi_ratios, pi_tilde_ratios, distance_matrix_subset)

def solver_lp(distance_matrix_subset, pi_ratios, pi_tilde_ratios):
    """
    Solve for the optimal transport plan using linear programming.

    Parameters:
    - distance_matrix_subset (np.ndarray): Cost matrix.
    - pi_ratios (np.ndarray): Source distribution.
    - pi_tilde_ratios (np.ndarray): Target distribution.

    Returns:
    - np.ndarray: Optimal transport probability matrix.
    """
    num_rows, num_cols = distance_matrix_subset.shape
    c = distance_matrix_subset.flatten()
    A_eq, b_eq = [], []
    
    # Row constraints
    for i in range(num_rows):
        row_constraint = [0] * (num_rows * num_cols)
        for j in range(num_cols):
            row_constraint[i * num_cols + j] = 1
        A_eq.append(row_constraint)
        b_eq.append(pi_ratios[i])
    
    # Column constraints
    for j in range(num_cols):
        col_constraint = [0] * (num_rows * num_cols)
        for i in range(num_rows):
            col_constraint[i * num_cols + j] = 1
        A_eq.append(col_constraint)
        b_eq.append(pi_tilde_ratios[j])
    
    bounds = [(0, None)] * (num_rows * num_cols)
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    return result.x.reshape(num_rows, num_cols)

def Sinkhorn_iteration(distance_matrix, p1, p2, stopping_criterion, lambda_reg, max_iterations=1000):
    """
    Performs a stabilized Sinkhorn iteration in the log domain to compute the optimal transport plan.
    
    Parameters:
    - distance_matrix (np.ndarray): Cost matrix (assumed nonnegative).
    - p1 (np.ndarray): Source probability distribution (should sum to 1).
    - p2 (np.ndarray): Target probability distribution (should sum to 1).
    - stopping_criterion (float): Convergence threshold.
    - lambda_reg (float): Regularization parameter.
    - max_iterations (int): Maximum number of iterations.
    
    Returns:
    - np.ndarray: Optimal transport plan matrix.
    """
    # Compute logK to avoid direct exponentiation of extreme values
    logK = -lambda_reg * distance_matrix
    n1, n2 = distance_matrix.shape

    # Initialize dual variables in the log domain (u and v correspond to log(beta) and log(gamma))
    u = np.zeros(n1)
    v = np.zeros(n2)
    
    for iteration in range(max_iterations):
        u_prev = u.copy()
        # Update u using the log-sum-exp trick to avoid numerical issues
        u = np.log(p1) - logsumexp(logK + v[None, :], axis=1)
        v = np.log(p2) - logsumexp(logK.T + u[None, :], axis=1)
        
        # Convergence check: if the change in u is below the threshold, we break
        if np.sum(np.abs(u - u_prev)) < stopping_criterion:
            break
    
    # Recover the optimal transport plan using the dual variables
    transport_plan = np.exp(u[:, None] + v[None, :] + logK)
    return transport_plan