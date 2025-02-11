import numpy as np


def Sinkhorn_iteration(distance_matrix, p1, p2, stopping_criterion, lambda_reg):
    """
    Performs Sinkhorn iterations to compute the optimal transport plan.

    Parameters:
    - distance_matrix (np.ndarray): n1 x n2 matrix representing distances between nodes.
    - p1 (list of float): Marginal probabilities for the first distribution.
    - p2 (list of float): Marginal probabilities for the second distribution.
    - stopping_criterion (float): Threshold for convergence.
    - lambda_reg (float): Regularization parameter.

    Returns:
    - np.ndarray: Optimal transport plan matrix.
    """
    # Initialize the K matrix
    K = np.exp(-lambda_reg * distance_matrix)
    
    # Initialize beta and gamma
    n1, n2 = distance_matrix.shape
    beta = np.ones(n1)
    gamma = np.ones(n2)
    
    max_iterations = 1000
    iteration = 0
    epsilon = 1e-10  # Threshold for negligible values
    
    while iteration < max_iterations:
        iteration += 1
        
        # Store previous scaling vectors for convergence check
        beta_prev = beta.copy()
        gamma_prev = gamma.copy()
        
        # Update beta
        for i in range(n1):
            beta[i] = p1[i] / np.sum(K[i, :] * gamma)
        
        # Update gamma
        for j in range(n2):
            gamma[j] = p2[j] / np.sum(K[:, j] * beta)
            
        # Check for convergence
        beta_diff = np.sum(np.abs(beta - beta_prev))
        gamma_diff = np.sum(np.abs(gamma - gamma_prev))
        if beta_diff + gamma_diff < stopping_criterion or np.all(beta < epsilon) or np.all(gamma < epsilon):
            break

    # Compute the transport plan matrix
    pi = np.outer(beta, gamma) * K
    
    return pi