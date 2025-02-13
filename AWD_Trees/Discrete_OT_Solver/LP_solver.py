from scipy.optimize import linprog


import ot
import numpy as np

def solver_pot(distance_matrix_subset, pi_ratios, pi_tilde_ratios):
    """
    Solve for the optimal transport plan using the POT library's EMD solver.

    Parameters:
      distance_matrix_subset (np.ndarray): A 2D cost matrix.
      pi_ratios (np.ndarray): 1D source distribution (row marginals).
      pi_tilde_ratios (np.ndarray): 1D target distribution (column marginals).

    Returns:
      np.ndarray: The optimal transport plan (probability matrix).
    """
    # Check if total masses match (necessary for balanced optimal transport)
    if not np.isclose(np.sum(pi_ratios), np.sum(pi_tilde_ratios)):
        raise ValueError("The total mass of the source and target distributions must be equal.")

    # Compute the optimal transport plan using the EMD (Earth Mover's Distance) algorithm.
    probability_matrix = ot.emd(pi_ratios, pi_tilde_ratios, distance_matrix_subset)
    
    return probability_matrix

def solver_lp(distance_matrix_subset, pi_ratios, pi_tilde_ratios):
    """
    Solve for the optimal probability matrix that minimizes the cost when
    multiplied with the distance_matrix_subset.
    """
    num_rows, num_cols = distance_matrix_subset.shape

    # Flatten the distance matrix to use it as the cost vector in linprog
    c = distance_matrix_subset.flatten()

    # Constraints
    A_eq = []
    b_eq = []

    # Row constraints: each row should sum to the corresponding value in pi_ratios
    for i in range(num_rows):
        row_constraint = [0] * (num_rows * num_cols)
        for j in range(num_cols):
            row_constraint[i * num_cols + j] = 1
        A_eq.append(row_constraint)
        b_eq.append(pi_ratios[i])

    # Column constraints: each column should sum to the corresponding value in pi_tilde_ratios
    for j in range(num_cols):
        col_constraint = [0] * (num_rows * num_cols)
        for i in range(num_rows):
            col_constraint[i * num_cols + j] = 1
        A_eq.append(col_constraint)
        b_eq.append(pi_tilde_ratios[j])

    # Bounds: each entry in the probability matrix should be non-negative
    bounds = [(0, None)] * (num_rows * num_cols)

    # Solve the linear program
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    # Reshape the result back into a matrix of shape (num_rows, num_cols)
    probability_matrix = result.x.reshape(num_rows, num_cols)
    
    return probability_matrix