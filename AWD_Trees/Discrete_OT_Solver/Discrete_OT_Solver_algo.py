from scipy.optimize import linprog
import ot
import numpy as np
from scipy.special import logsumexp

"""
This module provides multiple methods to solve discrete optimal transport (OT) problems.
Each function implements a different approach to compute the OT plan, typically used
for solving sub-OT problems that occur in backward induction.

Methods included:
- `solver_pot` and `solver_lp_pot`: Uses the Earth Mover's Distance (EMD) from the POT library.
- `solver_lp`: Uses linear programming (LP) to find the OT plan.
- `Sinkhorn_iteration`: Implements the Sinkhorn algorithm for entropic regularized OT.
"""


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
        raise ValueError(
            "The total mass of the source and target distributions must be equal."
        )

    pi_ratios = np.array(pi_ratios, dtype=np.float64)
    pi_tilde_ratios = np.array(pi_tilde_ratios, dtype=np.float64)

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
    n, m = distance_matrix_subset.shape
    c = distance_matrix_subset.flatten()

    # Row constraints
    A_eq = np.zeros((n + m, n * m))
    b_eq = np.concatenate([pi_ratios, pi_tilde_ratios])

    for i in range(n):  # Row constraints
        A_eq[i, i * m : (i + 1) * m] = 1

    for j in range(m):  # Column constraints
        A_eq[n + j, j::m] = 1

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, method="highs")

    return res.x.reshape(n, m) if res.success else None


def Sinkhorn_iteration(
    distance_matrix, p1, p2, stopping_criterion, lambda_reg, max_iterations=1000
):
    """
    Performs a stabilized Sinkhorn iteration in the log domain (to avoid division by zero issues) to compute the optimal transport plan.

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


def solver_lp_pot(distance_matrix_subset, pi_ratios, pi_tilde_ratios, reg=1e-2):
    """
    Solve for the optimal transport plan using the POT library's (fast!) EMD solver.

    Parameters:
    - distance_matrix_subset (np.ndarray): A 2D cost matrix.
    - pi_ratios (np.ndarray): 1D source distribution (row marginals).
    - pi_tilde_ratios (np.ndarray): 1D target distribution (column marginals).

    Returns:
    - np.ndarray: The optimal transport plan (probability matrix).
    """
    pi_ratios = np.array(pi_ratios, dtype=np.float64)
    pi_tilde_ratios = np.array(pi_tilde_ratios, dtype=np.float64)

    return ot.lp.emd(pi_ratios, pi_tilde_ratios, distance_matrix_subset)


def solver_pot_sinkhorn(distance_matrix_subset, pi_ratios, pi_tilde_ratios, epsilon):
    """
    Solve for the optimal transport plan using the POT library's sinkhorn solver.

    Parameters:
    - distance_matrix_subset (np.ndarray): A 2D cost matrix.
    - pi_ratios (np.ndarray): 1D source distribution (row marginals).
    - pi_tilde_ratios (np.ndarray): 1D target distribution (column marginals).

    Returns:
    - np.ndarray: The optimal transport plan (probability matrix).
    """
    pi_ratios = np.array(pi_ratios, dtype=np.float64)
    pi_tilde_ratios = np.array(pi_tilde_ratios, dtype=np.float64)

    return ot.sinkhorn(pi_ratios, pi_tilde_ratios, distance_matrix_subset, epsilon)


# Some other technique I tried to be faster but failed
import jax.numpy as jnp
from ott.geometry import geometry  # Correct module for defining cost matrices
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

"""
I attempted to speed up the computation using JAX, but it did not yield significant improvements. 
I also experimented with parallelization, but the results were not satisfactory. 
However, if you have access to a CUDA-compatible GPU, these computations could be significantly accelerated.
"""


def solver_jax(distance_matrix_np, p1_np, p2_np, epsilon, threshold=1e-4):
    """
    Computes the entropically regularized optimal transport plan using OTT's Sinkhorn solver,
    with a precomputed distance matrix.

    Parameters:
      p1_np (np.ndarray): Source probability distribution (1D, sums to 1), shape (n,).
      p2_np (np.ndarray): Target probability distribution (1D, sums to 1), shape (m,).
      distance_matrix_np (np.ndarray): Cost matrix of shape (n, m).
      epsilon (float): Entropic regularization parameter.
      threshold (float): Convergence threshold.

    Returns:
      np.ndarray: Optimal transport plan matrix of shape (n, m).
    """
    # Convert inputs from NumPy to JAX arrays.

    p1_np = np.array(p1_np, dtype=np.float64)
    p2_np = np.array(p2_np, dtype=np.float64)
    distance_matrix_np = np.array(distance_matrix_np, dtype=np.float64)

    p1_np /= np.sum(p1_np)
    p2_np /= np.sum(p2_np)

    distance_matrix_np /= np.max(distance_matrix_np)

    p1 = jnp.array(p1_np)
    p2 = jnp.array(p2_np)
    cost_mat = jnp.array(distance_matrix_np)

    # Create geometry using the precomputed cost matrix.
    geom = geometry.Geometry(cost_matrix=cost_mat, epsilon=epsilon)

    # Set up the linear problem with the given marginals.
    prob = linear_problem.LinearProblem(geom, a=p1, b=p2)

    # Create the Sinkhorn solver instance.
    solver = sinkhorn.Sinkhorn(
        threshold=threshold,
        max_iterations=1000,
        norm_error=2,
        lse_mode=True,
    )

    # Solve the OT problem.
    out = solver(prob)

    # Extract the transport plan.
    transport_plan = out.matrix  # Correct way to extract the transport matrix

    # Convert the transport plan back to a NumPy array.
    return np.array(transport_plan)
