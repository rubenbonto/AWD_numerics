import numpy as np
import rfcde
import time

def train_conditional_density_rfcde(data, d_X=1, d_Y=1, 
                                    n_trees=100, mtry=4, node_size=20, 
                                    n_basis=30, basis_system='cosine', 
                                    flambda=None):
    """
    Trains a conditional density estimator using the RFCDE random forest.
    
    Parameters:
        data (np.ndarray): Array of shape (n_samples, d_X + d_Y) where the first 
            d_X columns are inputs (X) and the next d_Y columns are targets (Y).
        d_X (int): Dimension of input X (default: 1).
        d_Y (int): Dimension of target Y (default: 1). Note: RFCDE supports univariate Y.
        n_trees (int): Number of trees in the random forest (default: 100).
        mtry (int): Number of features to consider at each split (default: 4).
        node_size (int): Minimum number of observations in a leaf node (default: 20).
        n_basis (int): Number of basis functions for density estimation (default: 30).
        basis_system (str): The basis system to use (e.g. 'cosine') (default: 'cosine').
        flambda (float, optional): If provided, trains the fRFCDE variant with this parameter.
        
    Returns:
        model: The trained RFCDE model with an additional attribute 'z_range' 
               that stores (min(Y), max(Y)) for use in evaluation.
    """
    # Ensure data is in float64.
    X = data[:, :d_X].astype(np.float64)
    Y = data[:, d_X:(d_X+d_Y)].squeeze().astype(np.float64)
    
    model = rfcde.RFCDE(n_trees=n_trees, mtry=mtry, node_size=node_size, 
                        n_basis=n_basis, basis_system=basis_system)
    
    start_time = time.time()
    if flambda is not None:
        model.train(X, Y, flambda=flambda)
    else:
        model.train(X, Y)
    end_time = time.time()
    print("RFCDE training took {:.2f} seconds.".format(end_time - start_time))
    
    # Save the range of Y values for later evaluation.
    model.z_range = (np.min(Y), np.max(Y))
    return model


def evaluate_conditional_density_rfcde(model, x_val, n_samples=1, 
                                        n_grid=1000, bandwidth=0.01):
    """
    Evaluates the trained RFCDE model at given x value(s) by sampling from the predicted density.
    
    Parameters:
        model: Trained RFCDE model with a 'z_range' attribute.
        x_val (float or list/np.ndarray of floats): x value(s) at which to evaluate.
        n_samples (int): Number of samples to draw (default: 1).
        n_grid (int): Number of grid points for density estimation (default: 1000).
        bandwidth (float): Bandwidth for the KDE (default: 0.01).
        
    Returns:
        For a float x_val: a NumPy array of shape (n_samples,) with sampled y values.
        For a list/array of x values: a list of such arrays.
    """
    def sample_from_density(x):
        # Convert x to float64 explicitly.
        x_arr = np.array([x], dtype=np.float64)
        z_min, z_max = model.z_range
        z_grid = np.linspace(z_min, z_max, n_grid)
        cde = model.predict(x_arr, z_grid, bandwidth=bandwidth)[0]
        p = cde / np.sum(cde)
        samples = np.random.choice(z_grid, size=n_samples, replace=True, p=p)
        return samples
    
    if isinstance(x_val, (int, float)):
        return sample_from_density(x_val)
    elif isinstance(x_val, (list, np.ndarray)):
        return [sample_from_density(x) for x in x_val]
    else:
        raise ValueError("x_val must be a float or a list/array of floats")
