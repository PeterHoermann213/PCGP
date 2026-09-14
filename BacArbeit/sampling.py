import jax.numpy as jnp
import jax
import torch
import gpytorch
import matplotlib.pyplot as plt

from PCGP.deep_tensor.helper_functions import build_structure
from PCGP.deep_tensor.mll import single_mll

from torch import Tensor

import deep_tensor as dt




def sample_parameters(neglogpost, domain, num_samples=5000, device="cpu", order=30):
    """
    Samples parameters using the DIRT

    Parameters
    ----------
    log_target_fn : callable
        targetfunction returning the log-likelihood / log-posterior.
    domain : list or tuple
        Domain bounds [a_min, a_max] for parameter 'a'.
    num_samples : int, optional
        Number of samples to draw (default: 5000).
    device : torch.device or str, optional
        Torch device ('cpu' or 'cuda').
    order : int, optional
        Legendre polynomial degree for spatial approximation (default: 30).

    Returns
    -------
    torch.Tensor
        1D Tensor of shape (num_samples,) containing parameter samples.
    
    """
    a_min, a_max = domain[0], domain[1]

    
    

    target_func = dt.TargetFunc(neglogpost)

    # Define Dirt Parameters
    bounds = torch.tensor([[a_min, a_max]], dtype=torch.float64, device=device)
    reference = dt.GaussianReference()
    preconditioner = dt.UniformMapping(bounds, reference)

    
    basis = dt.Legendre(order=order)
    #basis = dt.Lagrange1(num_elems=order)
        
    bases = dt.ApproxBases(basis, dim=1)  # 1D parameter space

    tt = dt.TT(options=dt.TTOptions(verbose=0))
    ftt = dt.FTT(bases, tt)

    # Construct DIRT Approximation
    dirt = dt.DIRT(target_func, preconditioner, ftt)#bridge=dt.SingleLayer()

    rs = reference.random(n=num_samples, d=1)

    # Transform the samples according to SIRT approximation
    x_samples, neglogfxs_sirt = dirt.eval_irt(rs)
    # Compute potential function of the (unnormalised) target density at each SIRT sample
    neglogfxs_exact = target_func(x_samples)

    res = dt.run_independence_sampler(x_samples, neglogfxs_sirt, neglogfxs_exact)
    # create smaples
    #neglogfxs_sirt has a mean of 2.25 which is nearer at 3.0
    #return neglogfxs_sirt

    #debugging:
    

    print("SIRT samples:")
    print(x_samples[:20])

    print("SIRT potential:")
    print(neglogfxs_sirt[:20])
    exact = target_func(x_samples)

    print("Exact potential:")
    print(exact[:20])
    return res
