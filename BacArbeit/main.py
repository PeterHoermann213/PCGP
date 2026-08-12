
import jax.numpy as jnp
import jax
import torch
import gpytorch
import matplotlib.pyplot as plt

from PCGP.deep_tensor import dirt_functions #functions for DIRT Algorythmn

from PCGP.deep_tensor.helper_functions import build_structure
from PCGP.deep_tensor.mll import single_mll

from torch import Tensor

import deep_tensor as dt

import math # delete after deubugging


# TODO: replace with your generated kernel file -
from kernel_f1 import kernel#_0 as kernel

# TODO: replace with your implemented sampler from helpers_*.py
#from helpers_sampling import sample_parameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)

torch.manual_seed(1)
#bypass torch.compile, otherwise can throw spatial dimension errors while calculating (with differntly large x_0 and x_1)
if hasattr(kernel, "_torch_dynamo_orig_callable"):
    kernel = kernel._torch_dynamo_orig_callable
elif hasattr(kernel, "__wrapped__"):
    kernel = kernel.__wrapped__
def function_1(a, x):
    return 3*torch.exp(-a*x)

def addTrainingsuncertainty(y, sigma):
    """generates a random nois with fixed sigma in percent for an value y"""
    deviation = torch.randn_like(y, dtype=torch.float64) * sigma
    return y*deviation

def sample_parameters(log_target_fn, domain, num_samples=5000, device="cpu", order=30):
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

    
    def neglogpost(xs: Tensor) -> Tensor:
        neg_log_liks = []
        for i in range(xs.shape[0]):
            a_val = xs[i, 0]
            
            
            if a_val < a_min or a_val > a_max:
                log_lik = torch.tensor(-1e9, dtype=xs.dtype, device=xs.device)
            else:
                log_lik = log_target_fn(a_val)
                if not isinstance(log_lik, torch.Tensor):
                    log_lik = torch.tensor(log_lik, dtype=xs.dtype, device=xs.device)
                
            neg_log_liks.append(log_lik)

        res = torch.stack(neg_log_liks)

        # NaN/Inf-Werte bereinigen (verhindert leere Auswertungs-Tensors in DIRT)
        #res = torch.nan_to_num(res, nan=1e9, posinf=1e9, neginf=-1e9)
        
        # Extremwerte begrenzen (verhindert Underflow/Overflow beim Exponentiieren)
        #res = torch.clamp(res, min=-1e7, max=1e9)

        return res

    target_func = dt.TargetFunc(neglogpost)

    # Define Dirt Parameters
    bounds = torch.tensor([[a_min, a_max]], dtype=torch.float64, device=device)
    reference = dt.GaussianReference()
    preconditioner = dt.UniformMapping(bounds, reference)

    
    basis = dt.Legendre(order=order)
    bases = dt.ApproxBases(basis, dim=1)  # 1D parameter space

    tt = dt.TT(options=dt.TTOptions(verbose=0))
    ftt = dt.FTT(bases, tt)

    # Construct DIRT Approximation
    dirt = dt.DIRT(target_func, preconditioner, ftt)

    rs = reference.random(n=num_samples, d=1)

    # Transform the samples according to SIRT approximation
    x_samples, neglogfxs_sirt = dirt.eval_irt(rs)
    # Compute potential function of the (unnormalised) target density at each SIRT sample
    neglogfxs_exact = target_func(x_samples)

    res = dt.run_independence_sampler(x_samples, neglogfxs_sirt, neglogfxs_exact)
    # create smaples
    #samples = dirt.sample(num_samples)

    return res



def sample_parameters2(log_target_fn, domain, num_samples=5000, device="cpu", order=15):
    a_min, a_max = float(domain[0]), float(domain[1])
    eps = (a_max - a_min) * 1e-5

    # Vorab-Scan für max_log Skalierung
    test_pts = torch.linspace(a_min, a_max, 50, device=device)
    log_vals = []
    for p in test_pts:
        v = log_target_fn(p.item())
        v_float = float(v.item() if isinstance(v, Tensor) else v)
        if v_float > -1e8 and not math.isnan(v_float):
            log_vals.append(v_float)
    if not log_vals:
        raise ValueError(f"log_target_fn lieferte keine gültigen Werte im Domain-Bereich [{a_min}, {a_max}].")

    max_log = max(log_vals)

    def target_density(xs: Tensor) -> Tensor:
        orig_shape = xs.shape
        xs_flat = xs.reshape(-1)
        N = xs_flat.numel()

        densities_flat = torch.zeros(N, dtype=xs.dtype, device=xs.device)

        for i in range(N):
            # explizit als Skalar-Float extrahieren
            a_val = float(xs_flat[i].item())

            if a_val < (a_min - eps) or a_val > (a_max + eps):
                densities_flat[i] = 0.0
                continue

            try:
                log_p = log_target_fn(a_val)
                log_p_val = float(log_p.item() if isinstance(log_p, Tensor) else log_p)
                
                shifted = log_p_val - max_log
                densities_flat[i] = math.exp(max(-80.0, min(0.0, shifted)))
            except Exception:
                densities_flat[i] = 0.0

        return densities_flat.reshape(orig_shape)

    target_func = dt.TargetFunc(target_density)

    bounds = torch.tensor([[a_min, a_max]], dtype=torch.float64, device=device)
    reference = dt.GaussianReference()
    preconditioner = dt.UniformMapping(bounds, reference)

    basis = dt.Legendre(order=order)
    bases = dt.ApproxBases(basis, dim=1)

    tt = dt.TT()
    ftt = dt.FTT(bases, tt)

    dirt = dt.DIRT(target_func, preconditioner, ftt)
    samples = dirt.sample(num_samples)

    return samples.squeeze()


def log_target(a):
    if isinstance(a, torch.Tensor):
        a_val = float(a.squeeze().item()) if a.numel() == 1 else float(a.reshape(-1)[0].item())
    else:
        a_val = float(a)

    a_tensor = torch.tensor(a_val, dtype=full_train_x.dtype, device=full_train_x.device)
    params = {**fixed_params, "a": a_tensor}

    try:
        val = single_mll(params, full_train_x, full_train_y, sigma, kernel, structure)
        val_float = float(val.item() if isinstance(val, torch.Tensor) else val)
        
        if math.isnan(val_float) or math.isinf(val_float):
            return -1e9
        return val_float

    except Exception as e:
        print(f"[log_target Error bei a={a_val:.4f}]: {type(e).__name__}: {e}")
        return -1e9

# --- ground truth -----------------------------------------------------------
# TODO: set the true value of your physical parameter
#theta_true = ...   # e.g. torch.tensor(1.5)
a_true = torch.tensor(3.0, dtype=torch.float64)
sigma_train = 0.05 #5% std
# --- simulate training data -------------------------------------------------
# TODO: implement simulate_data(theta, x_coords) using the analytical solution
#       of your ODE; return (train_x, train_y) in the format expected by the kernel
#       train_x shape: (N * num_tasks, num_input_dims + 1)  -- last col = task label
#       train_y shape: (N * num_tasks,)
N_train = 100
N_g = 10
train_x_0= torch.linspace(0.1, 4.0, N_train)
train_x_1 = torch.linspace(0.1, 4.0, N_g)
train_i_0 = torch.zeros_like(train_x_0)

train_i_1 = torch.ones_like(train_x_1)
train_x = torch.cat([train_x_0, train_x_1]) # für Troubleshooting hier eventuell [] entfernen
train_i = torch.cat([train_i_0, train_i_1])
full_train_x = torch.stack([train_x, train_i], dim=-1)

train_x_1_i = torch.linspace(0.1, 4, N_train)
train_x_0_i = torch.linspace(0.1, 4, N_train)

true_y_0 = function_1(a_true, train_x_0)
train_y_0 = true_y_0 + addTrainingsuncertainty(true_y_0, sigma_train)

train_y_1 = torch.zeros_like(train_x_1)
full_train_y = torch.cat([train_y_0, train_y_1])

print("Unique tasks in full_train_x:", torch.unique(full_train_x[:, -1]))
print("Shape of full_train_x:", full_train_x.shape)

#true_y = model.solve(full_train_x) #wie in SIR_MODELL als test




# --- simulate test data -----------------------------------------------------


test_x = torch.linspace(0, 4, 50)
test_y = function_1(a_true, test_x)



sigma = 0.05   # observation noise std

num_tasks = 2   # number of tasks (= rows of your B-matrix)

# --- precompute structure, needed for kernel compilation ---------------------------------------------------
structure = build_structure(full_train_x, full_train_x, num_tasks=num_tasks)
# --- fixed (known) kernel hyperparameters -----------------------------------
# TODO: set the kernel hyperparameters that are NOT being inferred
#       (e.g. lengthscale, amplitude if known or pre-fitted)

fixed_params = {"lengthscale": torch.tensor(0.01, dtype=torch.float64, device=device),
    "amplitude": torch.tensor(0.1, dtype=torch.float64, device=device)}
# --- define target density --------------------------------------------------
# The sampler requires a scalar-valued function of the parameter(s) to infer.
# Here we fix all kernel hyperparameters and vary only the physical parameter a.
a_guess = torch.tensor(3.8, dtype=torch.float64)

#model = kernel(full_train_x, full_train_x, parameters={**fixed_params, "a": a_guess}, structure=structure)

def log_target_original(a):
    """Log-likelihood as a scalar function of the physical parameter a."""
    if isinstance(a, torch.Tensor):
        a_val = float(a.squeeze().item()) if a.numel() == 1 else float(a.reshape(-1)[0].item())
    else:
        a_val = float(a)

    # Erstelle params mit 0D Skalar-Tensor für 'a'
    a_tensor = torch.tensor(a_val, dtype=torch.float64, device=device)
    params = {**fixed_params, "a": a_tensor}

    try:
        val = single_mll(params, full_train_x, full_train_y, sigma, kernel, structure)
        print(f"[log_target a={a_val:.4f}]")
        
        # Sicherstellen, dass das Ergebnis ein 0D-Skalar ist
        if isinstance(val, torch.Tensor):
            val = val.squeeze()
            if val.numel() > 1:
                val = val.sum()
            if torch.isnan(val) or torch.isinf(val):
                return torch.tensor(-1e9, dtype=torch.float64, device=device)
            return val
        else:
            val_t = torch.tensor(float(val), dtype=torch.float64, device=device)
            if torch.isnan(val_t) or torch.isinf(val_t):
                return torch.tensor(-1e9, dtype=torch.float64, device=device)
            return val_t
            
    except Exception as e:
        print(f"[log_target Error bei a={a_val:.4f}]: {type(e).__name__}: {e}")
        return -1e9

# --- run deep_tensor sampler ------------------------------------------------
# TODO: specify the domain [a_min, a_max] over which to sample
a_min = 1.0
a_max = 100000.0

# TODO: sample_parameters should return dictionary of samples drawn from the likelihood
#       distribution of the parameters, using the DIRT algorithm implemented in helpers_*.py
# Pass log_target directly into the sampler

samples_result = sample_parameters(
    log_target_fn=log_target_original,
    domain=[a_min, a_max],
    num_samples=1000,
    device=device,
    order=30
)

samples = samples_result.xs
#samples_np = samples.detach().cpu().numpy()
# --- evaluate results -------------------------------------------------------
print(f"True a:       {a_true:.4f}")
print(f"Mean of samples:  {samples.mean():.4f}")
print(f"Std of samples:   {samples.std():.4f}")

# TODO: compare to reference methods (e.g. Laplace approximation, grid evaluation, MCMC)
#       from the InverseProblem Tutorial

# --- plot -------------------------------------------------------------------
plt.figure()
plt.hist(samples.numpy(), bins=30, density=True, label="deep_tensor samples")
plt.axvline(a_true.item(), color="red", linestyle="--", label="true value")
plt.xlabel("a")
plt.ylabel("density")
plt.legend()
plt.title("Posterior samples of physical parameter")
plt.show()

