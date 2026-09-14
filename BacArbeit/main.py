
import jax.numpy as jnp
import jax
import torch
import gpytorch
import matplotlib.pyplot as plt

import sampling #sampling Algorythm via DIRT

from PCGP.deep_tensor.helper_functions import build_structure
from PCGP.deep_tensor.mll import single_mll

from torch import Tensor

import deep_tensor as dt

import math # delete after deubugging

#from kernel_f1 import kernel#_0 as kernel
#from kernel_2 import kernel
from kernel_tutorial import kernel



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)

torch.manual_seed(1)
#bypass torch.compile, otherwise can throw spatial dimension errors while calculating (with differntly large x_0 and x_1)
if hasattr(kernel, "_torch_dynamo_orig_callable"):
    kernel = kernel._torch_dynamo_orig_callable
elif hasattr(kernel, "__wrapped__"):
    kernel = kernel.__wrapped__
def function_1(a, x):
    return torch.sin(x)

def function_2(a, x):
    return 2*torch.cos(x)

def addTrainingsuncertainty(y, sigma):
    """generates a random nois with fixed sigma in percent for an value y"""
    deviation = torch.randn_like(y, dtype=torch.float64) * sigma
    return deviation


# --- ground truth -----------------------------------------------------------

a_true = torch.tensor(2.0, dtype=torch.float64)
sigma = 0.01

# --- simulate training data -------------------------------------------------
#       train_x shape: (N * num_tasks, num_input_dims + 1)  -- last col = task label
#       train_y shape: (N * num_tasks,)
N_train = 100
N_g = 40
train_x_0= torch.linspace(0.0, 4.0, N_train)
train_x_1 = torch.linspace(0.0, 4.0, N_g)
train_i_0 = torch.zeros_like(train_x_0)

train_i_1 = torch.ones_like(train_x_1)
train_x = torch.cat([train_x_0, train_x_1]) # für Troubleshooting hier eventuell [] entfernen
train_i = torch.cat([train_i_0, train_i_1])
full_train_x = torch.stack([train_x, train_i], dim=-1)

true_y_0 = function_1(a_true, train_x_0)
train_y_0 = true_y_0 + addTrainingsuncertainty(true_y_0, sigma)

#train_y_1 = torch.zeros_like(train_x_1)
true_y_1 = function_2(a_true, train_x_1)
train_y_1 = true_y_1
full_train_y = torch.cat([train_y_0, train_y_1])
print(full_train_y)

#print("Unique tasks in full_train_x:", torch.unique(full_train_x[:, -1]))
#print("Shape of full_train_x:", full_train_x.shape)

#true_y = model.solve(full_train_x) #wie in SIR_MODELL als test



num_tasks = 2   # number of tasks (= rows of your B-matrix)

# --- precompute structure, needed for kernel compilation ---------------------------------------------------
structure = build_structure(full_train_x, full_train_x, num_tasks=num_tasks)
# --- fixed (known) kernel hyperparameters -----------------------------------

fixed_params = {"lengthscale": torch.tensor(0.1, dtype=torch.float64, device=device),
    "amplitude": torch.tensor(3.0, dtype=torch.float64, device=device)}

def log_target(a):
    """Log-likelihood as a scalar function of the physical parameter a."""
    
    params = {**fixed_params, "a": a}

    
    val = single_mll(params, full_train_x, full_train_y, sigma, kernel, structure)
    #print(f"[log_target a={a:.4f}]")
    return -val
    
            


# --- define target density --------------------------------------------------
# The sampler requires a scalar-valued function of the parameter(s) to infer.
# Here we fix all kernel hyperparameters and vary only the physical parameter a.

#model = kernel(full_train_x, full_train_x, parameters={**fixed_params, "a": a_guess}, structure=structure)

def targetDensity(xs: Tensor) -> Tensor:
        neg_log_liks = []
        for i in range(xs.shape[0]):
            a_val = xs[i, 0]
            
            
            if a_val < a_min or a_val > a_max:
                log_lik = torch.tensor(-1e9, dtype=xs.dtype, device=xs.device)
            else:
                log_lik = log_target(a_val)
                if not isinstance(log_lik, torch.Tensor):
                    log_lik = torch.tensor(log_lik, dtype=xs.dtype, device=xs.device)
                
            neg_log_liks.append(log_lik)

        res = torch.stack(neg_log_liks)

        return res


# --- run deep_tensor sampler ------------------------------------------------
# TODO: specify the domain [a_min, a_max] over which to sample
a_min = 1.0
a_max = 4.5

samples_result = sampling.sample_parameters(
    log_target_fn=targetDensity,
    domain=[a_min, a_max],
    num_samples=10000,
    device=device,
    order=60
)

samples = samples_result.xs
#samples = samples_result

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

plt.savefig("testPlot.png")
plt.show()





# --- DEBUG -------------------------------------------------------------------
#ONLY DEBUG PLOTS HERE:
#will be removed in final version


for a_test in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    a_tensor = torch.tensor(
        a_test,
        dtype=torch.float64,
        device=full_train_x.device
    )

    params = {
        **fixed_params,
        "a": a_tensor
    }

    mll = single_mll(
        params,
        full_train_x,
        full_train_y,
        sigma,
        kernel,
        structure
    )

    print(f"a = {a_test:5.2f}, MLL = {mll.item(): .8e}")





#lossFunc plot:
plt.figure()

a_linespace = torch.linspace(1, 4, 4)
lossArray = []
for a in a_linespace:
    lossArray.append(log_target(a))
lossArray = torch.FloatTensor(lossArray)
print(lossArray)
plt.plot(a_linespace, torch.exp(lossArray - lossArray.max()), color="red", linestyle="--", label="loss_func")
plt.xlabel("a")
plt.ylabel("density")
plt.legend()
plt.title("Posterior samples of physical parameter")

plt.savefig("testPlotLossF.png")
plt.show()


def physical_nll(a):
    #prediction = 3.0 * torch.exp(-a * train_x_0)
    prediction = a * torch.cos(train_x_0)
    residual = train_y_0 - prediction

    return 0.5 * torch.sum(
        (residual / sigma)**2
    )



a_grid = torch.linspace(1.5, 4.5, 601)

# GP posterior
mll = torch.stack([
    single_mll(
        {**fixed_params, "a": a},
        full_train_x,
        full_train_y,
        sigma,
        kernel,
        structure
    )
    for a in a_grid
])

gp_posterior = torch.exp(mll - mll.max())

# Direct physical likelihood
nll = torch.stack([
    physical_nll(a)
    for a in a_grid
])

physical_posterior = torch.exp(
    -(nll - nll.min())
)

plt.figure(figsize=(8, 5))

plt.plot(
    a_grid.numpy(),
    physical_posterior.numpy(),
    label="Direct physical likelihood"
)


plt.plot(
    a_grid.numpy(),
    gp_posterior.numpy(),
    label="GP marginal likelihood"
)

plt.axvline(
    2.0,
    linestyle="--",
    label="true a"
)

plt.xlabel("a")
plt.ylabel("relative posterior density")
plt.legend()
plt.grid()
plt.savefig("gaussianMaximum")
plt.show()

#debug further:


def single_mll_debug(params, train_x, train_y, sigma, kernel, structure):

    K = kernel(train_x, train_x, params, structure)

    N = K.shape[0]

    K = K + sigma**2 * torch.eye(
        N,
        dtype=train_x.dtype,
        device=train_x.device
    )

    L = torch.linalg.cholesky(K)

    L_inv_y = torch.linalg.solve_triangular(
        L,
        train_y.unsqueeze(-1),
        upper=False
    ).squeeze(-1)

    fit_term = -0.5 * torch.sum(L_inv_y**2)

    complexity_term = -torch.sum(
        torch.log(torch.diagonal(L))
    )

    constant_term = -0.5 * N * math.log(2 * math.pi)

    return fit_term, complexity_term, constant_term



for a_test in [2.7, 2.8, 2.9, 3.0, 3.1, 3.2]:

    params = {
        **fixed_params,
        "a": torch.tensor(
            a_test,
            dtype=torch.float64,
            device=device
        )
    }

    fit, complexity, constant = single_mll_debug(
        params,
        full_train_x,
        full_train_y,
        sigma,
        kernel,
        structure
    )

    print(
        f"a={a_test:.2f} | "
        f"fit={fit.item():.6f} | "
        f"complexity={complexity.item():.6f} | "
        f"MLL={(fit+complexity+constant).item():.6f}"
    )

    a_grid = torch.linspace(1.5, 4.5, 3001)

mll = torch.stack([
    single_mll(
        {**fixed_params, "a": a},
        full_train_x,
        full_train_y,
        sigma,
        kernel,
        structure
    )
    for a in a_grid
])

# Normalize for numerical stability
p = torch.exp(mll - mll.max())

# Normalize integration weights
da = a_grid[1] - a_grid[0]
Z = torch.sum(p) * da

p = p / Z

# Exact GP posterior mean
posterior_mean = torch.sum(a_grid * p) * da

# Exact GP posterior variance
posterior_var = torch.sum(
    (a_grid - posterior_mean)**2 * p
) * da

posterior_std = torch.sqrt(posterior_var)

# MAP / MLE
idx = torch.argmax(mll)
a_map = a_grid[idx]

print(f"GP MAP:       {a_map.item():.6f}")
print(f"GP mean:      {posterior_mean.item():.6f}")
print(f"GP std:       {posterior_std.item():.6f}")