###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from gp import train_gp
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube


class Turbo1:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        use_cylinder=False,
        budget=16*8,
        use_decay=False,
        decay_threshold=0.5,
        decay_alpha=0.8,
        use_pull=0,
        use_lcb=0,
        kappa=2.0,
        length_min=0.5**7,
        length_max=1.8,
        length_init=0.8,
        length_multiplier=2.0,
        used_budget=0
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        #cylinder
        self.use_cylinder = use_cylinder

        #decay
        self.budget = budget
        self.used_budget = used_budget
        self.use_decay = use_decay
        self.decay_alpha = decay_alpha
        self.decay_threshold = decay_threshold

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # pull
        self.use_pull = use_pull
        self.prob_pull = np.ones((0, self.dim)) / self.dim

        #lcb
        self.use_lcb = use_lcb
        self.kappa = kappa

        # Tolerances and counters
        self.n_cand = min(100 * self.dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = length_min
        self.length_max = length_max
        self.length_init = length_init
        self.length_multiplier = length_multiplier

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self._predictions = []
        self.failcount = 0
        self.succcount = 0
        self.initial = 1
        self.pull = 1
        self.length = self.length_init
        self.prob_pull = np.ones(self.dim) / self.dim
        self.prob_push = np.ones(self.dim) / self.dim
        self.init_iter = True
        #print(self.prob_pull)

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1
        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([self.length_multiplier * self.length, self.length_max])
            self.succcount = 0
            self.pull = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= self.length_multiplier
            self.failcount = 0
            self.pull = 1
        print('Use or not decay: ', self.use_decay)
        if self.use_decay:
            print(self.used_budget)
            if self.used_budget > self.decay_threshold * self.budget:
                print("Applying decay...")
                self.length *= self.decay_alpha #* min(np.random.lognormal(1, 2, 1), 1)

        diff_std = np.std(self.X - self.X[np.argmin(self.fX)], axis=0)
        self.prob_pull = np.exp(diff_std) / np.exp(
            np.std(self.X - self.X[np.argmin(self.fX)], axis=0)).sum()
        c = 0.1 # regularizer
        self.prob_push = np.exp(diff_std.max() - diff_std) / np.exp(
            diff_std.max() - diff_std).sum()

    def _create_candidates(self, X, fX, length, n_training_steps, hypers, used_budget=None):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        if used_budget is not None:
            self.used_budget = used_budget
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers,
                use_cylinder=self.use_cylinder, dim=self.dim
            )

            # Save state dict
            hypers = gp.state_dict()
        self._errors = self.fX - np.array(self._predictions)
        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        if not self.use_cylinder:
            weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        else:
            #weights = gp.covar_module.base_kernel.radial_base_kernel.lengthscale.cpu().detach().numpy().ravel()
            weights = gp.covar_module.base_kernel.angular_weights.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        #print('weights', weights)
        # TODO: REMOVE
        #prob_pert = np.log(self.budget - len(self.fX)) / np.log(self.budget)
        #print('prob of pulling appliance:', prob_pert)
       # appliance = np.random.choice((1, 0), p=(prob_pert, 1 - prob_pert))
       # print('pull or not: ', appliance)
        if self.use_pull == 1:
            print("Applying pulling...")
            if self.pull:
                print('Prob of pulling:', self.prob_pull)
                to_pull = np.random.choice(range(0,self.dim), size=min(self.dim, 2), p=self.prob_pull.flatten())
                weights[to_pull] *= 2
            else:
                print('Prob of pushing:', self.prob_push)
                to_push = np.random.choice(range(0, self.dim), size=min(self.dim, 2), p=self.prob_push.flatten())
                weights[to_push] /= 2
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)
        #print('lb', lb)
        #print('ub', ub)
        self.cand_lb = lb
        self.cand_ub = ub

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            f_preds = gp.likelihood(gp(X_cand_torch))
            self.f_var = f_preds.variance.cpu().detach().numpy()
            #print(self.f_var.shape)
            y_cand = f_preds.sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()
            #print(y_cand.shape)
        self.gp = deepcopy(gp)
        self.init_iter = False
        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand
        #print(y_cand.shape)
        return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        _y_cand = deepcopy(y_cand)
        if self.use_lcb:
            print("Applying LCB...")
            f_var = np.expand_dims(np.sqrt(self.f_var), 1).repeat(self.batch_size, axis=1)
            #print(f_var.shape)
            #print(_y_cand.shape)
            _y_cand = y_cand - self.kappa * f_var
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(_y_cand[:, i])
            self._predictions.append(_y_cand[indbest, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            _y_cand[indbest, :] = np.inf
        return X_next
