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

import gpytorch
import numpy as np
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel, CylindricalKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
#from botorch.models import SingleTaskGP, FixedNoiseGP


# GP Model
class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=2.5)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def map_box_ball(x, dim):
    #dim = x.shape[1]
    # from borders to [-1, 1]^d
    x = (x - 0.5) * 2
    # from [-1, 1]^d to Ball(0, 1)
    x = x / np.sqrt(dim)
    return x


def map_ball_box(x, dim):
    #dim = len(borders)
    # from Ball(0, 1) to [-1, 1]^d
    x = np.sqrt(dim) * x
    # from [-1, 1]^d to borders
    x = x * 0.5 + 0.5
    return x


class KumaAlphaPrior(gpytorch.priors.Prior):
    def __init__(self):
        super(KumaAlphaPrior, self).__init__()
        self.log_a_max = np.log(2)
        pass

    def log_prob(self, x):
        x = torch.log(x)
        loc = torch.tensor(0.).to(x)
        scale = torch.tensor(0.01).to(x)
        return torch.sum(torch.log(
            torch.distributions.Normal(loc=loc, scale=scale).log_prob(x).exp() + 0.5 / self.log_a_max
        ))


class KumaBetaPrior(gpytorch.priors.Prior):
    def __init__(self):
        super(KumaBetaPrior, self).__init__()
        self.log_b_max = np.log(2)
        pass

    def log_prob(self, x):
        x = torch.log(x)
        loc = torch.tensor(0.).to(x)
        scale = torch.tensor(0.01).to(x)
        return torch.sum(torch.log(
            torch.distributions.Normal(loc=loc, scale=scale).log_prob(x).exp() + 0.5 / self.log_b_max
        ))


class AngularWeightsPrior(gpytorch.priors.Prior):
    def __init__(self):
        super(AngularWeightsPrior, self).__init__()

    def log_prob(self, x):
        x = torch.log(x)
        loc = torch.tensor(0.).to(x)
        scale = torch.tensor(2.).to(x)
        return torch.distributions.Normal(loc=loc, scale=scale).log_prob(x).sum()


class CustomCylindricalGP(ExactGP):  # FixedNoiseGP SingleTaskGP
    def __init__(self, train_X, train_Y, likelihood, dim, lengthscale_constraint, outputscale_constraint, ard_dims):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y, likelihood)  # GaussianLikelihood())  # GaussianLikelihood() noise.squeeze(-1)
        self.dim = dim
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(CylindricalKernel(
            num_angular_weights=ard_dims,
            alpha_prior=KumaAlphaPrior(),
            alpha_constraint=gpytorch.constraints.constraints.Interval(lower_bound=0.5, upper_bound=1.),
            beta_prior=KumaBetaPrior(),
            beta_constraint=gpytorch.constraints.constraints.Interval(lower_bound=1., upper_bound=2.),
            radial_base_kernel=MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=1, nu=2.5),
            # angular_weights_constraint=gpytorch.constraints.constraints.Interval(lower_bound=np.exp(-12.),
            #                                                                      upper_bound=np.exp(20.)),
            angular_weights_prior=AngularWeightsPrior()
        ))
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        x = map_box_ball(x, self.dim)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_ard, num_steps, hypers={}, use_cylinder=True, dim=1):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    noise_constraint = Interval(5e-4, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    ard_dims = train_x.shape[1] if use_ard else None
    if use_cylinder:
        model = CustomCylindricalGP(
            train_X=train_x,
            train_Y=train_y,
            likelihood=likelihood,
            dim=dim,
            lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
            ard_dims=ard_dims,
        ).to(device=train_x.device, dtype=train_x.dtype)
    else:
        model = GP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
            ard_dims=ard_dims,
        ).to(device=train_x.device, dtype=train_x.dtype)


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        if not use_cylinder:
            hypers["covar_module.outputscale"] = 1.0
            hypers["covar_module.base_kernel.lengthscale"] = 0.5
            hypers["likelihood.noise"] = 0.005
        model.initialize(**hypers)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model
