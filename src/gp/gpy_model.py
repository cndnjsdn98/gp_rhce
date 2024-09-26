#!/usr/bin/env python

""" 
    GPyTorch Implementation of the GP Regression model for the data-augmented MPC.
"""

import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                sig_f=None, sig_n=None, l=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if sig_n is None or sig_f is None or l is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel())
        else:
            outputscale_prior = gpytorch.priors.NormalPrior(0, sig_f)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale=torch.Tensor([l])
                ),
                outputscale_prior=outputscale_prior  
            )
            self.covar_module.outputscale = outputscale_prior.variance
            likelihood.noise_covar.register_prior(
                "noise_std_prior",
                gpytorch.priors.NormalPrior(0, sig_n),
                lambda module: module.noise.sqrt()
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, learn_inducing_points=True):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-1))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations= learn_inducing_points
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )