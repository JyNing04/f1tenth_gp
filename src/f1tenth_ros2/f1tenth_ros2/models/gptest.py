import torch
import gpytorch
import math
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np


def franke(X, Y):
    term1 = .75*torch.exp(-((9*X - 2).pow(2) + (9*Y - 2).pow(2))/4)
    term2 = .75*torch.exp(-((9*X + 1).pow(2))/49 - (9*Y + 1)/10)
    term3 = .5*torch.exp(-((9*X - 7).pow(2) + (9*Y - 3).pow(2))/4)
    term4 = .2*torch.exp(-(9*X - 4).pow(2) - (9*Y - 7).pow(2))

    f = term1 + term2 + term3 - term4
    dfx = -2*(9*X - 2)*9/4 * term1 - 2*(9*X + 1)*9/49 * term2 + \
          -2*(9*X - 7)*9/4 * term3 + 2*(9*X - 4)*9 * term4
    dfy = -2*(9*Y - 2)*9/4 * term1 - 9/10 * term2 + \
          -2*(9*Y - 3)*9/4 * term3 + 2*(9*Y - 7)*9 * term4

    return f, dfx, dfy

xv, yv = torch.meshgrid([torch.linspace(0, 1, 10), torch.linspace(0, 1, 10)])
train_x = torch.cat((
    xv.contiguous().view(xv.numel(), 1),
    yv.contiguous().view(yv.numel(), 1)),
    dim=1
)

f, dfx, dfy = franke(train_x[:, 0], train_x[:, 1])
train_y = torch.stack([f, dfx, dfy], -1).squeeze(1)

train_y += 0.05 * torch.randn(train_y.size()) # Add noise to both values and gradients
print(train_y.size())
class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)  # Value + x-derivative + y-derivative
model = GPModelWithDerivatives(train_x, train_y, likelihood)

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print(loss)
    print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.squeeze()[0],
        model.covar_module.base_kernel.lengthscale.squeeze()[1],
        model.likelihood.noise.item()
    ))
    optimizer.step()