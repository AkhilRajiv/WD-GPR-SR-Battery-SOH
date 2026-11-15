import torch
import torch.nn as nn
import gpytorch
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional

class DeepFeatureExtractor(nn.Module):
    """
    Non-linear neural feature extractor for the deep branch of the GPR.
    Maps 4-dimensional health indicators into a latent space where
    complex electro-chemical degradation patterns become stationary.
    """
    def __init__(self, input_dim: int = 4, latent_dim: int = 4):
        super(DeepFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WideAndDeepGP(gpytorch.models.ApproximateGP):
    """
    Wide-and-Deep Gaussian Process Regression architecture (Problem Statement 2).
    
    Mathematical Formulation:
      - Wide Linear Branch: k_wide(x, x') = x^T Sigma_wide x'
        Captures global linear covariance and first-order linear interactions
        across the raw feature space (Discharge Time, CC Charge Time, Voltage, Resistance).
      - Deep Branch: k_deep(x, x') = sigma_f^2 * Matern_{1.5}(f_theta(x), f_theta(x'))
        Learns non-linear representations through an MLP feature extractor
        with Automatic Relevance Determination (ARD) length-scales.
      - Composite Kernel: k_WD(x, x') = k_wide(x, x') + k_deep(f_theta(x), f_theta(x'))
      - Sparse Variational Inference: Inducing points Z in R^{M x 4} learned jointly
        with theta and kernel hyperparameters by maximizing the Variational ELBO.
    """
    def __init__(self, inducing_points: torch.Tensor, feature_extractor: nn.Module):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(WideAndDeepGP, self).__init__(variational_strategy)

        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()

        # Wide Branch: Linear kernel on raw input space
        self.wide_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )

        # Deep Branch: Matérn (nu=1.5) kernel with ARD on neural latent representations
        self.deep_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=feature_extractor.latent_dim)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        # Constant prior mean
        mean_x = self.mean_module(x)

        # Wide kernel operates on raw normalized inputs
        covar_wide = self.wide_covar(x)

        # Deep kernel operates on non-linear neural latent features
        projected_x = self.feature_extractor(x)
        covar_deep = self.deep_covar(projected_x)

        # Composite Wide-and-Deep covariance
        covar_x = covar_wide + covar_deep

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_wd_gpr(
    train_x: torch.Tensor,
    train_y_scaled: torch.Tensor,
    num_inducing: int = 100,
    num_epochs: int = 150,
    lr: float = 0.01,
    random_seed: int = 42,
    device: torch.device = torch.device('cpu')
) -> Tuple[WideAndDeepGP, gpytorch.likelihoods.GaussianLikelihood, Dict[str, Any]]:
    """
    Trains the Wide-and-Deep GPR jointly using Variational ELBO.
    """
    torch.manual_seed(random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(random_seed)

    # Initialize inducing points strictly from training points
    perm = torch.randperm(train_x.size(0))[:min(num_inducing, train_x.size(0))]
    inducing_points = train_x[perm].clone().to(device)

    # Initialize model components
    feature_extractor = DeepFeatureExtractor(input_dim=train_x.size(1), latent_dim=4).to(device)
    model = WideAndDeepGP(inducing_points, feature_extractor).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    # Joint optimization of NN weights, kernel hyperparameters, and inducing points
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.wide_covar.parameters(), 'lr': lr},
        {'params': model.deep_covar.parameters(), 'lr': lr},
        {'params': model.mean_module.parameters(), 'lr': lr},
        {'params': model.variational_parameters(), 'lr': lr},
        {'params': likelihood.parameters(), 'lr': lr}
    ])

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y_scaled.size(0))

    start_time = time.time()
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y_scaled)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    training_time = time.time() - start_time

    train_info = {
        'epochs': num_epochs,
        'final_loss': losses[-1],
        'training_time': training_time,
        'losses': losses
    }

    return model, likelihood, train_info


def predict_wd_gpr(
    model: WideAndDeepGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    test_x: torch.Tensor,
    y_train_mean: float,
    y_train_std: float,
    device: torch.device = torch.device('cpu')
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes predictive mean and predictive standard deviation,
    re-scaling back to the original physical target domain.
    """
    model.eval()
    likelihood.eval()

    test_x = test_x.to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        # Scaled predictions
        mean_scaled = pred_dist.mean.cpu().numpy()
        std_scaled = pred_dist.stddev.cpu().numpy()

    # Transform back to original target domain (e.g. SOH %)
    y_pred = (mean_scaled * y_train_std) + y_train_mean
    sigma_pred = std_scaled * y_train_std

    return y_pred, sigma_pred


def save_wd_gpr_checkpoint(
    filepath: str,
    model: WideAndDeepGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    metadata: Dict[str, Any]
):
    """
    Saves full model state, feature extractor weights, likelihood, inducing points,
    and experiment configuration for exact reproducibility and Phase 9 XAI reloading.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_extractor_state_dict': model.feature_extractor.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': model.variational_strategy.inducing_points.detach().cpu(),
        'metadata': metadata
    }
    torch.save(checkpoint, filepath)


def load_wd_gpr_checkpoint(
    filepath: str,
    device: torch.device = torch.device('cpu')
) -> Tuple[WideAndDeepGP, gpytorch.likelihoods.GaussianLikelihood, Dict[str, Any]]:
    """
    Loads saved checkpoint and restores model and likelihood ready for inference or XAI.
    """
    checkpoint = torch.load(filepath, map_location=device)
    metadata = checkpoint['metadata']

    inducing_points = checkpoint['inducing_points'].to(device)
    feature_extractor = DeepFeatureExtractor(
        input_dim=metadata.get('input_dim', 4),
        latent_dim=metadata.get('latent_dim', 4)
    ).to(device)

    model = WideAndDeepGP(inducing_points, feature_extractor).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    model.eval()
    likelihood.eval()

    return model, likelihood, metadata


class WideGP(gpytorch.models.ApproximateGP):
    """
    Ablation A: Wide-GPR architecture with linear kernel only.
    k_wide(x, x') = x^T Sigma_wide x'
    """
    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(WideGP, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_wide_gpr(
    train_x: torch.Tensor,
    train_y_scaled: torch.Tensor,
    num_inducing: int = 100,
    num_epochs: int = 150,
    lr: float = 0.01,
    random_seed: int = 42,
    device: torch.device = torch.device('cpu')
) -> Tuple[WideGP, gpytorch.likelihoods.GaussianLikelihood, Dict[str, Any]]:
    torch.manual_seed(random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(random_seed)

    perm = torch.randperm(train_x.size(0))[:min(num_inducing, train_x.size(0))]
    inducing_points = train_x[perm].clone().to(device)

    model = WideGP(inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.covar_module.parameters(), 'lr': lr},
        {'params': model.mean_module.parameters(), 'lr': lr},
        {'params': model.variational_parameters(), 'lr': lr},
        {'params': likelihood.parameters(), 'lr': lr}
    ])

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y_scaled.size(0))

    start_time = time.time()
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y_scaled)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    training_time = time.time() - start_time

    train_info = {
        'epochs': num_epochs,
        'final_loss': losses[-1],
        'training_time': training_time,
        'losses': losses
    }

    return model, likelihood, train_info


def predict_wide_gpr(
    model: WideGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    test_x: torch.Tensor,
    y_train_mean: float,
    y_train_std: float,
    device: torch.device = torch.device('cpu')
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    likelihood.eval()

    test_x = test_x.to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        mean_scaled = pred_dist.mean.cpu().numpy()
        std_scaled = pred_dist.stddev.cpu().numpy()

    y_pred = (mean_scaled * y_train_std) + y_train_mean
    sigma_pred = std_scaled * y_train_std

    return y_pred, sigma_pred


def save_wide_gpr_checkpoint(
    filepath: str,
    model: WideGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    metadata: Dict[str, Any]
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': model.variational_strategy.inducing_points.detach().cpu(),
        'metadata': metadata
    }
    torch.save(checkpoint, filepath)


def load_wide_gpr_checkpoint(
    filepath: str,
    device: torch.device = torch.device('cpu')
) -> Tuple[WideGP, gpytorch.likelihoods.GaussianLikelihood, Dict[str, Any]]:
    checkpoint = torch.load(filepath, map_location=device)
    metadata = checkpoint['metadata']
    inducing_points = checkpoint['inducing_points'].to(device)

    model = WideGP(inducing_points).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    model.eval()
    likelihood.eval()

    return model, likelihood, metadata


class DeepGP(gpytorch.models.ApproximateGP):
    """
    Ablation B: Deep-GPR architecture with neural feature extractor + Matérn 1.5 ARD kernel only.
    k_deep(x, x') = sigma_f^2 * Matern_{1.5}(f_theta(x), f_theta(x'))
    """
    def __init__(self, inducing_points: torch.Tensor, feature_extractor: nn.Module):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(DeepGP, self).__init__(variational_strategy)

        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=feature_extractor.latent_dim)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        projected_x = self.feature_extractor(x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_deep_gpr(
    train_x: torch.Tensor,
    train_y_scaled: torch.Tensor,
    num_inducing: int = 100,
    num_epochs: int = 150,
    lr: float = 0.01,
    random_seed: int = 42,
    device: torch.device = torch.device('cpu')
) -> Tuple[DeepGP, gpytorch.likelihoods.GaussianLikelihood, Dict[str, Any]]:
    torch.manual_seed(random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(random_seed)

    perm = torch.randperm(train_x.size(0))[:min(num_inducing, train_x.size(0))]
    inducing_points = train_x[perm].clone().to(device)

    feature_extractor = DeepFeatureExtractor(input_dim=train_x.size(1), latent_dim=4).to(device)
    model = DeepGP(inducing_points, feature_extractor).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.covar_module.parameters(), 'lr': lr},
        {'params': model.mean_module.parameters(), 'lr': lr},
        {'params': model.variational_parameters(), 'lr': lr},
        {'params': likelihood.parameters(), 'lr': lr}
    ])

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y_scaled.size(0))

    start_time = time.time()
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y_scaled)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    training_time = time.time() - start_time

    train_info = {
        'epochs': num_epochs,
        'final_loss': losses[-1],
        'training_time': training_time,
        'losses': losses
    }

    return model, likelihood, train_info


def predict_deep_gpr(
    model: DeepGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    test_x: torch.Tensor,
    y_train_mean: float,
    y_train_std: float,
    device: torch.device = torch.device('cpu')
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    likelihood.eval()

    test_x = test_x.to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        mean_scaled = pred_dist.mean.cpu().numpy()
        std_scaled = pred_dist.stddev.cpu().numpy()

    y_pred = (mean_scaled * y_train_std) + y_train_mean
    sigma_pred = std_scaled * y_train_std

    return y_pred, sigma_pred


def save_deep_gpr_checkpoint(
    filepath: str,
    model: DeepGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    metadata: Dict[str, Any]
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_extractor_state_dict': model.feature_extractor.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': model.variational_strategy.inducing_points.detach().cpu(),
        'metadata': metadata
    }
    torch.save(checkpoint, filepath)


def load_deep_gpr_checkpoint(
    filepath: str,
    device: torch.device = torch.device('cpu')
) -> Tuple[DeepGP, gpytorch.likelihoods.GaussianLikelihood, Dict[str, Any]]:
    checkpoint = torch.load(filepath, map_location=device)
    metadata = checkpoint['metadata']
    inducing_points = checkpoint['inducing_points'].to(device)
    feature_extractor = DeepFeatureExtractor(
        input_dim=metadata.get('input_dim', 4),
        latent_dim=metadata.get('latent_dim', 4)
    ).to(device)

    model = DeepGP(inducing_points, feature_extractor).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    model.eval()
    likelihood.eval()

    return model, likelihood, metadata
