import torch
import gpytorch

class PGLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    PGLikelihood: Custom likelihood for Gaussian process classification with Pòlya-Gamma data augmentation.

    This likelihood is based on the paper:
    Florian Wenzel, Theo Galy-Fajou, Christan Donner, Marius Kloft, Manfred Opper.
    "Efficient Gaussian process classification using Pòlya-Gamma data augmentation."
    Proceedings of the AAAI Conference on Artificial Intelligence. 2019.
    and the implementation in the GPyTorch documentation
    https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/PolyaGamma_Binary_Classification.html
    """

    def expected_log_prob(self, target, input):
        """
        Compute the expected log likelihood contribution.

        Parameters:
            - target: Target values.
            - input: Input distribution.

        Returns:
            Expected log likelihood contribution.
        """
        mean, variance = input.mean, input.variance
        # Compute the expectation E[f_i^2]
        raw_second_moment = variance + mean.pow(2)

        # Translate targets to be -1, 1
        target = target.to(mean.dtype).mul(2.).sub(1.)

        # We detach the following variable since we do not want
        # to differentiate through the closed-form PG update.
        c = raw_second_moment.detach().sqrt()

        # Compute mean of PG auxiliary variable omega: 0.5 * Expectation[omega]
        # See Eqn (11) and Appendix A2 and A3 ref above for details.
        half_omega = 0.25 * torch.tanh(0.5 * c) / c

        # Expected log likelihood
        res = 0.5 * target * mean - half_omega * raw_second_moment
        # Sum over data points in mini-batch
        res = res.sum(dim=-1)
        return res

    def forward(self, function_samples):
        """
        Define the likelihood.

        Parameters:
            - function_samples: Function samples.

        Returns:
            Bernoulli distribution.
        """
        return torch.distributions.Bernoulli(logits=function_samples)

    def marginal(self, function_dist):
        """
        Define the marginal likelihood using Gauss Hermite quadrature.

        Parameters:
            - function_dist: Function distribution.

        Returns:
            Bernoulli distribution.
        """
        # Define a lambda function to compute Bernoulli probabilities from function samples
        prob_lambda = lambda function_samples: self.forward(function_samples).probs
        # Use Gauss Hermite quadrature to compute marginal likelihood probabilities
        probs = self.quadrature(prob_lambda, function_dist)
        return torch.distributions.Bernoulli(probs=probs)

class GPModel(gpytorch.models.ApproximateGP):
    """
    GPModel: Gaussian process model with binary emission distribution.

    This model uses PGLikelihood and includes inducing points, mean, and covariance modules.
    """

    def __init__(self, inducing_points):
        """
        Initialize the GPModel.

        Parameters:
            - inducing_points: Inducing points for the variational distribution.
        """
        # Set up the variational distribution and strategy with inducing points
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        # Initialize the GPModel using the variational strategy
        super(GPModel, self).__init__(variational_strategy)
        # Set the mean module (zero mean in this case)
        self.mean_module = gpytorch.means.ZeroMean()
        # Set the covariance module (ScaleKernel with RBFKernel)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """
        Forward pass through the GPModel.

        Parameters:
            - x: Input data.

        Returns:
            Multivariate normal distribution.
        """
        # Compute the mean and covariance of the GPModel
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Return the MultivariateNormal distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
