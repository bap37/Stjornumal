import jax
import jax.numpy as jnp
import jax.scipy as jsp

##############
# jax-compatible functions live in this code block; must start with "Dist"
##############

def DistGaussian(key, theta, correlation):
    """
    Generate Gaussian samples using inverse transform sampling.

    key:         JAX PRNG key
    theta:       Array of shape (batch_size, 2) -> [mu, sigma]
    correlation: Unused for now

    Returns:
        Gaussian samples of shape (batch_size, N)
    """

    theta = jnp.asarray(theta, dtype=jnp.float32)

    if theta.ndim == 1:
        theta = theta[None, :]


    # Generate uniform random variables
    u = jax.random.uniform(
        key,
        minval=1e-7,
        maxval=1.0 - 1e-7
    )

    # Extract Gaussian parameters
    mu = theta[:, 0:1]
    sigma = theta[:, 1:2]

    # Inverse transform sampling (probit)
    z = jsp.special.ndtri(u)

    # Transform into Gaussian samples
    x = mu + sigma * z

    return x.squeeze()

def DistGaussian_EVOL(key, theta, correlation):
    """
    Generate a single Gaussian random value with evolving mean.

    theta:       [mu0, sigma, m]
    correlation: Scalar correlation value

    Returns:
        Scalar JAX array
    """

    theta = jnp.asarray(theta, dtype=jnp.float32)
    correlation = jnp.asarray(correlation, dtype=jnp.float32)

    u = jax.random.uniform(
        key,
        shape=(),
        minval=1e-7,
        maxval=1.0 - 1e-7
    )

    z = jsp.special.ndtri(u)

    mu0, sigma, m = theta.reshape(-1)[0], theta.reshape(-1)[1], theta.reshape(-1)[2]

    mu = mu0 + m * correlation

    return (mu + sigma * z).squeeze()

def DistExponential(key, theta, correlation):
    """
    Generate a single exponential random value using inverse transform sampling.

    theta:       [tau]
    correlation: Unused

    Returns:
        Scalar JAX array
    """

    theta = jnp.asarray(theta, dtype=jnp.float32)

    u = jax.random.uniform(
        key,
        shape=(),
        minval=1e-7,
        maxval=1.0 - 1e-7
    )

    tau = theta.reshape(-1)[0]

    x = -tau * jnp.log(1.0 - u)

    return x.squeeze()

def DistExponential_EVOL(key, theta, correlation):
    """
    Generate a single exponential random value with evolving scale.

    theta:       [tau0, m]
    correlation: Scalar correlation value

    Returns:
        Scalar JAX array
    """

    theta = jnp.asarray(theta, dtype=jnp.float32)
    correlation = jnp.asarray(correlation, dtype=jnp.float32)

    u = jax.random.uniform(
        key,
        shape=(),
        minval=1e-7,
        maxval=1.0 - 1e-7
    )

    tau0, m = theta.reshape(-1)[0], theta.reshape(-1)[1]

    tau = tau0 + m * correlation

    x = -tau * jnp.log(1.0 - u)

    return x.squeeze()

def DistDoubleGaussian(key, theta, correlation):
    """
    Generate a single random value from a double Gaussian mixture.

    theta:       [mu1, sigma1, mu2, sigma2, a]
    correlation: Unused

    Returns:
        Scalar JAX array
    """

    theta = jnp.asarray(theta, dtype=jnp.float32).reshape(-1)

    mu1, sigma1, mu2, sigma2, a = theta

    # Generate uniform random variable
    u = jax.random.uniform(
        key,
        shape=(),
        minval=1e-7,
        maxval=1.0 - 1e-7
    )

    # Mixture CDF
    def mixture_cdf(x):
        G1 = jsp.special.ndtr((x - mu1) / sigma1)
        G2 = jsp.special.ndtr((x - mu2) / sigma2)
        return (G1 + a * G2) / (1.0 + a)

    # Establish numerical search bounds
    lower = jnp.minimum(mu1 - 10.0 * sigma1, mu2 - 10.0 * sigma2)
    upper = jnp.maximum(mu1 + 10.0 * sigma1, mu2 + 10.0 * sigma2)

    # Bisection to invert mixture CDF
    def bisect(i, bounds):
        lo, hi = bounds
        mid = (lo + hi) / 2.0

        cdf_mid = mixture_cdf(mid)

        return jax.lax.cond(
            cdf_mid < u,
            lambda _: (mid, hi),
            lambda _: (lo, mid),
            operand=None
        )

    lower, upper = jax.lax.fori_loop(
        0, 100, bisect, (lower, upper)
    )

    return ((lower + upper) / 2.0).squeeze()

def DistLogistic(key, theta, correlation):
    """
    Generate a single Gaussian random value with a logistic-shaped evolving mean.

    theta:       [L, k, sigma]
    correlation: Scalar correlation value

    Returns:
        Scalar JAX array
    """

    theta = jnp.asarray(theta, dtype=jnp.float32).reshape(-1)
    correlation = jnp.asarray(correlation, dtype=jnp.float32)

    u = jax.random.uniform(
        key,
        shape=(),
        minval=1e-7,
        maxval=1.0 - 1e-7
    )

    z = jsp.special.ndtri(u)

    L, k, sigma = theta[0], theta[1], theta[2]

    # Logistic-shaped evolving mean
    mean = L / (1.0 + jnp.exp(k * correlation - 10.0)) + 2.0

    # Gaussian sample
    x = mean + sigma * z

    return x.squeeze()

def DistGamma(key, theta, correlation):
    """
    Generate a single Gamma random value using inverse transform sampling.

    theta:       [tau, gamma]
    correlation: Unused

    Returns:
        Scalar JAX array
    """

    theta = jnp.asarray(theta, dtype=jnp.float32).reshape(-1)

    tau, gamma = theta[0], theta[1]

    # Generate uniform random variable
    u = jax.random.uniform(
        key,
        shape=(),
        minval=1e-7,
        maxval=1.0 - 1e-7
    )

    # Convert scale to rate
    lambda_target = 1.0 / tau

    # Gamma CDF
    def gamma_cdf(x):
        return jsp.special.gammainc(
            gamma,
            x * lambda_target
        )

    # Establish numerical search bounds
    lower = jnp.array(0.0, dtype=jnp.float32)
    upper = tau * (gamma + 10.0 * jnp.sqrt(gamma) + 10.0)

    # Bisection to invert Gamma CDF
    def bisect(i, bounds):
        lo, hi = bounds
        mid = (lo + hi) / 2.0

        cdf_mid = gamma_cdf(mid)

        return jax.lax.cond(
            cdf_mid < u,
            lambda _: (mid, hi),
            lambda _: (lo, mid),
            operand=None
        )

    lower, upper = jax.lax.fori_loop(
        0, 100, bisect, (lower, upper)
    )

    return ((lower + upper) / 2.0).squeeze()

def DistIdentity(key, theta, correlation):

    theta = jnp.asarray(theta, dtype=jnp.float32)

    return theta.reshape(())


def host_step(key, theta, correlation):
    correlated_value = correlation[0]
    threshold = correlation[1]

    return jnp.where(
        correlated_value > threshold,
        -theta / 2,
        theta / 2
    )

def Selection_Spline(x,theta, basis):
    import torch
    """
    theta : (B, K)
    basis : (N, K)

    Returns
    -------
    weights : (B, N)
    """

    weights = theta @ basis.T

    # Turn into probabilities
    return torch.sigmoid(weights)

