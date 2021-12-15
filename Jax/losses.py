from functools import partial

import jax
from jax import jit, numpy as jnp
import optax

@jit
def KL(mean, logits_var):
    var = jnp.exp(logits_var)
    mean_square = jnp.square(mean)
    # KL term acording to https://arxiv.org/pdf/1606.05908.pdf
    kl_term = 0.5 * jnp.sum(- var - mean_square + logits_var, axis=-1)
    return kl_term


@jit
def conditional_cross_entropy(x_logits, mean, logits_var, z, x, alpha):
    kl_term = KL(mean, logits_var)
    cross_entr_func = optax.sigmoid_binary_cross_entropy
    cross_entr = jnp.sum(cross_entr_func(logits=x_logits, labels=x),
        axis=[-1, -2, -3]
        )
    return jnp.mean(cross_entr - alpha * kl_term)


@jit
def conditional_mse(x_logits, mean, logits_var, z, x, alpha):
    kl_term = KL(mean, logits_var)
    x_rec = jax.nn.sigmoid(x_logits)
    mse = jnp.sum(jnp.square(x_rec - x), axis=[-1, -2, -3])
    return jnp.mean(mse - alpha * kl_term)


@jit
def elbo(x_logits, mean, logits_var, z, x, alpha):
    cross_entr_func = optax.sigmoid_binary_cross_entropy
    def log_normal_pdf(sample, mean, logits_var, axis=1):
        return jnp.sum(
            -0.5 * ((sample - mean) ** 2 * jnp.exp(-logits_var) + logits_var),
            axis=axis
            )
    cross_ent = cross_entr_func(logits=x_logits, labels=x)
    logpx_z = - jnp.sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logits_var)
    return jnp.mean(- logpx_z - logpz + logqz_x)
