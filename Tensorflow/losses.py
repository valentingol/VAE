import tensorflow as tf

@tf.function
def KL(mean, logits_var):
    var = tf.exp(logits_var)
    mean_square = tf.square(mean)
    # KL term acording to https://arxiv.org/pdf/1606.05908.pdf
    kl_term = 0.5 * tf.reduce_sum(- var - mean_square + logits_var,
                                  axis=-1)
    return kl_term


@tf.function
def conditional_cross_entropy(x_logits, mean, logits_var, z, x, alpha):
    kl_term = KL(mean, logits_var)
    cross_entr_func = tf.nn.sigmoid_cross_entropy_with_logits
    cross_entr = tf.reduce_sum(
        cross_entr_func(logits=x_logits, labels=x),
        axis=[-1, -2, -3]
        )
    return cross_entr - alpha * kl_term


@tf.function
def conditional_mse(x_logits, mean, logits_var, z, x, alpha):
    kl_term = KL(mean, logits_var)
    x_rec = tf.nn.sigmoid(x_logits)
    mse = tf.reduce_sum(
        tf.square(x_rec - x),
        axis=[-1, -2, -3]
        )
    return mse - alpha * kl_term


def elbo(x_logits, mean, logits_var, z, x, alpha):
    cross_entr_func = tf.nn.sigmoid_cross_entropy_with_logits
    def log_normal_pdf(sample, mean, logits_var, raxis=1):
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logits_var) + logits_var),
            axis=raxis
            )
    cross_ent = cross_entr_func(logits=x_logits, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logits_var)
    return - logpx_z - logpz + logqz_x
