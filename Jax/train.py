from functools import partial
from time import time

import haiku as hk
from jax import jit, nn, numpy as jnp, random, value_and_grad as vgrad
import matplotlib.pyplot as plt
import optax
import tensorflow as tf

from Jax.losses import conditional_cross_entropy
from Jax.plot import plot_latent, plot_images

def mnist_img_dataset(batch_size):
    def preprocess(x):
        x = tf.where(x < 255 // 2, 0.0, 1.0) # binarize, float32
        x = tf.reshape(x, [-1, 28, 28, 1]) # (batch_size, 28, 28, 1)
        return x

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    X = tf.concat([x_train, x_test], axis=0)
    ds = tf.data.Dataset.from_tensor_slices(X).cache()
    ds = ds.shuffle(X.shape[0]).batch(batch_size).prefetch(1)
    ds = ds.map(preprocess)
    return ds


class ConvVAE(hk.Module):
    def __init__(self, latent_dim = 10, name=None):
        super(ConvVAE, self).__init__(name=name)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        h, w, c = input_shape[-3:]
        self.encoder = hk.Sequential([
            hk.Conv2D(32, kernel_shape=3, stride=2, padding='SAME'),
            nn.relu,
            hk.Conv2D(64, kernel_shape=3, stride=2, padding='SAME'),
            nn.relu,
            hk.Flatten(),
            hk.Linear(2 * self.latent_dim)
        ])
        h_r, w_r = h // 4, w // 4
        self.decoder = hk.Sequential([
            hk.Linear(h_r * w_r * 32),
            nn.relu,
            hk.Reshape((h_r, w_r, 32)),
            hk.Conv2DTranspose(64, kernel_shape=3, stride=2,
                               padding='SAME'),
            nn.relu,
            hk.Conv2DTranspose(32, kernel_shape=3, stride=2,
                               padding='SAME'),
            nn.relu,
            hk.Conv2DTranspose(c, kernel_shape=3, stride=1,
                               padding='SAME'),
        ])

    def encode(self, x):
        return jnp.split(self.encoder(x), 2, axis=-1)

    def decode(self, mean, logits_var):
        z = random.normal(hk.next_rng_key(), jnp.shape(mean))
        # exp(logits_var / 2) = sqrt(var) = std
        z = mean + jnp.exp(logits_var / 2) * z
        imgs_logits = self.decoder(z)
        return imgs_logits, z

    def generate(self, img_shape, z=None, n=None):
        def decoder(x):
            self.build(img_shape)
            return self.decoder(x)

        if z is None:
            z = random.normal(hk.next_rng_key(), shape=(n, self.latent_dim))
        return nn.sigmoid(decoder(z))

    def __call__(self, x):
        self.build(jnp.shape(x))
        mean, logits_var = self.encode(x)
        x_logits, z = self.decode(mean, logits_var)
        return x_logits, (mean, logits_var, z)


@hk.transform
def generate(latent_dim, img_shape, z=None, n=None):
    vae = ConvVAE(latent_dim=latent_dim)
    return vae.generate(img_shape=img_shape, z=z, n=n)


@hk.transform
def fwd(X):
    vae = ConvVAE(latent_dim=latent_dim)
    return vae(X)


def init_model(key, optimizer, X_batch):
    params = fwd.init(key, X_batch)
    opt_state = optimizer.init(params)
    return params, opt_state


def train(key, optimizer, loss_func, alpha, dataset, n_epochs=10,
          max_time=None, verbose=True, gen_imgs=False, num_images=9):
    max_time = max_time or 1e9 # ~ infinity iif max_time = 0 or None
    # Create figure
    side = int(num_images ** 0.5)
    _, axes = plt.subplots(side, side, figsize=(8, 8))

    # Initialization
    X_batch = next(dataset.as_numpy_iterator())
    img_shape = X_batch.shape
    params, opt_state = init_model(key, optimizer, X_batch)

    @jit
    def update(params, opt_state, key, X_batch, alpha):
        def fwd_loss(params, key, X_batch, alpha):
            X_logits, (mean, logits_var, z) = fwd.apply(params, key, X_batch)
            loss = loss_func(X_logits, mean, logits_var, z, X_batch, alpha)
            return loss, (mean, logits_var)

        (loss, (mean, logits_var)), grads = vgrad(fwd_loss, has_aux=True)(
            params, key, X_batch, alpha
            )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, mean, jnp.exp(logits_var)

    n_batch = len(dataset)
    start_t = time()
    for epoch in range(n_epochs):
        mean_loss = 0.0
        if verbose:
            print(f'\nEpoch {epoch + 1}/{n_epochs}')
        for i_batch, X_batch in enumerate(dataset):
            key, newkey = random.split(key)
            X_batch = jnp.array(X_batch)
            params, opt_state, loss, mean, var = update(
                params, opt_state, newkey, X_batch, alpha
                )
            mean_loss = (mean_loss * i_batch + loss) / (i_batch + 1)
            if verbose:
                t = time() - start_t
                t_1batch = (t / (epoch * n_batch + (i_batch + 1)))
                eta = ((n_epochs - epoch + 1) * n_batch - i_batch) * t_1batch
                eta_str = f'{eta // 60:.0f}m {eta % 60:.0f}s'
                mean = jnp.square(mean).mean()
                var = var.mean()
                print(f'  batch: {i_batch + 1}/{n_batch} '
                      f'- loss: {mean_loss: .4f} '
                      f'- eta: {eta_str} '
                      f'- mean: {mean: .3f} '
                      f'- var:{var: .3f}    ', end='\r')
            if time() - start_t > max_time:
                # Time limit reached: skip remaining batches
                break
        if gen_imgs:
            key, newkey = random.split(key)
            plot_images(axes, generate, img_shape, params, latent_dim, newkey,
                            num_images, block=False)
        if time() - start_t > max_time:
            # Time limit reached: skip remaining epochs
            if verbose: print()
            print('Time limit reached.')
            break
    if verbose: print()
    key, newkey = random.split(key)
    plot_images(axes, generate, img_shape, params, latent_dim, newkey,
                    num_images, block=True)
    return params, mean_loss


if __name__ == '__main__':
    ## Configs
    # alpha: weight of KL term in loss
    # max_time: in sec, 0 or None for no limit
    # gen_imgs: if True, generate images at every epochs
    name = 'conv_vae'
    save = False

    latent_dim = 2
    learning_rate = 1e-3
    alpha = 0.2
    batch_size = 512
    n_epochs = 10
    max_time = None
    seed = 0

    verbose = True
    gen_imgs = True
    num_images = 25

    # Model, optimizer and loss function
    optimizer = optax.adam(learning_rate=learning_rate)
    loss_func = conditional_cross_entropy

    key1, key2 = random.split(random.PRNGKey(seed))

    dataset = mnist_img_dataset(batch_size=batch_size)
    params = train(key=key1,
                   optimizer=optimizer,
                   loss_func=loss_func,
                   alpha=alpha,
                   dataset=dataset,
                   n_epochs=n_epochs,
                   max_time=max_time,
                   verbose=verbose,
                   gen_imgs=gen_imgs,
                   num_images=num_images)[0]

    # Plot images with an ordered grid of latent vectors from N(0, 1)
    X_batch = next(dataset.as_numpy_iterator())
    img_shape = X_batch.shape
    plot_latent(generate, img_shape, params, key2, 20, latent_dim,
                    im_size=28)

    # if save:
    #     vae.save('./tf/models/' + name, save_format='tf')
