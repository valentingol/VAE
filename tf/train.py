import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import matplotlib.pyplot as plt
import tensorflow as tf
from time import time

from tf.losses import conditional_cross_entropy
from utils.plot import plot_latent_tf, plot_images
kl = tf.keras.layers

def mnist_img_dataset(batch_size):
    def preprocess(x):
        x = tf.where(x < 255 // 2, 0.0, 1.0) # binarize, float32
        x = tf.reshape(x, [-1, 28, 28, 1]) # (batch_size, 28, 28, 1)
        return x

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    X = tf.concat([x_train, x_test], axis=0)
    ds = tf.data.Dataset.from_tensor_slices(X)
    ds = ds.shuffle(X.shape[0]).batch(batch_size).cache().prefetch(1)
    ds = ds.map(preprocess)
    return ds


class ConvVAE(tf.keras.Model):
    def __init__(self, latent_dim = 10, name=None):
        super(ConvVAE, self).__init__(name=name)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        h, w, c = input_shape[-3:]
        self.encoder = tf.keras.Sequential([
            kl.Conv2D(filters=32, kernel_size=3, strides=2, padding='same'),
            kl.ReLU(),
            kl.Conv2D(filters=64, kernel_size=3, strides=2, padding='same'),
            kl.ReLU(),
            kl.Flatten(),
            kl.Dense(2 * self.latent_dim)
        ])
        h_r, w_r = h // 4, w // 4
        self.decoder = tf.keras.Sequential([
            kl.Dense(units=h_r * w_r * 32, activation='relu'),
            kl.Reshape(target_shape=(h_r, w_r, 32)),
            kl.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                               padding='same'),
            kl.Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                               padding='same'),
            kl.Conv2DTranspose(filters=c, kernel_size=3, strides=1,
                               padding='same'),
        ])

    def encode(self, x):
        return tf.split(self.encoder(x), 2, axis=-1)

    def decode(self, mean, logits_var):
        z = tf.random.normal(tf.shape(mean))
        # exp(logits_var / 2) = sqrt(var) = std
        z = mean + tf.exp(logits_var / 2) * z
        imgs_logits = self.decoder(z)
        return imgs_logits, z

    @tf.function
    def generate(self, z=None, n=None, seed=None):
        if z is None:
            z = tf.random.normal(shape=(n, self.latent_dim), seed=seed) + 1
        return tf.nn.sigmoid(self.decoder(z))

    def call(self, x):
        mean, logits_var = self.encode(x)
        x_logits, z = self.decode(mean, logits_var)
        return x_logits, (mean, logits_var, z)


def train(model, optimizer, loss_func, alpha, dataset, n_epochs=10,
          max_time=None, verbose=True, gen_imgs=False, num_images=9):
    max_time = max_time or 1e9 # ~ infinity iif max_time = 0 or None
    # Create figure
    side = int(num_images ** 0.5)
    _, axes = plt.subplots(side, side, figsize=(8, 8))

    @tf.function
    def update(model, X_batch):
        with tf.GradientTape() as tape:
            X_logits, (mean, logits_var, z) = model(X_batch)
            loss_batch = loss_func(X_logits, mean, logits_var, z, X_batch,
                                   alpha)
            loss = tf.reduce_mean(loss_batch)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mean, tf.exp(logits_var)

    n_batch = len(dataset)
    start_t = time()
    for epoch in range(n_epochs):
        mean_loss = 0.0
        if verbose:
            print(f'\nEpoch {epoch + 1}/{n_epochs}')
        for i_batch, X_batch in enumerate(dataset):
            loss, mean, var = update(model, X_batch)
            mean_loss = mean_loss * i_batch + loss.numpy()
            mean_loss /= (i_batch + 1)
            if verbose:
                t = time() - start_t
                t_1batch = (t / (epoch * n_batch + (i_batch + 1)))
                eta = ((n_epochs - epoch + 1) * n_batch - i_batch) * t_1batch
                eta_str = f'{eta // 60:.0f}m {eta % 60:.0f}s'
                mean = tf.square(mean).numpy().mean()
                var = var.numpy().mean()
                print(f'  batch: {i_batch + 1}/{n_batch} '
                      f'- loss: {mean_loss: .4f} '
                      f'- eta: {eta_str} '
                      f'- mean: {mean: .3f} '
                      f'- var:{var: .3f}    ', end='\r')
            if time() - start_t > max_time:
                # Time limit reached: skip remaining batches
                break
        if gen_imgs:
            plot_images(axes, model, num_images, block=False)
        if time() - start_t > max_time:
            # Time limit reached: skip remaining epochs
            if verbose: print()
            print('Time limit reached.')
            break

    plot_images(axes, model, num_images, block=True)
    return mean_loss


if __name__ == '__main__':
    ## Configs
    # alpha: weight of KL term in loss
    # max_time: in sec, 0 or None for no limit
    # gen_imgs: if True, generate images at every epochs
    name = 'conv_vae'
    save = False

    latent_dim = 3
    learning_rate = 1e-3
    alpha = 0.2
    batch_size = 512
    n_epochs = 10
    max_time = 20

    verbose = False
    gen_imgs = True
    num_images = 25

    # Model, optimizer and loss function
    vae = ConvVAE(latent_dim=latent_dim, name=name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_func = conditional_cross_entropy

    vae.compile(optimizer=optimizer)
    dataset = mnist_img_dataset(batch_size=batch_size)
    train(model=vae,
          optimizer=optimizer,
          loss_func=loss_func,
          alpha=alpha,
          dataset=dataset,
          n_epochs=n_epochs,
          max_time=max_time,
          verbose=verbose,
          gen_imgs=gen_imgs,
          num_images=num_images)

    plot_latent_tf(vae, 20, latent_dim, im_size=28)

    if save:
        vae.save('./tf/models/' + name, save_format='tf')
