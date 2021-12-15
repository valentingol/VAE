from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def plot_images(axes, model, num_images=9, block=False):
    imgs = model.generate(n=num_images)
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap='Greys_r')
        ax.axis('off')
    plt.show(block=block)
    if not block:
        plt.pause(0.5)


def plot_latent(model, n, latent_dim, im_size=28):
    """Plots n x n digit images decoded from the latent space."""

    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = im_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi] + [0.0] * (latent_dim - 2)])
            x = model.generate(z)
            digit = tf.reshape(x[0], (im_size, im_size))
            image[i * im_size: (i + 1) * im_size,
                j * im_size: (j + 1) * im_size] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.show()
