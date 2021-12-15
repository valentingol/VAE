from jax import jit, numpy as jnp, random
from matplotlib import pyplot as plt
import numpy as np

def plot_images(axes, generate, img_shape,  params, latent_dim, key,
                num_images=9, block=False):
    imgs = generate.apply(params, key, latent_dim, img_shape, n=num_images)
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap='Greys_r')
        ax.axis('off')
    plt.show(block=block)
    if not block:
        plt.pause(0.5)


def plot_latent(generate, img_shape, params, key, n, latent_dim, im_size=28):
    """Plots n x n digit images decoded from the latent space."""
    norm = random.normal(key, shape=(n, ))
    grid_x = jnp.quantile(norm, np.linspace(0.05, 0.95, n))
    grid_y = jnp.quantile(norm, np.linspace(0.05, 0.95, n))
    image_width = im_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = jnp.array([[xi, yi] + [0.0] * (latent_dim - 2)])
            x = generate.apply(params, None, latent_dim, img_shape, z=z)
            digit = jnp.reshape(x[0], (im_size, im_size))
            image[i * im_size: (i + 1) * im_size,
                j * im_size: (j + 1) * im_size] = np.array(digit)

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.show()
