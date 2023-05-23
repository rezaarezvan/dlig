import matplotlib.pyplot as plt
import numpy as np


def sample_images(generator, image_grid_rows=4, image_grid_columns=4, z_dim=100):
    """
    Generate and plot images using the generator model.

    Parameters:
    generator (Model): The generator.
    image_grid_rows (int): The number of rows of the image grid.
    image_grid_columns (int): The number of columns of the image grid.
    z_dim (int): The dimension of the random noise vector.
    """
    # Generate images from random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    gen_imgs = generator.predict(z)

    # Rescale image pixel values from [-1, 1] to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    # Output a grid of images
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[i*image_grid_columns + j])
            axs[i, j].axis('off')

    plt.show()

