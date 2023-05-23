import numpy as np
from generate import sample_images
from tqdm import tqdm


def train(generator, discriminator, gan, dataset, batch_size=128, z_dim=100, epochs=50):
    """
    Train the GAN system.

    Parameters:
    generator (Model): The generator.
    discriminator (Model): The discriminator.
    gan (Model): The whole GAN.
    dataset (ndarray): The training dataset.
    batch_size (int): The training batch size.
    z_dim (int): The dimension of the random noise vector.
    epochs (int): The number of epochs to train for.
    """

    # Labels for real and fake images
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for _ in tqdm(range(len(dataset)//batch_size)):
            # Training the Discriminator

            # Get a random batch of real images
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            imgs = dataset[idx]

            # Generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, z_dim))
            gen_imgs = generator.predict(z)

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(imgs, real)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Training the Generator

            # Generate a batch of noise vectors
            z = np.random.normal(0, 1, (batch_size, z_dim))

            # Train the generator
            g_loss = gan.train_on_batch(z, real)

        print(
            f"d_loss={d_loss[0]}, d_acc={d_loss[1]}, g_loss={g_loss}")

        # Output a sample of generated image
        if epoch % 10 == 0:
            sample_images(generator)
            # save models
            generator.save(f'generator_model_{epoch}.h5')
            discriminator.save(f'discriminator_model_{epoch}.h5')
