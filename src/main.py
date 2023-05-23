from data_loader import load_data
from model import build_generator, build_discriminator, build_gan, compile_gan
from train import train
from generate import sample_images


def main():
    # Set hyperparameters
    z_dim = 100
    batch_size = 128
    epochs = 50

    # Load dataset
    train_images = load_data()

    # Get the shape of the training images
    img_shape = train_images[0].shape

    # Build and compile the discriminator
    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

    # Build the generator
    generator = build_generator(z_dim)

    # Build and compile the GAN with the generator and the discriminator
    gan = build_gan(generator, discriminator)
    compile_gan(gan, discriminator, generator)

    # Train the GAN
    train(generator, discriminator, gan,
          train_images, batch_size, z_dim, epochs)

    # Generate images using the trained generator
    sample_images(generator)


if __name__ == '__main__':
    main()
