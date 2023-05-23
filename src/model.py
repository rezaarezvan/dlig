import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Size of the noise vector, used as input to the Generator
z_dim = 100


def build_generator(z_dim):
    """
    Builds a generator model using the hyperparameter values defined above.
    """
    model = Sequential()

    # Fully connected layer
    model.add(tf.keras.layers.Dense(256 * 8 * 8, input_dim=z_dim))

    # Reshape layer
    model.add(tf.keras.layers.Reshape((8, 8, 256)))

    # Transposed Convolution Layer
    model.add(tf.keras.layers.Conv2DTranspose(
        128, kernel_size=3, strides=2, padding='same'))

    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Transposed Convolution Layer
    model.add(tf.keras.layers.Conv2DTranspose(
        64, kernel_size=3, strides=1, padding='same'))

    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Transposed Convolution Layer
    model.add(tf.keras.layers.Conv2DTranspose(
        3, kernel_size=3, strides=2, padding='same'))

    # TanH activation
    model.add(tf.keras.layers.Activation('tanh'))

    return model


def build_discriminator(img_shape):
    """
    Builds a discriminator model that follows the architecture for CNN from the DCGAN paper.
    """
    model = Sequential()

    # Convolutional Layer
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2,
                                     input_shape=img_shape, padding='same'))

    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Convolutional Layer
    model.add(tf.keras.layers.Conv2D(
        64, kernel_size=3, strides=2, padding='same'))

    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Convolutional Layer
    model.add(tf.keras.layers.Conv2D(
        128, kernel_size=3, strides=2, padding='same'))

    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

    # Flatten the tensor
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model


def build_gan(generator, discriminator):
    """
    Builds a GAN that chains the generator and the discriminator.
    """
    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model


def compile_gan(gan, discriminator, generator, learning_rate=0.0002):
    """
    Compiles the GAN with appropriate optimizer and loss function.
    The discriminator's trainable is set to False during the compilation
    of the GAN model so that the generator part can be trained.
    """
    # Compile the Discriminator
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=Adam(learning_rate),
                          metrics=['accuracy'])

    # Compile the GAN
    discriminator.trainable = False
    gan.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate))
