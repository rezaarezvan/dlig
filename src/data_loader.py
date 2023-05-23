import tensorflow as tf
from tensorflow.keras.datasets import cifar10


def load_data():
    """
    Loads and preprocesses the CIFAR-10 dataset.
    Normalizes pixel values to be between -1 and 1.

    Returns:
        numpy.ndarray: Preprocessed image data.
    """
    # Load CIFAR-10 dataset
    (train_images, _), (_, _) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0

    # Scale images to be between -1 and 1
    train_images = (train_images - 0.5) * 2

    return train_images
