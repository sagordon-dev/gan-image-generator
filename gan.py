"""
This module contains the implementation of a Generative Adversarial Network 
(GAN) using TensorFlow and Keras.

The GAN is composed of a generator and a discriminator, both implemented as 
Keras models. The generator takes a random noise vector as input and produces 
an image, while the discriminator takes an image as input and classifies it 
as real or fake.

The module provides functions to build the generator, the discriminator, and 
the complete GAN, as well as to train the GAN on the MNIST dataset and 
generate and save images.

Functions:
    build_generator(latent_dim: int) -> models.Model
    build_discriminator(img_shape: tuple) -> models.Model
    build_gan(generator: models.Model, discriminator: models.Model) -> models.
        Model
    train_gan(generator: models.Model, discriminator: models.Model, 
        gan: models.Model, mnist_data: np.array, latent_dim: int, 
        epochs: int, batch_size: int, sample_interval: int)
    generate_and_save_images(generator: models.Model, epoch: int, 
        test_input: np.array)
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_generator(latent_dim):
    """
    Build the generator part of the GAN.

    Args:
        latent_dim (int): The size of the random noise vector used as input 
        for the generator.

    Returns:
        model: A Keras model representing the generator.
    """
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))

    model.add(layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(1, (7, 7), activation='tanh', padding='same'))
    return model


def build_discriminator(img_shape):
    """
    Build the discriminator part of the GAN.

    Args:
        img_shape (tuple): The shape of the input images.

    Returns:
        model: A Keras model representing the discriminator.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2),
              padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    """
    Build the complete GAN by stacking the generator and discriminator.

    Args:
        generator (Keras model): The generator model.
        discriminator (Keras model): The discriminator model.

    Returns:
        model: A Keras model representing the complete GAN.
    """
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


def load_mnist():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        X_train (numpy array): The preprocessed training images.
    """
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1)
    return X_train


def train_gan(generator, discriminator, gan, mnist_data, latent_dim, 
              epochs=10000, batch_size=128, sample_interval=1000):
    """
    Train the GAN on the MNIST data.

    Args:
        generator (Keras model): The generator model.
        discriminator (Keras model): The discriminator model.
        gan (Keras model): The complete GAN model.
        mnist_data (numpy array): The preprocessed training images.
        latent_dim (int): The size of the random noise vector used as input 
        for the generator.
        epochs (int, optional): The number of epochs to train for. Defaults 
        to 10000.
        batch_size (int, optional): The batch size. Defaults to 128.
        sample_interval (int, optional): The interval at which to save 
        generated images. Defaults to 1000.
    """
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, mnist_data.shape[0], half_batch)
        imgs = mnist_data[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(
            imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(
            gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress and save generated images at specified intervals
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
            save_generated_images(epoch, generator)


def generate_and_save_images(generator, epoch, test_input):
    """
    Generate and save images using the generator and display them using matplotlib.

    Args:
        generator (Keras model): The generator model.
        epoch (int): The current epoch.
        test_input (numpy array): A random noise vector used as input for the 
        generator.
    """
    predictions = generator(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.show()

def save_generated_images(epoch, generator, latent_dim=100, examples=100, dim=(10, 10), figsize=(10, 10)):
    """
    Save the images generated by the generator.

    Args:
        epoch (int): The current epoch.
        generator (Keras model): The generator model.
        latent_dim (int, optional): The size of the random noise vector used as input for the generator. Defaults to 100.
        examples (int, optional): The number of examples to generate. Defaults to 100.
        dim (tuple, optional): The dimensions of the grid of images. Defaults to (10, 10).
        figsize (tuple, optional): The size of the figure. Defaults to (10, 10).
    """
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')
    

if __name__ == "__main__":
    # Define the size of the random vector used as input for the generator
    latent_dim = 100

    # Build the generator
    generator = build_generator(latent_dim)

    # Build the discriminator
    img_shape = (28, 28, 1)
