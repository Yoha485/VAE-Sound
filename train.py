from tensorflow.keras.datasets import mnist
import numpy as np
import os

from vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150

SPECTROGRAMS_PATH = './free-spoken-digit-dataset/spectrogram'

def load_fsdd(spectrograms_path):
  x_train = []
  for root, _, file_names in os.walk(spectrograms_path):
    for file_name in file_names:
      file_path = root + '/' + file_name
      spectrogram = np.load(file_path)
      x_train.append(spectrogram)
  x_train = np.array(x_train)
  x_train = x_train[..., np.newaxis]
  print(x_train.shape)
  return x_train  

def load_mnist():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.astype('float32') / 255
  x_train = np.expand_dims(x_train, axis=3)

  x_test = x_test.astype('float32') / 255
  x_test = np.expand_dims(x_test, axis=3)

  return x_train, y_train, x_test, y_test

def train(x_train, learning_rate, batch_size, epochs):
  autoencoder = VAE(input_shape=(256,64,1), conv_filters=(512, 256, 128, 64, 32), conv_kernels=(3, 3, 3, 3, 3), conv_strides=(2, 2, 2, 2, (2,1)), latent_space_dim=128)
  # autoencoder = Autoencoder(input_shape=(28,28,1), conv_filters=(32, 64, 64, 64), conv_kernels=(3, 3, 3, 3), conv_strides=(1, 2, 2, 1), latent_space_dim=2)
  autoencoder.summary()
  autoencoder.compile(learning_rate)
  autoencoder.train(x_train, batch_size, epochs)
  return autoencoder
  

if __name__ == "__main__":
  # x_train, _, _, _ = load_mnist()
  # autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
  # autoencoder.save("model")
  x_train = load_fsdd(SPECTROGRAMS_PATH)
  autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
  autoencoder.save("model")