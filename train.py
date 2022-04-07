from tensorflow.keras.datasets import mnist
import numpy as np

from autoencoder import Autoencoder

LEARNING_RATE = 0.0005
BATCH_SIZE = 32 
EPOCHS = 20

def load_mnist():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.astype('float32') / 255
  x_train = np.expand_dims(x_train, axis=3)

  x_test = x_test.astype('float32') / 255
  x_test = np.expand_dims(x_test, axis=3)

  return x_train, y_train, x_test, y_test

def train(x_train, learning_rate, batch_size, epochs):
  autoencoder = Autoencoder(input_shape=(28,28,1), conv_filters=(32, 64, 64, 64), conv_kernels=(3, 3, 3, 3), conv_strides=(1, 2, 2, 1), latent_space_dim=2)
  autoencoder.summary()
  autoencoder.compile(learning_rate)
  autoencoder.train(x_train, batch_size, epochs)
  return autoencoder
  

if __name__ == "__main__":
  x_train, _, _, _ = load_mnist()
  autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
  autoencoder.save("model")