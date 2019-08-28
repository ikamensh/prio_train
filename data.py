BUFFER_SIZE = 60000
BATCH_SIZE = 64

import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

test_images = test_images.reshape(test_images.shape[0], 32, 32, 3).astype('float32')
test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

from sample import Sampler

cifar_sampler = Sampler(train_images, train_labels, 64)
