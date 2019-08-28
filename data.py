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

import numpy as np

ds_size = 50000
n_classes = 10
sample_per_class = ds_size // n_classes
ez_imgs = np.zeros([ds_size, 32, 32, 3], dtype=np.float)
ez_labels = np.zeros([ds_size], dtype=np.int)
for i in range(n_classes):
    ez_imgs[i * sample_per_class : (i+1) * sample_per_class] = \
        np.ones_like(ez_imgs[i * sample_per_class : (i+1) * sample_per_class]) * ((i - 5) / 10)
    ez_labels[i * sample_per_class: (i + 1) * sample_per_class] = \
        np.ones_like(ez_labels[i * sample_per_class: (i + 1) * sample_per_class]) * i

