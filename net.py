from tensorflow.python.keras import layers
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def make_discriminator_model(num_classes, color_ch = 3):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Conv2D(128, 4, padding='same'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2D(256, 5, strides=2, padding='same'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())
    cnn.add(layers.Dropout(0.2))

    cnn.add(layers.Conv2D(128, 4, padding='same'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2D(256, 5, strides=2, padding='same'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())
    cnn.add(layers.Dropout(0.2))

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(256))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Dense(128))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    image = layers.Input(shape=(32, 32, color_ch))
    features = cnn(image)

    classification = layers.Dense(num_classes, activation='softmax', name='auxiliary')(features)

    discriminator = tf.keras.Model(image, classification)

    assert discriminator.input_shape == (None, 32, 32, 3)

    discriminator.compile(optimizer=Adam(lr=3e-2), loss=cross_entropy)

    return discriminator


def mock_net(num_classes):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(num_classes, activation='softmax'))
    cnn.compile(optimizer=Adam(), loss=cross_entropy)
    return cnn
