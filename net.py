import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import sparse_categorical_accuracy

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

from tensorflow.python.keras.regularizers import l1_l2, l2

def make_discriminator_model(num_classes, color_ch = 3):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Conv2D(128, 4, padding='same', activity_regularizer=l2(1e-4), input_shape=(32, 32, color_ch)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())
    cnn.add(layers.Dropout(0.15))

    cnn.add(layers.Conv2D(256, 5, strides=2, padding='same', activity_regularizer=l2(1e-4)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())
    cnn.add(layers.Dropout(0.15))

    cnn.add(layers.Conv2D(128, 4, padding='same', activity_regularizer=l2(1e-4)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())
    cnn.add(layers.Dropout(0.15))

    cnn.add(layers.Conv2D(256, 5, strides=2, padding='same', activity_regularizer=l2(1e-4)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())
    cnn.add(layers.Dropout(0.15))

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(256, activity_regularizer=l2(1e-4)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())
    cnn.add(layers.Dropout(0.4))

    cnn.add(layers.Dense(128, activity_regularizer=l2(1e-4)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())
    cnn.add(layers.Dropout(0.3))

    cnn.add(layers.Dense(num_classes, activation='softmax', name='auxiliary'))



    cnn.compile(optimizer=Adam(decay=1e-5), loss=cross_entropy, metrics=[sparse_categorical_accuracy])

    return cnn



def make_sota_model():
    from efficient_net.efficient_net import get_efficientnet
    model = get_efficientnet((32, 32), classes=10)
    model.compile(optimizer=Adam(decay=1e-5), loss=cross_entropy, metrics=[sparse_categorical_accuracy])
    return model


def mock_net(num_classes):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(num_classes, activation='softmax'))
    cnn.compile(optimizer=Adam(), loss=cross_entropy)
    return cnn


if __name__ == "__main__":
    from data import train_images, train_labels

    net = make_discriminator_model(10)
    net.summary()
    out = net.evaluate(train_images[:1000], train_labels[:1000])

