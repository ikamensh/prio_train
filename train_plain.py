import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

from data import train_images, train_labels
from net import make_discriminator_model


def calc_n_images(epochs):
    return ( epochs + 0.5 ) * len(train_images)

d = make_discriminator_model(num_classes=10)
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
d.compile(optimizer=Adam(), loss=cross_entropy)

print("Started training")

history = d.fit(train_images, train_labels, epochs=20)

from ilya_ezplot import Metric, ez_plot
m_train_loss = Metric("images trained", "loss_training_set")

for k, v in enumerate(history.history['loss']):
    m_train_loss.add_record(calc_n_images(k), v)

ez_plot(m_train_loss, 'plots', name="usual")







