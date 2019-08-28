import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

from data import train_images, train_labels
from net import make_discriminator_model


def calc_n_images(epochs):
    return ( epochs + 0.5 ) * len(train_images)

d = make_discriminator_model(num_classes=10)

print("Started training")

history = d.fit(train_images, train_labels, batch_size=64, epochs=50, callbacks=[TensorBoard()])

from ilya_ezplot import Metric, ez_plot
m_train_loss = Metric("images trained", "loss_training_set")

for k, v in enumerate(history.history['loss']):
    m_train_loss.add_record(calc_n_images(k), v)

ez_plot(m_train_loss, 'plots', name="usual")







