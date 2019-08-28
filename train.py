import time

import tensorflow as tf
import numpy as np

from data import cifar_sampler, train_images, train_labels
from net import make_discriminator_model

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
d = make_discriminator_model(num_classes=10)

from data import BATCH_SIZE
batches_per_epoch = 300


from ilya_ezplot import Metric, ez_plot, plot_group

m_batch_loss = Metric("images trained", "loss_batches")
m_train_loss = Metric("images trained", "loss_training_set")


def calc_n_images(epochs, batches):
    return ( epochs * batches_per_epoch + batches ) * BATCH_SIZE

print("Started training")
for epoch in range(150):
    print("Epoch", epoch)

    t = time.time()
    for i in range(batches_per_epoch):
        x, y = cifar_sampler.sample()
        loss = d.train_on_batch(x, y)
        m_batch_loss.add_record(calc_n_images(epoch, i), loss)


    print(f"Training done in {time.time() - t:.3f}")

    t = time.time()

    y_pred = d.predict(train_images)
    losses = cross_entropy.call(train_labels, y_pred)
    m_train_loss.add_record(calc_n_images(epoch+1, 0), np.mean(losses))
    cifar_sampler.update_chances(losses)

    print(f"Chances changed in {time.time() - t:.3f}")

plot_group({"batch_loss": m_batch_loss, "training_set_loss": m_train_loss}, 'plots', name="high_loss_first")







