import time
import os

import numpy as np
import tensorflow as tf

from data import cifar_sampler, train_images, train_labels, test_labels, test_images
from net import make_discriminator_model, cross_entropy
from util.plotting import plot_statistic


from data import BATCH_SIZE

batches_per_epoch = len(train_labels) // BATCH_SIZE


from ilya_ezplot import Metric


def calc_n_images(epochs, batches):
    return (epochs * batches_per_epoch + batches) * BATCH_SIZE


def train_epoch_prio(net, current_epoch, metric_batch_loss):
    for i in range(batches_per_epoch):
        x, y = cifar_sampler.sample()
        loss = net.train_on_batch(x, y)
        metric_batch_loss.add_record(calc_n_images(current_epoch, i), loss)


def train_epoch_plain(net, current_epoch, metric_batch_loss):
    net.fit(train_images, train_labels, epochs=1, batch_size=BATCH_SIZE)

class Tags:
    Prio = "Prio"
    Standard = "Standard"

def train_model(d, epochs: int, tag: str, train_epoch_callable):

    m_batch_loss = Metric(
        name=f"batch_loss_{tag}", x_label="images trained", y_label="loss_batches"
    )
    m_train_loss = Metric(
        name=f"train_loss_{tag}", x_label="images trained", y_label="loss_training_set"
    )
    m_test_loss = Metric(
        name=f"test_loss_{tag}", x_label="images trained", y_label="loss_test_set"
    )

    print("Started training")
    for epoch in range(epochs):
        print("Epoch", epoch)

        t = time.time()

        train_epoch_callable(d, epoch, m_batch_loss)

        print(f"Epoch {epoch} done in {time.time() - t:.3f}")

        y_pred = tf.convert_to_tensor(d.predict(train_images))
        losses = cross_entropy.call(train_labels, y_pred)
        plot_statistic(
            losses,
            f"losses_{epoch}",
            folder=os.path.join("plots", tag, "losses"),
        )
        m_train_loss.add_record(calc_n_images(epoch + 1, 0), np.mean(losses))

        if tag == Tags.Prio:
            cifar_sampler.update_chances(losses)
            plot_statistic(
                cifar_sampler._chances,
                f"chances_{epoch}",
                folder=os.path.join("plots", tag, "chances"),
            )

        test_pred = tf.convert_to_tensor(d.predict(test_images))
        test_losses = cross_entropy.call(test_labels, test_pred)
        plot_statistic(
            test_losses,
            f"test_losses_{epoch}",
            folder=os.path.join("plots", tag, "test_losses"),
        )
        m_test_loss.add_record(calc_n_images(epoch + 1, 0), np.mean(test_losses))

        print(np.mean(losses), np.mean(test_losses))
        print(f"Chances changed in {time.time() - t:.3f}")

    if tag == Tags.Prio:
        m_batch_loss.save()
    m_train_loss.save()
    m_test_loss.save()


if __name__ == "__main__":
    model = make_discriminator_model(num_classes=10)
    train_model(model, 3, tag=Tags.Prio, train_epoch_callable=train_epoch_prio)
    train_model(model, 3, tag=Tags.Standard, train_epoch_callable=train_epoch_plain)

