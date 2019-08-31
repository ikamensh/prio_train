import time
import os

import numpy as np
import tensorflow as tf

from data import cifar_sampler, train_images, train_labels, test_labels, test_images
from net import make_discriminator_model, cross_entropy, make_sota_model
from util.plotting import plot_statistic


from data import BATCH_SIZE

batches_per_epoch = len(train_labels) // BATCH_SIZE


from ilya_ezplot import Metric


def calc_n_images(epochs, batches):
    return (epochs * batches_per_epoch + batches) * BATCH_SIZE


def train_epoch_prio(net, current_epoch, metric_batch_loss, m_batch_acc):
    for i in range(batches_per_epoch):
        x, y = cifar_sampler.sample()
        loss, acc = net.train_on_batch(x, y)
        metric_batch_loss.add_record(calc_n_images(current_epoch, i), loss)
        m_batch_acc.add_record(calc_n_images(current_epoch, i) ,acc)

def train_epoch_plain(net, *args):
    net.fit(train_images, train_labels, epochs=1, batch_size=BATCH_SIZE, verbose=1)

class Tags:
    Prio = "Prio"
    Standard = "Standard"

def train_model(epochs: int, tag: str, train_epoch_callable):

    d = make_sota_model()

    m_batch_loss = Metric(
        name=f"batch_loss_{tag}", x_label="images trained", y_label="loss_batches"
    )

    m_batch_acc = Metric(
        name=f"batch_acc_{tag}", x_label="images trained", y_label="acc_batches"
    )

    m_train_loss = Metric(
        name=f"train_loss_{tag}", x_label="images trained", y_label="loss_training_set"
    )
    m_test_loss = Metric(
        name=f"test_loss_{tag}", x_label="images trained", y_label="loss_test_set"
    )

    m_train_acc = Metric(
        name=f"train_acc_{tag}", x_label="images trained", y_label="acc_training_set"
    )
    m_test_acc = Metric(
        name=f"test_acc_{tag}", x_label="images trained", y_label="acc_test_set"
    )

    print("Started training")
    for epoch in range(epochs):
        print("Epoch", epoch)

        t = time.time()

        train_epoch_callable(d, epoch, m_batch_loss, m_batch_acc)

        loss, tr_acc = d.evaluate(train_images, train_labels, verbose=0)
        m_train_acc.add_record(calc_n_images(epoch + 1, 0), tr_acc)

        loss, tst_acc = d.evaluate(test_images, test_labels, verbose=0)
        m_test_acc.add_record(calc_n_images(epoch + 1, 0), tst_acc)

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

        print(np.mean(losses), np.mean(test_losses), tr_acc, tst_acc)
        print(f"Chances changed in {time.time() - t:.3f}")

    if tag == Tags.Prio:
        m_batch_loss.save()
    m_train_loss.save()
    m_test_loss.save()
    m_train_acc.save()
    m_test_acc.save()


if __name__ == "__main__":
    train_model(epochs=100, tag=Tags.Standard, train_epoch_callable=train_epoch_plain)
    train_model(epochs=100, tag=Tags.Prio, train_epoch_callable=train_epoch_prio)

