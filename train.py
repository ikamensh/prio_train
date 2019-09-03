from __future__ import annotations
import time
import os
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from ilya_ezplot import Metric

from data import cifar_sampler, train_images, train_labels, test_labels, test_images
from net import cross_entropy, make_sota_model
from util.plotting import plot_statistic
import _globals

from data import BATCH_SIZE

if TYPE_CHECKING:
    from sample import Sampler


def calc_n_images(batches):
    return _globals.epoch * len(train_images) + batches * BATCH_SIZE


def train_epoch_prio(net, sampler: Sampler):
    loss_chronicle = {}
    acc_chronicle = {}

    for i in range(sampler.batches_per_epoch):
        x, y, w = sampler.sample()
        loss, acc = net.train_on_batch(x, y, sample_weight=w)
        loss_chronicle[calc_n_images(i)] = loss
        acc_chronicle[calc_n_images(i)] = acc

    return loss_chronicle, acc_chronicle

def train_epoch_plain(net: tf.keras.Sequential, x, y):
    net.fit(x, y, epochs=1, batch_size=BATCH_SIZE, verbose=1)

class Tags:
    Prio = "Prio"
    Standard = "Standard"

def train_model(epochs: int, tag: str, train_epoch_callable, model_callable):

    d = model_callable()

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
        _globals.epoch = epoch
        print("Epoch", epoch)

        t = time.time()

        chronicles = train_epoch_callable(d)
        if chronicles:
            loss_chronicle, acc_chronicle = chronicles
            m_batch_loss.add_dict(loss_chronicle)
            m_batch_acc.add_dict(acc_chronicle)

        _globals.epoch += 1

        loss, tr_acc = d.evaluate(train_images, train_labels, verbose=0)
        m_train_acc.add_record(calc_n_images(0), tr_acc)

        loss, tst_acc = d.evaluate(test_images, test_labels, verbose=0)
        m_test_acc.add_record(calc_n_images( 0), tst_acc)

        print(f"Epoch {epoch} done in {time.time() - t:.3f}")

        y_pred = tf.convert_to_tensor(d.predict(train_images))
        losses = cross_entropy.call(train_labels, y_pred)
        plot_statistic(
            losses,
            f"losses_{epoch}",
            folder=os.path.join("plots", tag, "losses"),
        )
        m_train_loss.add_record(calc_n_images( 0), np.mean(losses))

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
        m_test_loss.add_record(calc_n_images( 0), np.mean(test_losses))

        print(np.mean(losses), np.mean(test_losses), tr_acc, tst_acc)
        print(f"Chances changed in {time.time() - t:.3f}")

    if tag == Tags.Prio:
        m_batch_loss.save()
        m_batch_acc.save()

    m_train_loss.save()
    m_test_loss.save()
    m_train_acc.save()
    m_test_acc.save()


if __name__ == "__main__":
    for i in range(5):
        prio_train_callable = lambda net: train_epoch_plain(net, x=train_images, y=train_labels)
        train_model(epochs=30, tag=f"{Tags.Standard}${i}", train_epoch_callable=prio_train_callable, model_callable=make_sota_model)

        prio_train_callable = lambda net: train_epoch_prio(net, sampler=cifar_sampler)
        train_model(epochs=30, tag=f"{Tags.Prio}${i}", train_epoch_callable=prio_train_callable, model_callable=make_sota_model)

