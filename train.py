import time
import os

import numpy as np

from data import cifar_sampler, train_images, train_labels, test_labels, test_images
from net import make_discriminator_model, cross_entropy
from util.plotting import plot_statistic


from data import BATCH_SIZE
batches_per_epoch = len(train_labels) // BATCH_SIZE


from ilya_ezplot import Metric, plot_group


def calc_n_images(epochs, batches):
    return ( epochs * batches_per_epoch + batches ) * BATCH_SIZE

def train_model(d, epochs: int):

    m_batch_loss = Metric("images trained", "loss_batches")
    m_train_loss = Metric("images trained", "loss_training_set")
    m_test_loss = Metric("images trained", "loss_test_set")

    print("Started training")
    for epoch in range(epochs):
        print("Epoch", epoch)

        t = time.time()
        for i in range(batches_per_epoch):
            x, y = cifar_sampler.sample()
            loss = d.train_on_batch(x, y)
            m_batch_loss.add_record(calc_n_images(epoch, i), loss)


        print(f"Epoch {epoch} done in {time.time() - t:.3f}")

        y_pred = d.predict(train_images)
        losses = cross_entropy.call(train_labels, y_pred)
        plot_statistic(losses, f"losses_{epoch}", folder=os.path.join("plots","prioritized", "losses"))
        m_train_loss.add_record(calc_n_images(epoch+1, 0), np.mean(losses))
        cifar_sampler.update_chances(losses)
        plot_statistic(cifar_sampler._chances, f"chances_{epoch}", folder=os.path.join("plots", "prioritized", "chances"))

        test_pred = d.predict(test_images)
        test_losses = cross_entropy.call(test_labels, test_pred)
        plot_statistic(test_losses, f"test_losses_{epoch}", folder=os.path.join("plots","prioritized", "test_losses"))
        m_test_loss.add_record(calc_n_images(epoch+1, 0), np.mean(test_losses))

        print(np.mean(losses), np.mean(test_losses))
        print(f"Chances changed in {time.time() - t:.3f}")


    metrics = {"batch_loss": m_batch_loss,
               "training_set_loss": m_train_loss,
               "test_set_loss": m_test_loss}
    plot_group(metrics, 'plots', name="high_loss_first")

if __name__ == "__main__":
    model = make_discriminator_model(num_classes=10)
    train_model(model, 100)







