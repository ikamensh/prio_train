from data import cifar_sampler, train_images, train_labels
import tensorflow as tf
from net import make_discriminator_model
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

import time


d = make_discriminator_model(num_classes=10)
d.compile(optimizer='adam', loss=cross_entropy)


from data import BATCH_SIZE
batches_per_epoch = 1000


from ilya_ezplot import Metric, ez_plot

m = Metric("images trained", "loss")

def calc_n_images(epochs, batches):
    return ( epochs * batches_per_epoch + batches ) * BATCH_SIZE

print("Started training")
for epoch in range(10):
    print("Epoch", epoch)

    t = time.time()
    for i in range(batches_per_epoch):
        x, y = cifar_sampler.sample()
        loss = d.train_on_batch(x, y)
        m.add_record(calc_n_images(epoch, i), loss)


    print(f"Training done in {time.time() - t:.3f}")

    t = time.time()

    # y_pred = d.net.predict(train_images)
    # losses = cross_entropy.call(train_labels, y_pred)
    # cifar_sampler.update_chances(losses)
    #
    # print(f"Chances changed in {time.time() - t:.3f}")

ez_plot(m, 'plots')







