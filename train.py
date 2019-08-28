from data import cifar_sampler, train_images, train_labels
import tensorflow as tf
from classifier.discriminator import Discriminator
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

import time


d = Discriminator()
d.compile()

@tf.function
def step():
    x, y = cifar_sampler.sample()

    with tf.GradientTape() as tape:
        predictions = d.net(x)
        loss = d.loss(predictions, y)

    d.update(tape, loss)


print("Started training")
for epoch in range(10):
    print("Epoch", epoch)

    t = time.time()
    for i in range(1000):
        step()

    d.log_metrics()

    print(f"Training done in {time.time() - t:.3f}")

    t = time.time()

    y_pred = d.net.predict(train_images)
    losses = cross_entropy.call(train_labels, y_pred)
    cifar_sampler.update_chances(losses)

    print(f"Chances changed in {time.time() - t:.3f}")







