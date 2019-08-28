import tensorflow as tf
import os

generated_dir = "generated"
n_classes = 10

from classifier.net import make_discriminator_model
from model import Model
import _globals

class GanMetrics:
    aux_loss = 'discr_aux_loss'


bin_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


class Discriminator(Model):
    check_dir = os.path.join(generated_dir, "checkpoints", "classifier")

    def __init__(self):
        self.net = make_discriminator_model(n_classes)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.aux_loss = tf.metrics.Mean()

        super().__init__()

    def loss(self, class_pred, labels):

        class_loss = cross_entropy(labels, class_pred)
        self.aux_loss.update_state(class_loss)
        return class_loss

    def log_metrics(self):

        tf.summary.scalar(GanMetrics.aux_loss, self.aux_loss.result(), _globals.step)
        self.aux_loss.reset_states()




