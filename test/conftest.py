import pytest
import numpy as np
from sample import Sampler

@pytest.fixture()
def fake_cifar():

    ds_size = 100
    n_classes = 10

    labels = np.repeat(np.arange(0, n_classes, dtype=np.int), ds_size // n_classes)
    print(labels.shape, set(labels))
    r = np.expand_dims(np.sin(labels), axis=0)
    g = np.expand_dims(np.cos(labels), axis=0)
    b = np.expand_dims(2 * (labels / n_classes) - 1, axis=0)

    rgb = np.vstack([r, g, b])
    rgb = rgb.swapaxes(0, 1)
    rgb = rgb.reshape([rgb.shape[0], 1, 1, -1])
    data = np.tile(rgb, [1, 32, 32, 1])

    yield data, labels


@pytest.fixture()
def fake_sampler(fake_cifar):
    images, labels = fake_cifar
    yield Sampler(images, labels, batch_size=16)