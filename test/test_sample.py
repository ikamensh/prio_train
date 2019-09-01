from sample import Sampler
import numpy as np
import pytest


@pytest.fixture()
def data_100():
    ds_size = 100
    imgs = np.zeros([ds_size, 2, 2, 3])
    for i in range(ds_size):
        imgs[i] = np.ones_like(imgs[i]) * i

    labels = np.expand_dims(np.arange(0, ds_size), axis=1)
    yield imgs, labels


@pytest.mark.parametrize("batch_size", [1, 4])
def test_sampler_data_format(data_100, batch_size):
    imgs, labels = data_100

    s = Sampler(imgs, labels, batch_size)

    images1, labels1, w = s.sample()
    assert isinstance(images1, np.ndarray)
    assert len(images1.shape) == 4
    assert images1.shape[0] == batch_size

    assert isinstance(labels1, np.ndarray)
    assert len(labels1.shape) == 2
    assert labels1.shape[0] == batch_size


def test_sampler_images_match_labels(data_100):
    imgs, labels = data_100
    batch_size = 4
    s = Sampler(imgs, labels, batch_size)

    images1, labels1, w = s.sample()
    derived_labels = np.mean(images1, axis=(1, 2, 3))

    assert np.all(derived_labels == labels1[:, 0])


def test_sampler_different_samples(data_100):
    imgs, labels = data_100

    batch_size = 50

    s = Sampler(imgs, labels, batch_size)

    images1, labels1, w = s.sample()
    images2, labels2, w = s.sample()

    assert not np.all(labels1 == labels2)


def test_chances_deterministic(data_100):
    imgs, labels = data_100

    batch_size = 50

    s = Sampler(imgs, labels, batch_size)
    all_idx = 83
    s._chances = np.zeros_like(s._chances)
    s._chances[all_idx] = 1

    images1, labels1, w = s.sample()

    assert np.all(labels1 == np.ones_like(labels1) * all_idx)


def test_chances_range(data_100):
    imgs, labels = data_100

    batch_size = 50

    s = Sampler(imgs, labels, batch_size)
    s._chances = np.zeros_like(s._chances)
    s._chances[50:] = 1 / 50

    for i in range(10):
        images1, labels1, w = s.sample()
        assert np.mean(labels) < np.mean(labels1)


def test_update_chances(data_100):
    imgs, labels = data_100

    batch_size = 50

    s = Sampler(imgs, labels, batch_size)
    s.update_chances(np.ones_like(labels))

    images1, labels1, w = s.sample()
    assert isinstance(images1, np.ndarray)
    assert len(images1.shape) == 4
    assert images1.shape[0] == batch_size

    assert isinstance(labels1, np.ndarray)
    assert len(labels1.shape) == 2
    assert labels1.shape[0] == batch_size


def test_losses_range(data_100):
    imgs, labels = data_100

    batch_size = 50

    s = Sampler(imgs, labels, batch_size, min_chances=0.)
    losses = np.zeros_like(s._chances)
    losses[50:] = 1

    s.update_chances(losses)

    for i in range(10):
        images1, labels1, w = s.sample()
        assert np.mean(labels) < np.mean(labels1)


def test_max_prob(data_100):
    imgs, labels = data_100

    batch_size = 50

    s = Sampler(imgs, labels, batch_size, min_chances=0., max_chances=1e100)
    losses = np.zeros_like(s._chances)
    losses[50:] = 1
    losses[83] = 1e100

    s.update_chances(losses)

    for i in range(10):
        images1, labels1, w = s.sample()
        assert np.mean(labels1) == 83


def test_cap_prob(data_100):
    imgs, labels = data_100

    batch_size = 50

    s = Sampler(imgs, labels, batch_size, min_chances=0., max_chances=3)
    losses = np.zeros_like(s._chances)
    losses[50:] = 1
    losses[83] = 1e100

    s.update_chances(losses)

    for i in range(10):
        images1, labels1, w = s.sample()
        assert np.mean(labels1) < 83


def test_exclude_outliers(data_100):
    imgs, labels = data_100

    batch_size = 50

    s = Sampler(imgs, labels, batch_size, min_chances=0., max_chances=3)
    losses = np.ones_like(s._chances)
    losses[-1] = 10

    s.update_chances(losses, exclude_outliers=True)

    assert s._chances[-1] < np.mean(s._chances)

def test_cap_min_prob(data_100):
    imgs, labels = data_100

    batch_size = 50

    s = Sampler(imgs, labels, batch_size, min_chances=0.5, max_chances=1e100)
    losses = np.zeros_like(s._chances)
    losses[50:] = 1
    losses[83] = 1e100

    s.update_chances(losses)

    for i in range(10):
        images1, labels1, w = s.sample()
        assert np.mean(labels1) < 83
