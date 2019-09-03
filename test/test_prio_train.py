from train import train_model, train_epoch_prio, train_epoch_plain
from net import mock_net, make_sota_model

def test_no_exception_prio(tmpdir, fake_sampler):
    train_callable = lambda net: train_epoch_prio(net, fake_sampler)
    train_model(2, "test", train_callable, lambda :mock_net(10))

def test_no_exception_standard(tmpdir, fake_cifar):
    x, y = fake_cifar
    train_callable = lambda net: train_epoch_plain(net, x, y)
    train_model(2, "test", train_callable, lambda :mock_net(10))

# def test_no_exception_prio_sota(tmpdir, fake_sampler):
#     train_callable = lambda net: train_epoch_prio(net, fake_sampler)
#     train_model(2, "test", train_callable, lambda :make_sota_model())


import numpy as np

def test_one_train_call(fake_sampler):
    net = make_sota_model()

    fake_sampler.update_chances(losses=np.ones_like(fake_sampler.min_chances_array))
    x, y, w = fake_sampler.sample()
    loss, acc = net.train_on_batch(x, y, sample_weight=w)


def test_two_train_calls(fake_sampler):
    net = make_sota_model()


    x, y, w = fake_sampler.sample()
    loss, acc = net.train_on_batch(x, y, sample_weight=w)

    fake_sampler.update_chances(losses=np.ones_like(fake_sampler.min_chances_array))
    x, y, w = fake_sampler.sample()
    loss, acc = net.train_on_batch(x, y, sample_weight=w)


