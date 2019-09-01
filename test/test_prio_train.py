from train import train_model, train_epoch_prio, train_epoch_plain
from net import mock_net

def test_no_exception_prio(tmpdir):
    train_model(2, "test", train_epoch_prio, lambda :mock_net(10))


def test_no_exception_standard(tmpdir):
    train_model(2, "test", train_epoch_plain, lambda :mock_net(10))