from train import train_model
from net import mock_net

def test_no_exception(tmpdir):
    net = mock_net(10)
    train_model(net, 2)