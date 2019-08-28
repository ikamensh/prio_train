import numpy as np


class Sampler:
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        min_chances: float = 0.1,
        max_chances: float = 10,
    ):
        """

        :param images: 
        :param labels:
        :param batch_size:
        :param min_chances: how small the chance of encountering a sample can be, compared to average
        :param max_chances: how big the chance of encountering a sample can be, compared to average
        """
        assert (
            len(images.shape) == 4
        ), "images are expected as 4d tensor of shape [samples, w, h, ch]"
        assert (
            len(labels.shape) == 2
        ), "labels are expected as 2d tensor of shape [samples, 1]"

        self.size = images.shape[0]
        assert labels.shape[0] == self.size, "amount of images and labels must match"
        assert (
            self.size > batch_size
        ), "batch size must be less than the number of data samples"

        self.data = np.array(list(zip(images, labels)))
        self.batch_size = batch_size
        

        assert max_chances > 1
        self.max_chances = max_chances

        self._indexes = np.arange(0, self.size, dtype=np.int)
        self._chances = np.ones_like(self._indexes) / self.size

        assert 0 <= min_chances < 1
        self._min_chances = min_chances * np.ones_like(self._indexes) / self.size

    def update_chances(self, losses: np.ndarray):
        assert losses.shape in ((self.size,), (self.size, 1))
        losses = np.squeeze(losses)
        proportional = losses / np.median(losses)
        proportional = np.min([proportional, np.ones_like(proportional) * self.max_chances], axis=0)
        proportional /= np.sum(proportional)

        self._chances =  proportional + self._min_chances
        self._chances /= np.sum(self._chances)

    def sample(self):
        selected_idx = np.random.choice(self._indexes, self.batch_size, p=self._chances)
        data = self.data[selected_idx]
        imgs, labels = list(zip(*data))
        imgs = np.vstack([np.expand_dims(i, axis=0) for i in imgs])
        labels = np.vstack([np.expand_dims(i, axis=0) for i in labels])
        return imgs, labels
