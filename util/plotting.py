from matplotlib import pyplot as plt


import numpy as np

import os

def plot_statistic(array: np.ndarray, identifier: str, folder = "plots"):
    original = np.squeeze(array)
    assert len(original.shape) == 1
    srtd = np.sort(original)

    plt.clf()
    plt.plot(srtd)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"{identifier}.png"))

if __name__ == "__main__":
    o = np.random.uniform(0, 100, size=[2000, ])

    plot_statistic(o, "uniform")