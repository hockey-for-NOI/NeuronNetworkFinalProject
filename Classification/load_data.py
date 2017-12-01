import numpy as np
import os


def load_mnist_2d(data_dir, n_split):
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((n_split, -1, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((n_split, -1))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = (trX - 128.0) / 255.0
    teX = (teX - 128.0) / 255.0
    
    for split_id in range(n_split):
        for i in range(trY.shape[1]):
            if np.random.randint(n_split - 1) < split_id:
                trY[split_id][i] ^= 1
    
    return trX, teX, trY, teY
