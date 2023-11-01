import numpy as np

class Network(object):
    """
    Network class for the MLP / Sigmoid Neural Network
        sizes: Neurons per layer

    """
    def __init__(self, sizes):
        self.nLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]