import numpy as np


class Flatten:
    """
    this class practically does the same thing as .flatten() but we have more
    dimensional control over this
    """

    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        # convert 2d input to shape(1, h*w*d)
        input = np.reshape(input, (1, self.input_shape[0] * self.input_shape[1] * self.input_shape[2]))
        return input

    def backward(self, output_gradient):
        """
        reshape back to forward pass' input dimensions
        :param output_gradient:
        :return:
        """
        input_gradient = np.reshape(output_gradient, self.input_shape)
        return input_gradient
