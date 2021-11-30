import numpy as np


class Flatten:
    """
    this class practically does the same thing as .flatten() but we have more
    dimensional control over this
    """
    def __init__(self):
        self.input_shape = None

    def forward(self, image_vector):
        """
        convert 2d input to shape(1, h*w*d)
        :param image_vector: the image vector
        :return: flattened the array
        """
        self.input_shape = image_vector.shape
        return np.reshape(image_vector, (1, self.input_shape[0] * self.input_shape[1] * self.input_shape[2]))

    def backward(self, output_gradient):
        """
        reshape back to forward pass' input dimensions
        :param output_gradient:
        :return: the image vector after reshaping it
        """
        return np.reshape(output_gradient, self.input_shape)
