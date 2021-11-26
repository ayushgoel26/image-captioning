import numpy as np
from scipy import signal


class Convolution:

    def __init__(self, in_channels, out_channels, kernel_size):
        self.depth = out_channels

        '''
        kernel has 4 dimensions : output depth (out_channels) * input_depth (in_channels) * kernel_size * kernel_size
        '''
        self.kernels_shape = (self.depth, in_channels, kernel_size, kernel_size)

        '''initializing random values to kernel'''
        self.kernels = np.random.randn(*self.kernels_shape)
        self.input = None
        self.input_depth = None
        self.input_shape = None
        self.output_shape = None
        self.biases = None

    def forward(self, input_vector):
        self.input = input_vector

        self.input_depth = input_vector.shape[0]
        input_height = input_vector.shape[1]
        input_width = input_vector.shape[2]
        self.input_shape = (input_vector.shape[0], input_vector.shape[1], input_vector.shape[2])
        self.output_shape = (self.depth, input_height - self.kernels_shape[2] + 1,
                             input_width - self.kernels_shape[3] + 1)
        self.biases = np.random.randn(*self.output_shape)

        output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

        return output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
