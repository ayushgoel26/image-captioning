import numpy as np


class Maxpool:
    """
    Used to reduce the spacial dimension of the input for the next convolution layer
    """
    def __init__(self, kernel_size, padding, stride):
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.height_input = None
        self.width_input = None
        self.depth_input = None

    def forward(self, input_vector):
        """
        Forward propagation of maxpool
        :param input_vector: The vector who's size has to be modified
        :return: vector with reduced dimensions
        """
        self.height_input = input_vector.shape[0]  # height of the input
        self.width_input = input_vector.shape[1]  # width of the input
        self.depth_input = input_vector.shape[2]  # depth of the input

        # expected height of the output
        height_output = int(((self.height_input + 2 * self.padding - (self.kernel_size - 2)) / self.stride))
        width_output = int(((self.width_input + 2 * self.padding - (self.kernel_size - 2)) / self.stride))

        input_with_padding = np.zeros((self.height_input + 2 * self.padding,
                                       self.width_input + 2 * self.padding, self.depth_input))

        for d in range(0, self.depth_input):
            for i in range(self.padding, input_with_padding.shape[0] - self.padding):
                for j in range(self.padding, input_with_padding.shape[1] - self.padding):
                    input_with_padding[i][j][d] = input_vector[i - self.padding][j - self.padding][d]

        output = []

        for depth in range(0, self.depth_input):
            for i in range(0, input_with_padding.shape[0] - self.kernel_size, self.stride):
                for j in range(0, input_with_padding.shape[1] - self.kernel_size, self.stride):
                    mx = 100000
                    for a in range(i, i + self.kernel_size):
                        for b in range(j, j + self.kernel_size):
                            if mx < input_with_padding[a][b][depth]:
                                mx = input_with_padding[a][b][depth]
                    output.append(int(mx))
        output = np.reshape(output, (height_output, width_output, self.depth_input))
        return output

    def backward(self, output_gradient):
        """
        Backward propagation of maxpool
        :param output_gradient: The output gradient vector
        :return: The input gradient vector
        """
        input_gradient = np.zeros((self.height_input * self.width_input * self.depth_input))
        output_gradient = output_gradient.flatten() # have to confirm

        for i in range(len(output_gradient)):
            input_gradient[2 * i] = output_gradient[i]
        input_gradient = np.reshape(input_gradient, (self.height_input, self.width_input, self.depth_input))

        return input_gradient
