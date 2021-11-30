import numpy as np


class Maxpool:
    """
    Used to reduce the spacial dimension of the input for the next convolution layer
    """

    def __init__(self, kernel_size, padding, stride):
        self.kernel_size = kernel_size  # dimensions of kernel : kernel_size * kernel_size
        self.padding = padding
        self.stride = stride
        self.height_input = None  # input height for forward pass
        self.width_input = None  # input width for forward pass
        self.depth_input = None  # input depth for forward pass

    def forward(self, input_vector):
        """
        Forward propagation of maxpool
        :param input_vector: The vector who's size has to be modified
        :return: vector with reduced dimensions
        """
        self.height_input = input_vector.shape[0]  # height of the input
        self.width_input = input_vector.shape[1]  # width of the input
        self.depth_input = input_vector.shape[2]  # depth of the input

        # expected height and width of the output
        height_output = int(((self.height_input + 2 * self.padding - (self.kernel_size - 2)) / self.stride))
        width_output = int(((self.width_input + 2 * self.padding - (self.kernel_size - 2)) / self.stride))
        # if padding > 0, input matrix will have higher dimensions, hence calculating final dimensions of
        # input considering padding
        input_with_padding = np.zeros((self.height_input + 2 * self.padding,
                                       self.width_input + 2 * self.padding, self.depth_input))

        # copying actual input matrix X to padded input matrix
        for d in range(0, self.depth_input):
            for i in range(self.padding, input_with_padding.shape[0] - self.padding):
                for j in range(self.padding, input_with_padding.shape[1] - self.padding):
                    input_with_padding[i][j][d] = input_vector[i - self.padding][j - self.padding][d]

        output = []  # initializing output matrix

        # performing maxpool operation
        # Maxpool for kernel size 3*3 is finding the max value in input matrix after
        # overlapping it with kernel across height and depth
        for depth in range(0, self.depth_input):
            for i in range(0, input_with_padding.shape[0] - self.kernel_size, self.stride):
                for j in range(0, input_with_padding.shape[1] - self.kernel_size, self.stride):
                    mx = 100000
                    for a in range(i, i + self.kernel_size):
                        for b in range(j, j + self.kernel_size):
                            if mx < input_with_padding[a][b][depth]:
                                mx = input_with_padding[a][b][depth]
                    output.append(int(mx))

        # reshaping output to output dimensional requirements
        output = np.reshape(output, (height_output, width_output, self.depth_input))
        return output

    def backward(self, output_gradient):
        """
        Backward propagation of maxpool
        :param output_gradient: The output gradient vector
        :return: The input gradient vector
        """
        input_gradient = np.zeros((self.height_input * self.width_input * self.depth_input))
        """
                3 4 5
                6 7 8   -->   3 4 5 6 7 8 9 1 0   
                9 1 0
        """
        output_gradient = output_gradient.flatten()

        # since output gradient matrix has half the size as input,
        # incrementing input_gradient index by 2 and filling the values
        for i in range(len(output_gradient)):
            input_gradient[2 * i] = output_gradient[i]
        input_gradient = np.reshape(input_gradient, (self.height_input, self.width_input, self.depth_input))

        return input_gradient
