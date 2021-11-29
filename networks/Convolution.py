class Convolution:


    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, input_shape):
      
        self.depth = out_channels
        '''
        kernel has 4 dimensions : output depth (out_channels) * input_depth (in_channels) * kernel_size * kernel_size
        '''
        '''initializing random values to kernel'''
        self.kernels_shape = (self.depth, in_channels, kernel_size, kernel_size)
        self.input_shape = input_shape
        self.input_depth = input_shape[0]
        input_height = input_shape[1]
        input_width = input_shape[2]
        self.kernels = np.random.randn(*self.kernels_shape)
        self.output_shape = (self.depth, input_height - self.kernels_shape[2] + 1, input_width - self.kernels_shape[3] + 1)
        self.biases = np.random.randn(*self.output_shape)
        self.output = np.copy(self.biases)



    def forward(self, input):
        self.input = input
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

        return self.output


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