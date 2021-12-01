RESIZED_IMAGE_WIDTH = 112
RESIZED_IMAGE_HEIGHT = 112

# model
LEARNING_RATE = 0.01

# CNN CONFIGURATION
MAX_POOL_K_SIZE = 3
MAX_POOL_STRIDE = 2
MAX_POOL_PADDING = 1

CONVOLUTION_K_SIZE = 1
CONVOLUTION_STRIDE = 1
CONVOLUTION_PADDING = 0

IN_CHANNEL = [3, 64, 128, 254, 512]
OUT_CHANNEL = [64, 128, 254, 512, 512]

# RNN CONFIGURATION
INPUT_DIMENSION_RNN = 100
HIDDEN_DIMENSION_RNN = 25088  # CNN output size

# model image path
WORD_VECTORS_FILE = "word_vectors"
CNN_PARAMETERS_FILE = "cnn_parameters"
PROCESSED_DATA_FILE = "data.json"

#
IMAGE_LIMIT = 1500