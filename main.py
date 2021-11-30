from data.processing import Processor
from networks.net import CaptionGenerator
import pickle
import os.path

# dat = processor.data
#
# mxPoolKSize = 3
# mxPoolStride = 2
# mxPoolPadding = 1
#
# processedFeatures = {}
#
# '''Running CNN for test images'''

# input_dimensions = (3, 112, 112)
# for k in dat:
#     print('Creating features for ' + k)
#     if 'images' not in dat[k]: continue;
#     t = dat[k]['images']
#     t = np.asarray(t)
#     t = np.reshape(t, input_dimensions)
#
#     learning_rate = 0.01
#     t1 = convLayer1.forward(t)
#
#     t1 = np.reshape(t1, (t1.shape[1], t1.shape[2], t1.shape[0]))
#     t2 = mxPoolLayer1.forward(t1)
#     t2 = np.reshape(t2, (t2.shape[2], t2.shape[0], t2.shape[1]))
#
#     t3 = convLayer2.forward(t2)
#
#     t3 = np.reshape(t3, (t3.shape[1], t3.shape[2], t3.shape[0]))
#     t4 = mxPoolLayer2.forward(t3)
#     t4 = np.reshape(t4, (t4.shape[2], t4.shape[0], t4.shape[1]))
#
#     t5 = convLayer3.forward(t4)
#
#     t5 = np.reshape(t5, (t5.shape[1], t5.shape[2], t5.shape[0]))
#     t6 = mxPoolLayer3.forward(t5)
#     t6 = np.reshape(t6, (t6.shape[2], t6.shape[0], t6.shape[1]))
#
#     t7 = convLayer4.forward(t6)
#
#     t7 = np.reshape(t7, (t7.shape[1], t7.shape[2], t7.shape[0]))
#     t8 = mxPoolLayer4.forward(t7)
#     t8 = np.reshape(t8, (t8.shape[2], t8.shape[0], t8.shape[1]))
#
#     t9 = convLayer5.forward(t8)
#
#     t_final = flatten.forward(t9)
#     processedFeatures[k] = t_final
#
if not os.path.isfile('data.json'):

    processor = Processor()

    print("Processing Captions")
    processor.caption_reader()
    print("Captions processed")

    print("Pre Processing Images")
    processor.process_images()
    print("Images pre processed")

    print("Writing data into json file")
    with open('data.json', 'wb') as file:
        pickle.dump(processor.data, file)

print("Reading data from json file")
with open('data.json', 'rb') as file:
    data = pickle.load(file)
caption_generator = CaptionGenerator()
caption_generator.train_cnn()
