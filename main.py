from data_processing import Processor
from networksimport TrainData
from networks.Convolution import Convolution
from networks.Convolution import Flatten
from networks.Convolution import Maxpool
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials



"""getting captions and images from drive"""

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)



downloaded = drive.CreateFile({'id':"1ZyaOYHhL6jwzduBwDZgZLyAXlzqOBOnx"})   # get captions
downloaded.GetContentFile('results.csv')  

downloaded = drive.CreateFile({'id':"1vDJYsj8HRcMVK67TbNqma2GddW9otC4m"})   # get images
downloaded.GetContentFile('images.zip')

!unzip images.zip
""""""""""""""""""""""""""""""""""""""""""""


processor = Processor()

print("Processing Captions")
processor.caption_reader()
print("Captions processed")

print("Pre Processing Images")
processor.process_images()
print("Images pre processed")


mxPoolKSize = 3
mxPoolStride = 2
mxPoolPadding = 1

convKSize = 1
convPadding = 0
convStride = 1

in_channel = [3, 64, 128, 254, 512]
out_channel = [64, 128, 254, 512, 512]
input_shapes = [[in_channel[0], 112 , 112], [in_channel[1], 56 , 56] ,[in_channel[2], 28 , 28], [in_channel[3], 14 , 14], [in_channel[4], 7 , 7]]

convLayer1 = Convolution(in_channels = in_channel[0], out_channels = out_channel[0], kernel_size = convKSize ,padding = convPadding , stride=convStride, input_shape = input_shapes[0])
mxPoolLayer1 = Maxpool(mxPoolKSize, stride=mxPoolStride, padding = mxPoolPadding)

convLayer2 = Convolution(in_channels = in_channel[1], out_channels = out_channel[1], kernel_size = convKSize ,padding = convPadding , stride=convStride, input_shape = input_shapes[1])
mxPoolLayer2 = Maxpool(mxPoolKSize, stride=mxPoolStride, padding = mxPoolPadding)

convLayer3 = Convolution(in_channels = in_channel[2], out_channels = out_channel[2], kernel_size = convKSize ,padding = convPadding , stride=convStride, input_shape = input_shapes[2])
mxPoolLayer3 = Maxpool(mxPoolKSize, stride=mxPoolStride, padding = mxPoolPadding)

convLayer4 = Convolution(in_channels = in_channel[3], out_channels = out_channel[3], kernel_size = convKSize ,padding = convPadding , stride=convStride, input_shape = input_shapes[3])
mxPoolLayer4 = Maxpool(mxPoolKSize, stride=mxPoolStride, padding = mxPoolPadding)

convLayer5 = Convolution(in_channels = in_channel[4], out_channels = out_channel[4], kernel_size = convKSize ,padding = convPadding , stride=convStride, input_shape = input_shapes[4])

flatten = Flatten()

learning_rate = 0.01


trainingData = TrainData()
x_train, y_train = trainingData.getTrainData()

'''MNIST training of CNN'''
for dat, y in zip(x_train, y_train): 
  t = dat
  error = 0

  t1 = convLayer1.forward(t)

  t1 = np.reshape(t1,(t1.shape[1], t1.shape[2],t1.shape[0]))
  t2 = mxPoolLayer1.forward(t1)
  t2 = np.reshape(t2,(t2.shape[2], t2.shape[0],t2.shape[1]))



  t3 = convLayer2.forward(t2)

  t3 = np.reshape(t3,(t3.shape[1], t3.shape[2],t3.shape[0]))
  t4 = mxPoolLayer2.forward(t3)
  t4 = np.reshape(t4,(t4.shape[2], t4.shape[0],t4.shape[1]))


  t5 = convLayer3.forward(t4)

  t5 = np.reshape(t5,(t5.shape[1], t5.shape[2],t5.shape[0]))
  t6 = mxPoolLayer3.forward(t5)
  t6 = np.reshape(t6,(t6.shape[2], t6.shape[0],t6.shape[1]))


  t7 = convLayer4.forward(t6)


  t7 = np.reshape(t7,(t7.shape[1], t7.shape[2],t7.shape[0]))
  t8 = mxPoolLayer4.forward(t7)
  t8 = np.reshape(t8,(t8.shape[2], t8.shape[0],t8.shape[1]))

  t8 = convLayer5.forward(t8)


  t8 = flatten.forward(t8)
  predict = t8

  error += trainingData.binary_cross_entropy(y, predict)

  grad = trainingData.binary_cross_entropy_prime(y, predict)

  dE = grad
  dE = flatten.backward(dE)
  
  dEX = convLayer5.backward(dE,learning_rate)

  dE1 = np.reshape(dEX,(dEX.shape[1], dEX.shape[2],dEX.shape[0]))
  dE2 = mxPoolLayer4.backward(dE1)
  dE2 = np.reshape(dE2,(dE2.shape[2], dE2.shape[0],dE2.shape[1]))
  
  dEX = convLayer4.backward(dE2,learning_rate)  

  dE1 = np.reshape(dEX,(dEX.shape[1], dEX.shape[2],dEX.shape[0]))
  dE2 = mxPoolLayer3.backward(dE1)
  dE2 = np.reshape(dE2,(dE2.shape[2], dE2.shape[0],dE2.shape[1]))

  dE3 = convLayer3.backward(dE2,learning_rate)

  dE3 = np.reshape(dE3,(dE3.shape[1], dE3.shape[2],dE3.shape[0]))
  dE4 = mxPoolLayer2.backward(dE3)
  dE4 = np.reshape(dE4,(dE4.shape[2], dE4.shape[0],dE4.shape[1]))

  dE5 = convLayer2.backward(dE4,learning_rate)

  dE5 = np.reshape(dE5,(dE5.shape[1], dE5.shape[2],dE5.shape[0]))
  dE6 = mxPoolLayer1.backward(dE5)
  dE6 = np.reshape(dE6,(dE6.shape[2], dE6.shape[0],dE6.shape[1]))

  dE7 = convLayer1.backward(dE6,learning_rate)



dat = processor.data


mxPoolKSize = 3
mxPoolStride = 2
mxPoolPadding = 1

processedFeatures = {}

'''Running CNN for test images'''
input_dimensions = (3,112,112)
for k in dat:
  print('Creating features for ' + k)
  if 'images' not in dat[k] : continue;
  t = dat[k]['images']
  t = np.asarray(t)
  t = np.reshape(t, input_dimensions)

  learning_rate = 0.01
  t1 = convLayer1.forward(t)

  t1 = np.reshape(t1,(t1.shape[1], t1.shape[2],t1.shape[0]))
  t2 = mxPoolLayer1.forward(t1)
  t2 = np.reshape(t2,(t2.shape[2], t2.shape[0],t2.shape[1]))


  t3 = convLayer2.forward(t2)

  t3 = np.reshape(t3,(t3.shape[1], t3.shape[2],t3.shape[0]))
  t4 = mxPoolLayer2.forward(t3)
  t4 = np.reshape(t4,(t4.shape[2], t4.shape[0],t4.shape[1]))


  t5 = convLayer3.forward(t4)

  t5 = np.reshape(t5,(t5.shape[1], t5.shape[2],t5.shape[0]))
  t6 = mxPoolLayer3.forward(t5)
  t6 = np.reshape(t6,(t6.shape[2], t6.shape[0],t6.shape[1]))


  t7 = convLayer4.forward(t6)

  t7 = np.reshape(t7,(t7.shape[1], t7.shape[2],t7.shape[0]))
  t8 = mxPoolLayer4.forward(t7)
  t8 = np.reshape(t8,(t8.shape[2], t8.shape[0],t8.shape[1]))


  t9 = convLayer5.forward(t8)

  t_final = flatten.forward(t9)
  processedFeatures[k] = t_final
