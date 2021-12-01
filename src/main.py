from processing import Processor
from src.networks.model import CaptionGenerator

processor = Processor()

print("Processing Captions")
processor.caption_reader()  # reading and cleaning captions
print("Captions processed")

print("Pre Processing Images")
processor.process_images()  # reading and pre processing images
print("Images pre processed")

processor.visualize_word_embedding()    # visualizing the word to vectors created
caption_generator = CaptionGenerator(processor)     # making object for the testing and training class
caption_generator.train_cnn(1)  # training the CNN
caption_generator.save_cnn_parameters()     # saving the CNN Parameters
caption_generator.train_rnn(1) # Training the RNN with the trained CNN output
