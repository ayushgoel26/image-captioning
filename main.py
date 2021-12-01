import gensim.models
import pickle
import os.path

from data.processing import Processor
from networks.net import CaptionGenerator
from conf import WORD_VECTORS_FILE

processor = Processor()
if not os.path.isfile('data/data.json'):
    print("Processing Captions")
    processor.caption_reader()
    print("Captions processed")

    print("Pre Processing Images")
    processor.process_images()
    print("Images pre processed")

    print("Writing data into json file")
    with open('data/data.json', 'wb') as file:
        pickle.dump(processor.data, file)

else:
    print("Reading data from json file")
    with open('data/data.json', 'rb') as file:
        processor.data = pickle.load(file)
    processor.model = gensim.models.Word2Vec.load(WORD_VECTORS_FILE)
# processor.visualize_word_embedding()

caption_generator = CaptionGenerator(processor)
# caption_generator.get_word([0])
caption_generator.train_cnn(10)
