from data_processing import Processor
import pickle
import os.path

if not os.path.isfile('data.json'):
    processor = Processor()

    print("Processing Captions")
    processor.caption_reader()
    print("Captions Processed")

    print("Processing Images")
    processor.process_images()
    print("Images Processed")

    with open("data.json", "wb") as file:
        pickle.dump(processor.data, file)
