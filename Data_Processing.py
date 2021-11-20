import csv
import string
from PIL import Image
import os
import fnmatch
import numpy as np
import pandas as pd


class DataProcessing:
    """
    Class to Process the Data

    Methods
    -------
    caption_reader(): read captions from the txt/csv file
    clean_caption(): clean the captions
    process_images(): change the image dimensions and store in data
    """

    def __init__(self):
        """
        Constructor for Data Processing class
        """
        self.data = {}
        self.punctuations = string.punctuation
        self.embedding_dictionary = {}

    def caption_reader(self):
        """
        read data into dictionary from txt/csv file
        """
        # hardcoded path change in future
        caption_file_path = "/Users/revagupta/Documents/UTD/Second Sem/CS-6375 ML/ML_Project/captions.txt"
        with open(caption_file_path, mode='r') as file:  # open the file
            reader = csv.reader(file)  # read the file
            next(reader)  # ignore header line
            for row in reader:  # add data to dictionary
                if row[0] not in self.data:
                    self.data[row[0]] = {'captions': list()}
                self.data[row[0]]['captions'].append(self.clean_captions(row[1]))

    def clean_captions(self, caption):
        """
        cleaning data
        :param caption: caption to be cleaned
        :return: cleaned caption
        """
        word_list = caption.lower().split()  # tokenize
        word_list = [word for word in word_list if len(word) > 1]  # remove hanging words
        for punctuation in self.punctuations:  # remove punctuations
            if punctuation in word_list:
                word_list.remove(punctuation)
        return ' '.join(word_list)

    # def word_embedding(self):
    #     """
    #
    #     """
    #     glove_dir = "/Users/revagupta/Documents/UTD/Second Sem/CS-6375 ML/ML_Project/glove/glove.6B.200d.txt"
    #     with open(glove_dir, encoding="utf-8", mode='r') as file: # open the file
    #         reader = csv.reader(file)  # read the file
    #         for row in reader:
    #             print(row)
    #             # row = row[0].split()
    #             # print(row)
    #             # word = row[0]
    #             # coef = np.asarray(row[1:], dtype='float32')
    #             # self.embedding_dictionary[word] = coef

    def process_images(self):
        """
        Read the images, change its dimension and store in dictionary
        """
        # hardcoded path change in future
        image_folder_path = "/Users/revagupta/Documents/UTD/Second Sem/CS-6375 ML/ML_Project/Images"
        for file in os.listdir(image_folder_path):  # Pick list of images
            if fnmatch.fnmatch(file, "*.jpg"):  # check if the file is a jpg file
                image = Image.open(image_folder_path + "/" + file)  # opening the file
                image = np.asarray(image)   # converting image into array
                image_resize = np.resize(image, (224, 224, 3))  # reshaping the image
                self.data[file]["image_vector"] = image_resize  # storing the image in the dictionary


if __name__ == "__main__":
    data_processing = DataProcessing()
    print("Processing Captions")
    data_processing.caption_reader()
    print("Processing Images")
    data_processing.process_images()
