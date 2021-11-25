import csv
import string
from PIL import Image
import os
import fnmatch
import numpy as np
from nltk.tokenize import RegexpTokenizer
import torch


class Processor:
    """
    Class to Process the Textual Data

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
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.glove_dir = "/Users/ayush/Downloads/glove/glove.6B.200d.txt"
        self.caption_file_path = "/Users/ayush/Downloads/captions.txt"
        self.word_embeddings = self.word_vec_reader()

    def word_vec_reader(self):
        print('Reading Glove Vectors')
        glove_vecs = {}
        with open(self.glove_dir, encoding="utf-8", mode='r') as file:  # open the file
            for line in file.readlines():
                line = line.replace("\n", "").split(" ")
                glove_vecs[line[0]] = [float(i) for i in line[1:]]
        return glove_vecs

    def caption_reader(self):
        """
        read data into dictionary from txt/csv file
        """
        # hardcoded path change in future
        with open(self.caption_file_path, mode='r') as file:  # open the file
            reader = csv.reader(file)  # read the file
            next(reader)  # ignore header line
            for row in reader:  # add data to dictionary
                if row[0] not in self.data:
                    self.data[row[0]] = {'captions': list()}
                self.data[row[0]]['captions'].append(self.vectoriser(row[1]))

    def vectoriser(self, caption):
        """
        cleaning data
        :param caption: caption to be cleaned
        :return: cleaned caption
        """
        # tokenizing the sentence
        word_list = self.tokenizer.tokenize(caption.lower())
        # remove hanging words
        word_list = [self.get_word_embedding(word) for word in word_list if len(word) > 1 or word not in self.punctuations]
        return word_list

    def get_word_embedding(self, word):
        if word in self.word_embeddings:
            return torch.Tensor(self.word_embeddings[word])
        print('  Word not found: ' + word)
        return torch.zeros(1, 1, 100)

    def process_images(self):
        """
        Read the images, change its dimension and store in dictionary
        """
        # hardcoded path change in future
        image_folder_path = "/Users/revagupta/Documents/UTD/Second Sem/CS-6375 ML/ML_Project/Images"
        for file in os.listdir(image_folder_path):  # Pick list of images
            if fnmatch.fnmatch(file, "*.jpg"):  # check if the file is a jpg file
                image = Image.open(image_folder_path + "/" + file)  # opening the file
                image = np.asarray(image)  # converting image into array
                image_resize = np.resize(image, (224, 224, 3))  # reshaping the image
                image_resize = np.true_divide(image_resize, 255)   # returns the true division of the input
                self.data[file]["image_vector"] = image_resize  # storing the image in the dictionary
