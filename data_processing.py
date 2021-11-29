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
        glove_vecs = {"startseq": np.full(100, 1), "endseq": np.full(100, -1)}
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
        caption_file_path = "/content/results.csv"
        with open(caption_file_path, mode='r') as file:  # open the file
            reader = csv.reader(file)  # read the file
            next(reader)  # ignore header line
            for row in reader:  # add data to dictionary
            #row[0] itself has all the fields hence need to split wrt '|'
            #eg row[0] = 1000092795.jpg| 0| Two young guys with shaggy hair look at their hands while hanging out in the yard .
              line = row[0].split('|')
              #extract 1000092795.jpg from row[0]
              id = line[0]
              #in some cases split doesnt create length 3 meaning bad data
              if(len(line)<3): continue # hence skipping
              cap = line[2]
              if id not in self.data:
                  self.data[id] = {'captions': list()}
              self.data[id]['captions'].append(self.clean_captions(cap))


    def vectoriser(self, caption):
        """
        cleaning data
        :param caption: caption to be cleaned
        :return: cleaned caption
        """
        # tokenizing the sentence
        word_list = self.tokenizer.tokenize(caption.lower())
        word_list.append("endseq")
        word_list.insert(0, "startseq")
        print(word_list)
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
        image_folder_path = "/content/images/*.jpg"
        #sample file path will be /content/images/24343.jpg
        for file in glob.glob(image_folder_path):  # Pick list of images
            #file will be path : /content/images/24343.jpg
            if fnmatch.fnmatch(file, "*.jpg"):  # check if the file is a jpg file
                image = Image.open(file)  # opening the file
                image = np.asarray(image)   # converting image into array
                image_resize = np.resize(image, (3,112,112))  # reshaping the image , d * h * w, small change instead of h * w * d
                image_resize = np.true_divide(image_resize, 255) # normalization : diving values by 255
                #extract 24343.jpg from /content/images/24343.jpg
                id = file.split('/')[3]
                
                #id = (int)(id.split('.')[0])
                # self.data[id] = {}
                self.data[id] = {'images': (list())}
                #key : 24343.jpg , 'images' will be image matrix
                self.data[id]['images'].append(image_resize)  # storing the image in the dictionary


    '''
    This returns processed training data for MNIST
    '''
    def preprocess_data_mnist(self, x, y, limit):
        zero_index = np.where(y == 0)[0][:limit]
        one_index = np.where(y == 1)[0][:limit]
        all_indices = np.hstack((zero_index, one_index))
        all_indices = np.random.permutation(all_indices)
        x, y = x[all_indices], y[all_indices]
        x = x.reshape(len(x), 1, 28, 28)
        x = x.astype("float32") / 255
        y = np_utils.to_categorical(y)
        y = y.reshape(2,len(y))
        return x, y