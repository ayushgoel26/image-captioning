import string

from PIL import Image
import numpy as np
from nltk.tokenize import RegexpTokenizer
import torch
from gensim.models import Word2Vec
import requests
from io import BytesIO
from conf import WORD_VECTORS_FILE, IMAGE_LIMIT


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
        self.punctuations = string.punctuation  # picking up punctuation
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.caption_file_path = \
            "https://raw.githubusercontent.com/ayushgoel26/datasets/main/ImageProcessingData/captions.txt"
        self.image_file_path = "https://raw.githubusercontent.com/ayushgoel26/datasets/main/ImageProcessingData/Images/"
        # self.word_embeddings = self.word_vec_reader()
        self.captions_list = []     # storing all captions for word2vec training
        self.vocabulary = None      # store word corpus
        self.model = None           # store Word2Vec model
        self.words = None           # store the words
        self.word_vectors = None    # store word vectors for the corpus

    def caption_reader(self):
        """
        read data into dictionary from txt/csv file
        """
        page = requests.get(self.caption_file_path)     # read the caption url
        count = 0
        for text in page.text.split('\n')[1:-1]:    # read captions
            line = text.split(',')      # split the data
            if line[0] not in self.data:
                self.data[line[0]] = {'captions': list()}
            word_list = self.clean_caption(line[1])     # clean the caption
            self.captions_list.append(word_list)    # append the captions in list for word2vec training
            self.data[line[0]]['captions'].append(word_list)    # add the captions into the data dictionary
            count += 1
            if count == IMAGE_LIMIT:
                break
        self.generate_word_vectors()    # generate word vectors using gensim
        self.vectoriser()   # add vectorized captions to the data dictionary

    def generate_word_vectors(self):
        """
        Learning word embedding from our text corpus
        """
        # defining a model; minimum count -> words that occur less than this count will be ignored
        self.model = Word2Vec(self.captions_list, min_count=1)  # Training model
        print("Word to Vector model ->", self.model)
        self.vocabulary = list(self.model.wv.vocab)     # saving the words corpus in a list
        print("Vocabulary length ->", len(self.vocabulary))
        self.model.save(WORD_VECTORS_FILE)  # save the word to vector model
        self.generate_list_word_vecs()  # making a list of the words with its vectors

    def generate_list_word_vecs(self):
        """
        Make lists of the words and their word vectors
        """
        self.words = []
        self.word_vectors = []
        for k in self.model.wv.vocab:   # iterate through the words in the word2vec model that we saved
            self.words.append(k)
            self.word_vectors.append(torch.from_numpy(self.model[k]))
        self.word_vectors = torch.stack(self.word_vectors)  # convert a list of tensors into a overall tensor

    def visualize_word_embedding(self):
        """
        Visualize the trained word to vector
        """
        from sklearn.decomposition import PCA
        from matplotlib import pyplot
        # fit a 2d PCA model to the vectors
        x_axis = self.model[self.model.wv.vocab]
        # Performs linear dimensionality reduction using SVD of the data to decrease to lower dimensional space
        pca = PCA(n_components=2)
        result = pca.fit_transform(x_axis)
        # create a scatter plot of the prediction
        pyplot.scatter(result[:, 0], result[:, 1])
        for i, word in enumerate(self.vocabulary):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.title("Word Embedding")
        pyplot.show()

    def vectoriser(self):
        """
        Change the captions in the data to word vectors
        """
        for key in self.data:
            for index in range(len(self.data[key]['captions'])):
                # change the captions to word vector
                self.data[key]['captions'][index] = \
                    [self.get_word_embedding(word) for word in self.data[key]['captions'][index]]

    def clean_caption(self, caption):
        """
        cleaning data
        :param caption: caption to be cleaned
        :return: cleaned caption
        """
        # tokenizing the sentence
        word_list = self.tokenizer.tokenize(caption.lower())    # tokenize the caption
        word_list.append("endseq")  # append end words to the wordlist
        word_list.insert(0, "startseq")     # append start word to the start if wordlist
        # remove hanging words and punctuations
        word_list = [word for word in word_list if len(word) > 1 or word not in self.punctuations]
        return word_list

    def get_word_embedding(self, word):
        """
        gives the vector of the passed word
        :param word: The words whose vector is needed
        :return: vector of the word
        """
        if word in self.vocabulary:
            return torch.Tensor(self.model[word])   # searches the words vector in the gensim model
        print('  Word not found: ' + word)
        return torch.zeros(1, 1, 100) # returns a vector with zeros if word not found

    def process_images(self):
        """
        Read the images, change its dimension and store in dictionary
        """
        for key in self.data.keys():
            response = requests.get(self.image_file_path + key) # picking images from github
            image = Image.open(BytesIO(response.content))   # opening the images
            image = np.asarray(image)  # converting image into array
            # reshaping the image , d * h * w, small change instead of h * w * d
            image_resize = np.resize(image, (3, 112, 112))
            image_resize = np.true_divide(image_resize, 255)  # normalization : diving values by 255
            self.data[key]['image'] = image_resize  # storing the image in the dictionary
