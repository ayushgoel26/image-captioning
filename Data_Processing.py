import csv
import string
from pprint import pprint


class DataProcessing:
    """
    Class to Process the Data

    Methods
    -------
    caption_reader(): read captions from the txt/csv file
    """

    def __init__(self):
        """
        Constructor for Data Processing class
        """
        self.data = {}
        self.punctuations = string.punctuation

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
                    self.data[row[0]] = list()
                self.data[row[0]].append(self.clean_captions(row[1]))
            pprint(self.data)

    def clean_captions(self, caption):
        """
        cleaning data
        :param caption: caption to be cleaned
        :return: cleaned caption
        """
        word_list = caption.lower().split()  # tokenize
        for punctuation in self.punctuations:
            if punctuation in word_list:
                word_list.remove(punctuation)
        return ' '.join(word_list)


if __name__ == "__main__":
    data_processing = DataProcessing()
    data_processing.caption_reader()
