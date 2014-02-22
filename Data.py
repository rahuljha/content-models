#!/usr/bin/python

# This is the base class for reading data from any source and converting it to the format required by TopicSum
# For making this work on a new data set, you should inherit from this class and then implement the below two methods

class Data:

    def process_data(self):
        """
        read all the data and return a data structure of the following format that can be sent to TopicSum
        data: a hash with S keys, s_i representing each topic (or document set)
              each s_i contains D keys, d_i representing each document
              each d_i contains the vectors for each word
              """
        raise NotImplementedError("process_data Not Implemented")
    def get_vocab(self):
        """
        returns the vocab created 
        """

        raise NotImplementedError("process_data Not Implemented")
