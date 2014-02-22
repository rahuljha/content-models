#!/usr/bin/python

import os
import glob
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize

import numpy as np

from Data import Data

class SurveyorData(Data):
    
    def __init__(self, topics, data_root):
        """ accepts a location and reads sentences from all the documents in all the provided topics at this location
        """
        self.sentences = defaultdict(dict)
        vocab_hash = {}

        for topic in topics:
            input_files = glob.glob("/".join([data_root, topic, "*.txt"]))
            for infile in input_files:
                pid = os.path.basename(infile).replace(".txt", "")
                self.sentences[topic][pid] = []
                infh = open(infile, "r")
                for line in infh:
                    words = [w.lower() for w in wordpunct_tokenize(line.strip())]
                    self.sentences[topic][pid].append(words)
                    for w in words:
                        vocab_hash[w] = 1

        self.vocab = sorted(vocab_hash.keys())

    def generate_vectors(self, sents):
        vocab_size = len(self.vocab)
        m = np.zeros((len(sents), vocab_size))
        for i,sent in enumerate(sents):
            for word in sent:
                idx = self.vocab.index(word)
                m[i][idx] += 1
        
        return m

    def process_data(self):
        vectors = defaultdict(dict)

        for topic in self.sentences.keys():
            for doc in self.sentences[topic]:
                vectors[topic][doc] = self.generate_vectors(self.sentences[topic][doc])

        return vectors

    def get_vocab(self):
        return self.vocab

if __name__ == "__main__":
    topic_file = "/data0/projects/fuse/rdg_experimental_lab/experiments/surveyor_2013/final_experiments/topics.txt";
    topic_fh = open(topic_file, "r")
    topics = []
    for line in topic_fh:
        topics.append("_".join(line.strip().split(" ")).lower())

    dataObj = SurveyorData(topics[0:2], "/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/data/input_text/")
    dataObj.process_data()

    print "testing"
    
