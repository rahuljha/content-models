#!/usr/bin/python

import os
import glob
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize
import itertools
from string import punctuation
from collections import defaultdict

import numpy as np

from Data import Data

class SurveyorData(Data):
    
    def __init__(self, topics, data_root):
        """ accepts a location and reads sentences from all the documents in all the provided topics at this location
        """

        self.topic_doc_sents = defaultdict(dict)
        self.topic_doc_origs = defaultdict(dict)
        self.topic_doc_vecs = defaultdict(dict)

        self.vocab = []
        # read data from files and update topic_doc_sents and vocab, topic_doc_sentences now contains a hash from topics to docs to words
        self.read_data(topics, data_root)
        
        self.docsets = self.topic_doc_sents.keys()
        self.docs = list(itertools.chain.from_iterable([i.keys() for i in self.topic_doc_sents.values()]))
       
        numsents = sum([len(arr) for arr in [list(itertools.chain.from_iterable(i.values())) for i in self.topic_doc_sents.values()]])
        self.sent_vecs = np.zeros((numsents, len(self.vocab)), dtype=np.int)
        self.sent2docsets = np.zeros(numsents, dtype=np.int)
        self.sent2docs = np.zeros(numsents, dtype=np.int)

        self.docsets2sents = defaultdict(list)
        
        # updates self.doctopic2sent and self.numsents
        self.build_vectors()

    def build_vectors(self):

        sent_idx = 0
        for topic in self.topic_doc_sents:
            for doc in self.topic_doc_sents[topic]:
                topic_idx = self.docsets.index(topic)
                doc_idx = self.docs.index(doc)
                
                self.topic_doc_vecs[topic][doc] = []
                for sent in self.topic_doc_sents[topic][doc]:
                    for word in sent:
                        word_idx = self.vocab.index(word)
                        self.sent_vecs[sent_idx][word_idx] += 1

                    self.sent2docsets[sent_idx] = topic_idx
                    self.sent2docs[sent_idx] = doc_idx
                    self.topic_doc_vecs[topic][doc].append(self.sent_vecs[sent_idx])

                    sent_idx += 1

        assert len(self.sent2docs) == sent_idx

    def read_data(self, topics, data_root):

        vocab_hash = {}

        for topic in topics:
            input_files = glob.glob("/".join([data_root, topic, "*.txt"]))
            for infile in input_files:
                pid = os.path.basename(infile).replace(".txt", "")
                pid = topic+"_"+pid # this is done so that the same document appearing in two different topics will be treated as duplicates
                self.topic_doc_sents[topic][pid] = []
                self.topic_doc_origs[topic][pid] = []
                infh = open(infile, "r")
                for line in infh:
                    words = [w.strip(punctuation).lower() for w in wordpunct_tokenize(line.strip()) if w.strip(punctuation) != ""]
                    self.topic_doc_sents[topic][pid].append(words)
                    self.topic_doc_origs[topic][pid].append(line.strip())
                    for w in words:
                        vocab_hash[w] = 1

        self.vocab = sorted(vocab_hash.keys())

    def get_vocab(self):
        return self.vocab

if __name__ == "__main__":
    topic_file = "/data0/projects/fuse/rdg_experimental_lab/experiments/surveyor_2013/final_experiments/code/final_topics.txt";
    topic_fh = open(topic_file, "r")
    topics = []
    for line in topic_fh:
        topics.append(line.strip())

    dataObj = SurveyorData(topics[0:2], "/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/data/input_text/")


    
