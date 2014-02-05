#!/usr/bin/python

"""
(C) Rahul Jha - 2014
License: BSD 3 clause

Implementation of TopicSum based on Mathieu Blonedl's LDA implmentation.
See http://www.mblondel.org/journal/2010/08/21/latent-dirichlet-allocation-in-python/

"""

import numpy as np
import scipy as sp
from scipy.special import gammaln

from SurveyorData import SurveyorData

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class TopicSum:

    def __init__(self, data, alpha, beta):

        self.data = data

        # TODO: needs to change
        """
        data: a hash with S keys, s_i representing each document set, 
              each s_i contains D keys, d_i representing each document, 
              each d_i contains the vectors for each word
        alpha: a vector of size n_topics
        beta: a scalar (FIME: accept vector of size vocab_size)
        """

        num_docsets = len(data.keys())
        num_docs = sum([len(i.keys()) for i in data.values()])

        self.n_topics = 1 + num_docsets + num_docs
        self.alpha = alpha
        self.beta = beta

        # now assign topic ids to the topics for each docset and document
        self.topic_idx = {}

        self.topic_idx['background'] = 0
        idx = 1
        for docset in data.keys():
            self.topic_idx[docset] = idx
            idx += 1

            for doc in data[docset].keys():
                self.topic_idx[doc] = idx
                idx += 1

        assert idx == (len(self.n_topics) - 1)

        self._initialize()

    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:,w] + self.beta) / \
               (self.nz + self.beta * vocab_size)
        right = (self.nmz[m,:] + self.alpha) / \
                (self.nm[m] + sum(self.alpha))
        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.nzw.shape[1]
        num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num        

    def _initialize(self):

        class sentAssgmt:
            def __init__(self, docset, doc):
                self.docset = docset
                self.doc = doc
            def __repr__(self):
                return "(%s, %s)" % (self.docset, self.doc)
                       
        self.sent_assgmts = {}

        matrix_list = []
        for docset in self.data.keys():
            for doc in self.data[docset].keys():
                for sent in self.data[docset][doc]:
                    matrix_list.append(sent)
                    sent_idx = len(matrix_list) - 1
                    self.sent_assgmts[sent_idx] = sentAssgmt(docset, doc)

        matrix = np.matrix(matrix_list)

        n_sents, vocab_size = matrix.shape

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_sents, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size))
        self.nm = np.zeros(n_sents)
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for m in xrange(n_sents):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                cur_topic = np.random.rand(['background', 'content', 'docspecific'])
                if cur_topic == 'background':
                    z = self.topic_idx['background']
                elif cur_topic == 'content':
                    docset = self.sent_assgmts[m].docset
                    z = self.topic_idx[docset]
                elif cur_topic == 'docspecific':
                    doc = self.sent_assgmts[m].doc
                    z = self.topic_idx[doc]

                self.nmz[m,z] += 1
                self.nm[m] += 1
                self.nzw[z,w] += 1
                self.nz[z] += 1
                self.topics[(m,i)] = z    

    def run(self, matrix, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape
        self._initialize(matrix)

        for it in xrange(maxiter):
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m,i)]
                    self.nmz[m,z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w)
                    z = sample_index(p_z)

                    self.nmz[m,z] += 1
                    self.nm[m] += 1
                    self.nzw[z,w] += 1
                    self.nz[z] += 1
                    self.topics[(m,i)] = z

            # FIXME: burn-in and lag!
            yield self.phi()                


if __name__ == "__main__":
    
    topic_file = "/data0/projects/fuse/rdg_experimental_lab/experiments/surveyor_2013/final_experiments/code/final_topics.txt";
    topic_fh = open(topic_file, "r")
    topics = []
    for line in topic_fh:
        topics.append(line.strip())

    dataObj = SurveyorData(topics, "/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/data/input_text/")
    # TODO: add caching here so vectors are not computed again and again
    # store vocab and data vectors
    vectors = dataObj.process_data()

    alpha = [1.0, 5.0, 10.0]
    beta = 0.1
    tsObj = TopicSum(vectors, alpha, beta)
