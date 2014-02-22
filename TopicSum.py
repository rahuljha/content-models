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
import operator

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

            # TODO: the same doc might belong to multiple topics, fix that
            for doc in data[docset].keys():
                self.topic_idx[doc] = idx
                idx += 1

        assert len(self.topic_idx.keys()) == self.n_topics

        self._initialize()

    def _conditional_distribution(self, m, w, b, c, d):
        """
        Conditional distribution (vector of size 3) over background, content and docspecific topic
        """

        vocab_size = self.nzw.shape[1]
        left = (self.nzw[[b,c,d],w] + self.beta) / \
               (self.nz[[b,c,d]] + self.beta * vocab_size)

        # assert that the sum of words in the three topics for this sentence are the same as the total number of words counted for this sentence
#        assert self.nmz[m, [b,c,d]].sum() == self.nm[m]

        right = (self.nmz[m, [b,c,d]] + self.alpha) / \
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
        n_sents = self.nmz.shape[0]
        lik = 0

        b = self.topic_idx['background']
        b_counts = self.nzw[b,:]
        c_counts = np.zeros(b_counts.size) # since vocab size is the same
        d_counts = np.zeros(b_counts.size)

        # create c and d counts
        for docset in self.data.keys():
            c = self.topic_idx[docset]
            c_counts += self.nzw[c,:]
            for doc in self.data[docset].keys():
                d = self.topic_idx[doc]
                d_counts += self.nzw[d,:]
                
        nzw_top = np.vstack([b_counts, c_counts, d_counts])

        nmz_top = np.zeros((n_sents, 3))
        for m in xrange(n_sents):
                c = self.topic_idx[self.sent_assgmts[m].docset]
                d = self.topic_idx[self.sent_assgmts[m].doc]
#                assert set(self.nmz[m,:].nonzero()[0]) <= set([b,c,d]) # this sentence should not have word counts for any other topics
                nmz_top[m,:] = self.nmz[m,[b,c,d]]
                
#        assert np.sum(self.nzw) == np.sum(nzw_top) # number of words shouldn't change
#        assert np.sum(self.nmz) == np.sum(nmz_top) 
        
        for z in [0,1,2]:
            lik += log_multi_beta(nzw_top[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_sents):
            lik += log_multi_beta(nmz_top[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha)

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

        self.matrix = np.array(matrix_list)

        n_sents, vocab_size = self.matrix.shape

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
            for i, w in enumerate(word_indices(self.matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                choice = ['background', 'content', 'docspecific']
                cur_topic = choice[np.random.randint(3)]
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

#        assert np.sum(self.nmz) == np.sum(self.nzw) # since # words shouldn't be different

    def run(self, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        n_sents, vocab_size = self.matrix.shape
#        self._initialize(matrix)

        for it in xrange(maxiter):
            for m in xrange(n_sents):
                for i, w in enumerate(word_indices(self.matrix[m, :])):
                    z = self.topics[(m,i)]
                    self.nmz[m,z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1

                    d = self.topic_idx[self.sent_assgmts[m].doc]
                    c = self.topic_idx[self.sent_assgmts[m].docset]
                    b = self.topic_idx['background']

                    cand_topics = ['background', 'content', 'docspecific']
                    p_z = self._conditional_distribution(m, w, b, c, d)
                    cur_topic = cand_topics[sample_index(p_z)]
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

            # FIXME: burn-in and lag!
            yield self.phi() 


def write_topic(filekey, word_probs):
    
    fh = open("/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/code/out_topics/"+filekey+".txt", "w")
    for x in sorted(word_probs.iteritems(), key=operator.itemgetter(1), reverse=True):
        fh.write("%s\t%f\n" % (x[0], x[1]))
    fh.close()

if __name__ == "__main__":
    
    topic_file = "/data0/projects/fuse/rdg_experimental_lab/experiments/surveyor_2013/final_experiments/code/final_topics.txt";
    topic_fh = open(topic_file, "r")
    topics = []
    for line in topic_fh:
        topics.append(line.strip())

    dataObj = SurveyorData(topics[0:2], "/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/data/input_text/")
    # TODO: add caching here so vectors are not computed again and again
    # store vocab and data vectors
    vectors = dataObj.process_data()

    # alpha = [background, content, docspecific]
    alpha = [10.0, 1.0, 5.0]
    beta = 0.1
    tsObj = TopicSum(vectors, alpha, beta)
    cur_phi = None
    for it, phi in enumerate(tsObj.run(10)):
        print "Iteration", it
        print "Likelihood", tsObj.loglikelihood()
        all_phi = phi

    vocab = dataObj.get_vocab()
    write_topic('background', dict(zip(vocab, phi[0])))

    for docset in tsObj.data.keys():
        dtopic = tsObj.topic_idx[docset]
        dphi = phi[dtopic]
        word_probs = dict(zip(vocab, dphi))
        write_topic(docset, word_probs)

        for doc in tsObj.data[docset]:
            dtopic = tsObj.topic_idx[doc]
            dphi = phi[dtopic]
            word_probs = dict(zip(vocab, dphi))
            write_topic(doc, word_probs)  


