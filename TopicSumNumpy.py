#!/usr/bin/python

import numpy as np
from scipy.special import gammaln

from SurveyorData import SurveyorData
import operator


class TopicSumNumpy:

    def __init__(self, dataObj, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.dataObj = dataObj

        # assign topic numbers to each docset and document 
        self.topic_cnt = 1+len(dataObj.docsets)+len(dataObj.docs)

        # topic_idx = 0 will be assigned to background
        start_idx = 1
        # next |docsets| topics assigned to docsets
        self.docset2topic = np.array(range(start_idx, start_idx+len(dataObj.docsets)), dtype=np.uint)
        start_idx = start_idx+len(dataObj.docsets)
        # next |docs| topics assigned to docs
        self.doc2topic = np.array(range(start_idx, start_idx+len(dataObj.docs)), dtype=np.uint)

        # this is the beta copied for each topic separately so that phi can be calculated easily
        self.beta_ex = np.zeros(self.topic_cnt)
        # background
        self.beta_ex[0] = self.beta[0]
        # content topics
        self.beta_ex[self.docset2topic] = self.beta[1]
        # doc specific topics
        self.beta_ex[self.doc2topic] = self.beta[2]
            
        assert self.topic_cnt == len(self.docset2topic) + len(self.doc2topic) + 1
        
        # sets up nmz, nzw, nm, nz and state
        self.setUpVars()
        # initialize gibbs
        self.initGibbs()

    # m = documents, z = topics, w = words
    def setUpVars(self):
        sent_cnt = len(self.dataObj.sent_vecs)
        vocab_cnt = len(self.dataObj.vocab)
        self.nmz = np.zeros([sent_cnt, self.topic_cnt], dtype=np.uint)
        self.nzw = np.zeros([self.topic_cnt, vocab_cnt], dtype=np.uint)
        self.nm = np.zeros(sent_cnt, dtype=np.uint)
        self.nz = np.zeros(self.topic_cnt, dtype=np.uint)

        # self.state is a words x 5 matrix of uint32: first column is the document ident, second column is the word ident, third column is assigned topic ident, fourth column is the docset topic for this word, fifth column is the doc topic for this word
        self.state = np.empty((np.sum(dataObj.sent_vecs),3), dtype=np.uint)
        self.topic_ids = np.empty((np.sum(dataObj.sent_vecs),2), dtype=np.uint)
        index = 0
        for sent_idx, vec in enumerate(dataObj.sent_vecs):
            docset_topic_idx = self.docset2topic[self.dataObj.sent2docsets[sent_idx]]
            doc_topic_idx = self.doc2topic[self.dataObj.sent2docs[sent_idx]]

            for word_idx in np.nonzero(dataObj.sent_vecs[sent_idx])[0]:
                count = dataObj.sent_vecs[sent_idx][word_idx]
                for c in xrange(count):
                  self.state[index, 0] = sent_idx
                  self.state[index, 1] = word_idx
                  self.state[index, 2] = 1000000000 # Dummy bad value.
                  self.topic_ids[index, 0] = docset_topic_idx
                  self.topic_ids[index, 1] = doc_topic_idx
                  index += 1

    def initGibbs(self):

        # loop over words
        for seq_idx in xrange(self.state.shape[0]):
            # choose between background, content or document topic
            m = self.state[seq_idx, 0]
            w = self.state[seq_idx, 1]
            z = None            
            abstract_topic = np.random.randint(3) 

            if(abstract_topic == 0): # background
                z = 0
            elif(abstract_topic == 1): # docset specific topic
                z = self.topic_ids[seq_idx, 0]
            elif(abstract_topic == 2): # doc specific topic
                z = self.topic_ids[seq_idx, 1]

            self.nmz[m,z] += 1
            self.nzw[z,w] += 1
            self.nz[z] += 1
            self.state[seq_idx, 2] = z
                
    def runGibbs(self, iters = 20):

        for i in xrange(iters):
            print "iteration: %d" % i
            for seq_idx in xrange(self.state.shape[0]):
                m = self.state[seq_idx, 0]
                w = self.state[seq_idx, 1]
                z = self.state[seq_idx, 2]
                self.nmz[m,z] -= 1
                self.nzw[z,w] -= 1
                self.nz[z] -= 1

                # background, content and doc topics
                z_idx = [0, self.topic_ids[seq_idx, 0], self.topic_ids[seq_idx, 1]] 
                z_probs = [0,0,0]

                total = 0
                for t in range(3):
                    cur_z = z_idx[t]
                    top1 = self.nzw[cur_z, w] + beta[t]
                    bottom1 = self.nz[cur_z] + self.nzw.shape[1]*beta[t]
                    top2 = self.nmz[m, cur_z] + alpha[t]

                    z_probs[i] = (top1/bottom1)*top2
                    total += z_probs[t]

                # select the state
                rand = np.random.random_sample() * total
                total = 0
                new_z = 50000
                for t in range(3):
                    new_z = z_idx[t]
                    total += z_probs[t]
                    if(total > rand):
                        break

                self.state[seq_idx,2] = new_z
                self.nzw[new_z, w] += 1
                self.nz[new_z] += 1
                self.nmz[m, new_z] += 1

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.nzw.shape[1]
        num = self.nzw + np.transpose(np.tile(self.beta_ex, (V,1)))
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num        

    def write_topic(self, filekey, word_probs):

        fh = open("/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/code/test_topics_numpy/"+filekey+".txt", "w")
        for x in sorted(word_probs.iteritems(), key=operator.itemgetter(1), reverse=True):
            fh.write("%s\t%f\n" % (x[0], x[1]))
        fh.close()

            
if __name__ == "__main__":
    
    topic_file = "/data0/projects/fuse/rdg_experimental_lab/experiments/surveyor_2013/final_experiments/code/final_topics.txt";
    topic_fh = open(topic_file, "r")
    topics = []
    for line in topic_fh:
        topics.append(line.strip())

    dataObj = SurveyorData(topics[0:5], "/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/data/input_text/")
    # TODO: add caching here so vectors are not computed again and again
    # store vocab and data vectors

    # alpha = [background, content, docspecific]
    alpha = [10.0, 1.0, 5.0]
    beta = [1.0, 0.1, 1.0]

    tsObj = TopicSumWeave(dataObj, alpha, beta)
    tsObj.runGibbs()
    phi = tsObj.phi()

    vocab = dataObj.get_vocab()
    tsObj.write_topic('background', dict(zip(vocab, phi[0])))

    for (idx, docset) in enumerate(dataObj.docsets):
        ctopic = tsObj.docset2topic[idx]
        dphi = phi[ctopic]
        word_probs = dict(zip(vocab, dphi))
        tsObj.write_topic(docset, word_probs)

        for (idx, doc) in enumerate(dataObj.docs):
            dtopic = tsObj.doc2topic[idx]
            dphi = phi[dtopic]
            word_probs = dict(zip(vocab, dphi))
            tsObj.write_topic(doc, word_probs)
