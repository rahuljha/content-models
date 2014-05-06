#!/usr/bin/python

"""LexPageRank, a PageRank-inspired algorithm for generating multidocument.
sentence summaries."""

import itertools
from collections import defaultdict
import numpy as np
import operator
import sys

# The minimum similarity for sentences to be considered similar by LexPageRank.
# TODO: tune these
MIN_LEXPAGERANK_SIM = 0.2
EPSILON = 0.001

def sim_adj_matrix(sents, sim_hash, min_sim=0.1):
    """Compute the adjacency matrix of a list of tokenized sentences,
    with an edjge if the sentences are above a given similarity."""
    # return [[1 if sim_hash[s1][s2] > min_sim else 0
    #          for s2 in sents]
    #         for s1 in sents]

    return np.array([[sim_hash[s1][s2] for s2 in sents] for s1 in sents])


def normalize_matrix(matrix):
    """Given a matrix of number values, normalize them so that a column
    sums to 1."""

    nrows = matrix.shape[0]
    for col in xrange(matrix.shape[1]):
        tot = float(sum(matrix[:,col]))
        
        for row in xrange(nrows):
            try:
                matrix[row][col] = matrix[row][col]/tot
            except ZeroDivisionError:
                pass
    return matrix


def pagerank(matrix, bias, d=0.85):
    """Given a matrix of values, run the PageRank algorithm on them
    until the values converge. See Wikipedia page for source."""
    n = matrix.shape[0]
    rank = 0
    new_rank = np.array([1.0 / n] * n)
    for i in range(0,200):
        print "iteration: "+str(i)
        rank = new_rank
        new_rank = np.array([(1.0-d)/n] * n) + d * np.dot(matrix, rank)
#        new_rank = (1.0-d) * bias + d * np.dot(matrix, rank)
        # new_rank = [(((1.0-d) / n) +
        #              d * sum((rank[i] * link) for i, link in enumerate(row)))
        #             for row in matrix]
        if(has_converged(rank, new_rank)):
            break
    return new_rank

def has_converged(x, y, epsilon=EPSILON):
    """Are all the elements in x are within epsilon of their y's?"""
    for a, b in itertools.izip(x, y):
        if abs(a - b) > epsilon:
            return False
    return True

def has_converged(x, y, epsilon=EPSILON):
    """Are all the elements in x are within epsilon of their y's?"""
    for a, b in itertools.izip(x, y):
        if abs(a - b) > epsilon:
            return False
    return True


def gen_lexrank_summary(orig_sents, max_words):
    tok_sents = [tokenize.word_tokenize(orig_sent)
                 for orig_sent in orig_sents]
    adj_matrix = normalize_matrix(sim_adj_matrix(tok_sents))
    rank = pagerank(adj_matrix)
    return gen_summary_from_rankings(rank, tok_sents, orig_sents, max_words)

def get_length(sent_array):
    length = 0
    for sent in sent_array:
        length += len(sent)

    return length

###############################################################################
if __name__ == '__main__':

    topic = sys.argv[1]
    damping = float(sys.argv[2])
    
    in_sents = "/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/code/cache/kls/"+topic+".txt"
    in_sims = "/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/code/cache/sims/"+topic+".txt"

    text = {}
    sids = []
    bias = []

    sent_fh = open(in_sents, "r")
    for line in sent_fh:
        vals = line.strip().split("\t")
        sid = vals[1]+"_"+vals[2]
        sids.append(sid)
        bias.append(15-float(vals[3]))
#        bias.append(1/float(vals[3]))
        text[sid] = vals[4]

    bias = np.array(bias)
    bias /= sum(bias)
    sent_fh.close()

    sim_hash = defaultdict(dict)

    sim_fh = open(in_sims, "r")
    for line in sim_fh:
        (s1, s2, sim) = line.strip().split()
        sim_hash[s1][s2] = float(sim)
        sim_hash[s2][s1] = float(sim)

    # add self similarities
    for sid in sids:
        sim_hash[sid][sid] = 1
        
    sim_matrix = normalize_matrix(sim_adj_matrix(sids, sim_hash))
    prs = pagerank(sim_matrix, bias, d=damping)

    pr_hash = dict(zip(sids, prs))

    sids = reversed(sorted(pr_hash.iteritems(), key=operator.itemgetter(1)))

    max_len = 2000
    cur_len = 0
    cur_summary = []
    outfh = open("/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/code/out_lrs/"+topic+".txt", "w")
    for (sid,pr) in sids:
        outfh.write("%s\t%.5f\t%s\n" % (sid, pr, text[sid]))

    #     next_sent = text[sid]
    #     cur_len = get_length(cur_summary + [next_sent])

    #     if(cur_len >= max_len):
    #         past_len = get_length(cur_summary)
    #         diff = max_len-past_len
    #         cur_summary.append(next_sent[0:diff])
    #         break
    #     else:
    #         cur_summary.append(next_sent)

    # outfh = open("/data0/projects/fuse/rdg_experimental_lab/experiments/content_models/out_summaries/"+topic+"_"+str(damping)+".txt", "w")
    # for sent in cur_summary:
    #     outfh.write(sent+"\n")
        
    # outfh.close()
        
