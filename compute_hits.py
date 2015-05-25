#!/usr/bin/python

import sys
import re
import glob
from os.path import basename

from collections import defaultdict, Counter

import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import bigrams

sys.path.append("/data0/projects/aanpy")
from utils import compute_sim

topic = sys.argv[1]

G = nx.DiGraph()

sent_text = {}
citing_sents = defaultdict(list)
cited_sents = defaultdict(list)

# read from original files instead

# read your file with citing sentences, you only need a file from which you can get a sentence and sid as seen below
citing_sents_fh = open("/PATH_TO_YOUR_DATA/%s.txt" % topic)
for line in citing_sents_fh:
    (sid, temp1, temp2, sent) = line.strip().split(" ::: ")

    pid = sid.split("_")[0]
    sid = "citing-"+sid
    sent_text[sid] = sent
    citing_sents[pid].append(sid)
    G.add_node(sid)
        
citing_sents_fh.close()

# read all source sentences and add to network and a map from pid -> sent
cited_files = glob.glob("/PATH_TO_YOUR_DATA/cited_text/%s/*" % topic)
for fpath in cited_files:
    pid = basename(fpath).replace(".txt", "")
    infh = open(fpath)
    for (snum,line) in enumerate(infh):
        sid = "cited-%s_%d" % (pid, snum)
        sent_text[sid] = line.strip()
        cited_sents[pid].append(sid)
        G.add_node(sid)

    infh.close()

# read citation network and add an edge for sentences if sim > 0.2

idfs = {}
idfh = open("YOUR_IDF_FILE.idfs.txt")
for line in idfh:
    (w,idf) = line.strip().split("\t")
    idfs[w] = float(idf)

swords = set(stopwords.words('english'))
nw_fh = open("YOUR_CITATION_NETWORKS/%s.txt" % topic)

for line in nw_fh:
    (citing, cited) = line.strip().split("\t")
    for i in citing_sents[citing]:
        for j in cited_sents[cited]:

            ugrams_i = [w.lower() for w in re.split("\W", sent_text[i]) if w]
            ugrams_j = [w.lower() for w in re.split("\W", sent_text[j]) if w]
            
            bgrams_i = [" ".join(bg) for bg in bigrams(ugrams_i)]
            bgrams_j = [" ".join(bg) for bg in bigrams(ugrams_j)]

            ugrams_i_cw = [w for w in ugrams_i if w not in swords]
            ugrams_j_cw = [w for w in ugrams_j if w not in swords]

            bgrams_i_cw = [w for w in bgrams_i if ((w.split(" ")[0] not in swords) and w.split(" ")[1] not in swords)]
            bgrams_j_cw = [w for w in bgrams_j if ((w.split(" ")[0] not in swords) and w.split(" ")[1] not in swords)]

            tfs_i = Counter(ugrams_i_cw + bgrams_i_cw)
            tfs_j = Counter(ugrams_j_cw + bgrams_j_cw)

            lexsim = compute_sim(tfs_i, tfs_j, idfs)

            if(lexsim > 0.1):
                G.add_edge(i,j)
try:
    h,a = nx.hits(G)
except:
    try:
        h,a = nx.hits(G, tol=1e-4)
    except:
        try:
            h,a = nx.hits(G, tol=1e-2)
        except:
            h,a = nx.hits(G, tol=1e-1)

#print top 10 HITS nodes and their edges

# cnt = 0
# covered = set()
# for (sid,hit) in [i for i in reversed(sorted(h.items(), key = lambda x:x[1]))]:
#     if(cnt > 10):
#         break
#     successors = set(G.successors(sid))
#     if successors & covered:
#         continue

#     covered = covered | successors
#     cnt += 1
        
#     print "-------"
#     print "[["+sent_text[sid]+", "+str(hit)+"]]"
#     print "+++++"

#     for succ in G.successors(sid):
#         print "  --> "+succ+": "+sent_text[succ]
        
#     print "-------"

# sys.exit()


max_len = 2000
cur_len = 0
cur_summary = []
ret_summary = {}

for (sid,hit) in [i for i in reversed(sorted(h.items(), key = lambda x:x[1]))]:

    successors = set(G.successors(sid))

    next_sent = sent_text[sid]
    cur_len += len(next_sent)

    if(cur_len >= max_len):
        diff = max_len-cur_len
        cur_summary.append(next_sent[0:diff])
        ret_summary[sid.replace("citing-", "")] = next_sent
        break
    else:
        ret_summary[sid.replace("citing-", "")] = next_sent
        cur_summary.append(next_sent)

out_fh = open("YOUR_OUTPUT_DIR/hits/%s.txt" % topic, "w")

for (sid, sent) in ret_summary.items():
    out_fh.write("%s\t%s\n" % (sid, sent))

for sent in cur_summary:
    print sent
