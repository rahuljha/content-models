#!/usr/bin/python

class TopicSumWeave:

    def __init__(self, vectors, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        # assign topic numbers to each docset and document 
        self.topic

    def initializeGibbs(self):
        pass

    def runGibbs(self):
        pass



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
    beta = [1.0, 0.1, 1.0]

    tsObj = TopicSumWeave(vectors, alpha, beta)
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


