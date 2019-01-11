import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class WeightedVocab:
    def __init__(self, word_vec_fn, prep_num=49, isFunctional=True):
        self.word_vec_fn = word_vec_fn
        self.word_vectors = self.getVectorFromFile()
        self.prep_num = prep_num
        self.funcWords = self.readFuncWords(isFunctional)

        
    def getVectorFromFile(self):
        word_vectors = KeyedVectors.load_word2vec_format(self.word_vec_fn, binary=False)
        return word_vectors
    
        
    def readFuncWords(self, isFunctional=True):
            funcWords = set()
            if isFunctional:
                    f = open("../data/funcWords.txt", "r")
            for line in f.readlines():
                    funcWords.add(line.strip())
            return funcWords


    def getContextVecs(self, contexts, window):
        """
        stack context embedding from word2vec vectors
        """
        context_vecs = []
        dim = len(self.word_vectors['the'])
        for context in contexts:
            # get left and right context vectors
            context = context.lower().strip().split()
            try:
                start = context.index('<b>')
                end = context.index('<b>')
            except:
                print "error context:", context
                continue
            left_vecs = []
            right_vecs = []
            idx = start
            count = 0
            while idx > 0 and count < window:
                idx -= 1
                try:
                    left_vecs.append(word_vectors[context[idx]])
                    count += 1
                except:
                    pass
            idx = end
            count = 0
            while idx < len(context) -1 and count < window:
                idx += 1
                try:
                    right_vesc.append(word_vectors[context[idx]])
                    count += 1
                except:
                    pass
            if (len(left_vecs) < 1):
                left_vec = [[0]*dim]
            if (len(right_vecs) < 1):
                right_vec = [[0]*dim]
            context_vecs.append([left_vecs[:], right_vecs[:]])
        return context_vecs


    def getPrepVecs(self):
        prep_vecs = [self.word_vectors[str(prep_ind)] for prep_ind in range(self.prep_num)]
        return prep_vecs
        
        
