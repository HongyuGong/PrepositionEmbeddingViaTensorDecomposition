"""
vocab utility functions
"""
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


def normalizeMatrix(mat):
	matNorm = np.zeros(mat.shape)
	d = (np.sum(mat ** 2, 1) ** (0.5))
	matNorm = (mat.T / d).T

	return matNorm

class ALSVocab:
    
    def __init__(self, word_vec_fn, vocab_fn, selected_ind_fn, isAvg, \
                 z_vec_fn, z_vocab_fn, isFunctional=True, save_path=None):
        """
        self.vecs, self.vocab
        self.num, self.dim
        self.zVecs, self.zVocab
        self.zNum
        self.save_path
        """
        # load word vector
        if (isAvg):
            self.vecs, self.num, self.dim = self.getAvgVectorFromFile(word_vec_fn[0], word_vec_fn[1])
        else:
            self.vecs, self.num, self.dim = self.getVectorFromFile(word_vec_fn)
        # load word vocab
        self.vocab, self.vocabIndex = self.getVocabFromFile(vocab_fn, selected_ind_fn)
        # load extra dimension vector
        self.zVecs, self.zNum, _ = self.getVectorFromFile(z_vec_fn)
        self.zVocab, self.zVocabIndex  = self.getVocabFromFile(z_vocab_fn, None)
        # functiaonl words
        self.funcWords = self.readFuncWords(isFunctional)
        # the output path of embeddings
        self.save_path = save_path
        

    def getVectorFromFile(self, word_vec_fn):
        """
        read vector from mode*.mat
        """
        f = open(word_vec_fn, "r")
        lines = f.readlines()
        f.close()
        word_vecs = [line.strip().split() for line in lines]
        word_vecs = np.array(word_vecs).astype(np.float)
        word_num, dim = word_vecs.shape
        print "vector size:", word_num, "vector dim", dim
        return word_vecs, word_num, dim

    def getAvgVectorFromFile(self, fn1, fn2):
        """
        get average of word embeddings
        from mode1.txt and mode2.txt
        """
        vec1, num, dim = self.getVectorFromFile(fn1)
        vec2, num, dim = self.getVectorFromFile(fn2)
        avg_vec = 0.5 * (vec1 + vec2)
        return avg_vec, num, dim

    def getNonZeroWord(self, selected_ind_fn, vocab_fn):
        f = open(selected_ind_fn, "r")
        ind_str_seq = f.readlines()
        f.close()
        ind_seq = [int(ind_str.strip()) for ind_str in ind_str_seq]

        g = open(vocab_fn, "r")
        vocab_seq = g.readlines()
        g.close()
        vocab_seq = [vocab_seq[ind].strip() for ind in ind_seq]
        print "vocab size", len(vocab_seq)
        
        return vocab_seq

    def getVocabFromFile(self, vocab_fn, ind_fn):
        """
        get a list of vocabulary words,
        ind_fn: indices of non-zero words
        ind_fn==None: all words are selected
        """
        if (ind_fn == None):
            f = open(vocab_fn, "r")
            lines = f.readlines()
            vocab = [line.strip() for line in lines]
        else:
            vocab = self.getNonZeroWord(ind_fn, vocab_fn)
            
        vocabIndex = dict()
        idx = 0
        for w in vocab:
                vocabIndex[w] = idx
                idx += 1
        return vocab, vocabIndex

    def readFuncWords(self, isFunctional=True):
            funcWords = set()
            if isFunctional:
                    f = open("../data/funcWords.txt", "r")
            for line in f.readlines():
                    funcWords.add(line.strip())
            return funcWords

    """
    for preposition selection
    """

    def getDoubleContextIdList(self, contexts, window):
            contextList = list()
            for context in contexts:
                context = context.lower().strip().split()
                try:
                    start = context.index('<b>')
                    end = context.index('<b>')
                except:
                    print context
                    continue
                left_contextId = list()
                right_contextId = list()
                idx = start
                count = 0
                while idx > 0 and count < window:
                        idx -= 1
                        if context[idx] in self.funcWords:
                                continue
                        try:
                                left_contextId.append(self.vocabIndex[context[idx]])
                                count += 1
                        except:
                                pass
                #prepIdList = [vocabIndex[prep+"."+str(ind)] for ind in range(3)]
                #prepId = vocabIndex[prep]
                idx = end
                count = 0
                while idx < len(context) - 1 and count < window:
                        idx += 1
                        if context[idx] in self.funcWords:
                                continue
                        try:
                                right_contextId.append(self.vocabIndex[context[idx]])
                                count += 1
                        except:
                                pass
                if (len(left_contextId) < 1 or len(right_contextId) < 1):
                    contextList.append([None, None])
                else:
                    contextList.append([left_contextId, right_contextId])

            print "# contextIdList", len(contextList)
            return contextList


    def getContextVecs(self, contexts, window):
            contextIdxList = self.getDoubleContextIdList(contexts, window)
            contextVecs = []
            for leftIdxList, rightIdxList in contextIdxList:
                    if (leftIdxList == None):
                            leftVec = [[0] * self.dim]
                    else:
                            leftVec = self.vecs[np.array(leftIdxList)]
                    if (rightIdxList == None):
                            rightVec = [[0] * self.dim]
                    else:
                            rightVec = self.vecs[np.array(rightIdxList)]
                    contextVecs.append([leftVec, rightVec])
            return contextVecs       

    def getPrepVecs(self, prep_list):
            prep_ind = [self.zVocab.index(prep) for prep in prep_list]
            prep_vecs = self.zVecs[np.array(prep_ind)]
            print "# of prep_vecs", len(prep_vecs)
            return prep_vecs
            




