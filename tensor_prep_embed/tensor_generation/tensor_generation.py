"""
build tensor from word
co-occurrences
in a context window
"""

import numpy as np
import argparse
import pickle
from sklearn.externals import joblib
from collections import Counter

""" hyperparameter """
data_folder = "../../data/"
tensor_folder = "../../model/" # model folder
dump_folder = "../../dump/"
prep_window = 3
cxt_window = 6
vocab_size = 50000


def dumpVocab(corpus_name = data_folder+"en_wiki.txt", vocab_fn=None):
    f = open(corpus_name, "r")
    lines = f.readlines()
    f.close()
    vocab_cnt = Counter()
    for line in lines:
        seq = line.strip().lower().split()
        seq_len = len(seq)
        for ind in range(seq_len):
            if (seq[ind] in center_word_list):
                start = max(0, ind-prep_window)
                end = min(seq_len-1, ind+prep_window)
                for k in range(start, end+1):
                    vocab_cnt[seq[k]] += 1
    vocab_freq = vocab_cnt.most_common(vocab_size)
    vocab = [item[0] for item in vocab_freq]
    g = open(vocab_fn, "w")
    print >> g, "\n".join(vocab)
    g.close()
    print "done selecting vocab..."


def getVocabInd(word_seq, vocab_dict):
    """
    turn wiki word sequence into vocab_ind sequence
    """
    # -2~-4: center word
    # -1: out of vocabulary
    vocab_ind_seq = []
    for word in word_seq:
        if (word in center_word_list):
            vocab_ind_seq.append(center_word_list.index(word)-len(center_word_list)-1)
            continue
        try:
            vocab_ind = vocab_dict[word]
        except:
            vocab_ind = -1
        vocab_ind_seq.append(vocab_ind)
    return vocab_ind_seq


def dumpVocabInd(vocab_fn, vocab_dict_fn, vocab_seq_fn, corpus_fn):
    """
    vocab_ind_dict_50000.pickle: {word, vocab_ind}
    vocab_ind_seq_50000.sav: [vocab_ind, vocab_ind] for wiki data sequence
    """
    f = open(corpus_fn, "r")
    text = f.read().lower()
    f.close()
    word_seq = text.split()

    g = open(vocab_fn, "r")
    vocab_list = g.readlines()
    vocab_list = vocab_list[:vocab_size]
    vocab_list = [w.strip() for w in vocab_list]
    vocab_dict = dict()
    for ind in range(len(vocab_list)):
	vocab_dict[vocab_list[ind]] =ind
    g.close()
    
    with open(vocab_dict_fn, "wb") as handle:
        pickle.dump(vocab_dict, handle)
        
    vocab_ind_seq = getVocabInd(word_seq, vocab_dict)
    with open(vocab_seq_fn, "wb") as handle:
        joblib.dump(vocab_ind_seq, handle)
    print "vocab_ind_seq", vocab_ind_seq[:10]
    print "done dumping vocab_ind_dict_"+str(vocab_size)+".pickle"
    print "done dumping vocab_ind_seq_"+str(vocab_size)+".sav"


def countPrepSlab(vocab_dict_fn, vocab_seq_fn, prep_ind):
    # load vocab_ind_dict, vocab_ind_seq
    with open(vocab_dict_fn, "rb") as handle:
        vocab_dict = pickle.load(handle)
    with open(vocab_seq_fn, "rb") as handle:
        vocab_ind_seq = pickle.load(handle)
    vocab_size = len(vocab_dict)
    word_num = len(vocab_ind_seq)
    count_mat = np.zeros((vocab_size, vocab_size))
    # count prep-conditioned matrix
    for pos in range(word_num):
        vocab_ind = vocab_ind_seq[pos]
        if (vocab_ind == prep_ind - len(center_word_list) - 1): #??
            left_pos = max(0, pos-prep_window)
            right_pos = min(pos+prep_window, word_num-1)
            for left_cxt_pos in range(left_pos, right_pos+1):
                left_vocab_ind = vocab_ind_seq[left_cxt_pos]
                if (left_vocab_ind < 0):
                    continue
                for right_cxt_pos in range(left_cxt_pos+1, min(left_cxt_pos+cxt_window+1, right_pos+1)):
                    right_vocab_ind = vocab_ind_seq[right_cxt_pos]
                    if (right_vocab_ind < 0):
                        continue
                    count_mat[left_vocab_ind][right_vocab_ind] += 1
                    count_mat[right_vocab_ind][left_vocab_ind] += 1
    with open(tensor_folder+str(prep_ind)+".sav", "wb") as handle:
        joblib.dump(count_mat, handle)
    print "finish dumping", tensor_folder+str(prep_ind)+".sav"


def countExtraSlab(vocab_dict_fn, vocab_seq_fn, extra_ind):
    # load vocab_dict, vocab_ind_seq
    with open(vocab_dict_fn, "rb") as handle:
        vocab_dict = pickle.load(handle)
    with open(vocab_seq_fn, "rb") as handle:
        vocab_ind_seq = pickle.load(handle)
    vocab_size = len(vocab_dict)
    word_num = len(vocab_ind_seq)
    count_mat = np.zeros((vocab_size, vocab_size))
    for pos in range(word_num):
        if (vocab_ind_seq[pos] < -1):
            left_bound = max(0, pos-prep_window)
            right_bound = min(pos+prep_window, word_num-1)
            for neighbor_pos in range(left_bound, right_bound +1):
                if (vocab_ind_seq[neighbor_pos] >= 0):
                    vocab_ind_seq[neighbor_pos] = -1
    for pos in range(word_num):
        vocab_ind = vocab_ind_seq[pos]
        if (vocab_ind < 0):
            continue
        right_bound = min(pos+cxt_window, word_num-1)
        for cxt_pos in range(pos+1, right_bound+1):
            cxt_vocab_ind = vocab_ind_seq[cxt_pos]
            if (cxt_vocab_ind < 0):
                continue
            count_mat[vocab_ind][cxt_vocab_ind] += 1
            count_mat[cxt_vocab_ind][vocab_ind] += 1
    with open(tensor_folder+str(extra_ind)+".sav", "wb") as handle:
        joblib.dump(count_mat, handle)
    print "done dumping extra slab", tensor_folder+str(extra_ind)+".sav"



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', default="en_wiki.txt", type=str)
    parser.add_argument('--prep', default="in", type=str)
    parser.add_argument('--prepNum', default=49, type=int)
    parser.add_argument('--isCount', default=False, action="store_true")
    args = parser.parse_args()
    corpus_fn = args.fn
    prep = args.prep
    prep_num = args.prepNum
    isCount = args.isCount

    # get prep list
    f = open(data_folder+"prepositions_"+str(prep_num)+".txt", "r")
    lines = f.readlines()
    f.close()
    center_word_list = [line.strip() for line in lines]
    print "# of prepositions:", len(center_word_list)

    vocab_fn = dump_folder + "vocab.txt"
    vocab_dict_fn = dump_folder + "vocab_ind_dict_"+str(vocab_size)+".pickle"
    vocab_seq_fn = dump_folder + "vocab_ind_seq_"+str(vocab_size)+".sav"

    if (not isCount):
        print("Preparing vocab...")
        # dump vocab from corpus
        dumpVocab(corpus_fn, vocab_fn)
        # dump vocab_dict & vocab_seq
        dumpVocabInd(vocab_fn, vocab_dict_fn, vocab_seq_fn, corpus_fn)

    else:
        print("Counting triplets...")
        # count preposition-conditioned slab, prep_ind starts from 0
        for prep in center_word_list:
            prep_ind = center_word_list.index(prep)
            countPrepSlab(vocab_dict_fn, vocab_seq_fn, prep_ind)
        # save an extra slice
        prep_ind = len(center_word_list)
        countExtraSlab(vocab_dict_fn, vocab_seq_fn, prep_ind)                  








        
    
