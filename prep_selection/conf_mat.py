"""
get confusion matrix of prepositions
"""
import nltk
import numpy as np
import pickle

cikm_folder = "../data/cikm2014/"
prep_file = "../data/prepositions_49.txt"


def getPrepList():
    f = open(prep_file, "r")
    prep_list = []
    for line in f:
        prep = line.strip()
        prep_list.append(prep)
    f.close()
    return prep_list


def dumpConfMat(fn, verbose=True):
    """
    fn: training file
    return: prep list [prep]
    probability matrix {row: original prep, col: corrected prep}
    * normalize each row
    """
    prep_list = getPrepList()
    prep_num = len(prep_list)
    print "# of prep", prep_num
    conf_mat = np.zeros((prep_num, prep_num))
    prep_instances = 0

    f = open(fn, "r")
    for line in f:
        line = line.decode('utf8').strip().lower()
        seq = nltk.word_tokenize(line)
        for ind in range(len(seq)):
            word = seq[ind]
	    if (word not in prep_list and "*/" not in word):
		continue
            # prep without correction
            if (word in prep_list):
                original_prep = word
                correct_prep = word
            elif ("*/" in word):
                prep_pair = word.split("*/")
                original_prep, correct_prep = prep_pair
		if (original_prep not in prep_list):
		    print "prep wrong: ", seq
		    print "prep_pair", prep_pair
            # update confusion matrix
	    prep_instances += 1
            original_ind = prep_list.index(original_prep)
            correct_ind = prep_list.index(correct_prep)
            conf_mat[original_ind][correct_ind] += 1
    f.close()

    # normalize confusion matrix
    row_sum = np.sum(conf_mat, axis=1)
    
    if (verbose):
        # for debugging
        print "conf_mat:", conf_mat[0]
        print "conf_mat:", conf_mat[-1]
        print "# of prep_instances:", prep_instances
        # for debugging
        print "row_sum:", row_sum
        print "sum of row_sum:", np.sum(row_sum)

    for row_ind in range(prep_num):
        if (row_sum[row_ind] == 0):
            conf_mat[row_ind] = 0.01
            conf_mat[row_ind][row_ind] = 1
    row_sum = np.sum(conf_mat, axis=1)
    norm_conf_mat = 1.0 * conf_mat / row_sum.reshape((prep_num, 1))

    # dump prep_list and conf_mat
    with open('model/conf_score.pickle', 'wb') as handle:
        pickle.dump(norm_conf_mat, handle)
    print "finish dumping confusion score"


def loadConfMat(dump_file = 'model/conf_score.pickle'):
    with open(dump_file, 'rb') as handle:
        b = pickle.load(handle)
    conf_mat = b
    return conf_mat


if __name__=="__main__":
    train_fn = cikm_folder+"CLC_FCE.txt"
    dumpConfMat(train_fn)
