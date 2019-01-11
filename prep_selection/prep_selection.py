"""
label selection
"""

from sklearn import svm
import numpy as np
from conf_mat import getPrepList, loadConfMat
from util import readData
from als_vocab import ALSVocab
from weighted_vocab import WeightedVocab
import argparse
from binary_detection import scoreCxtPrep, getSim
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
from sklearn.neural_network import MLPClassifier
import os


def scoreCxtPrepVarWin(contexts, win_size_list, prep_list, isALS, embedding_path):
    """
    score context-prep with different window sizes
    designed for multi-selection part
    return: cxt_prep_scores_win (win_size, test_instances, prep_num)
    """
    if (isALS):
        vec_folder = "tensor_prepNum=49/" 
        word_vec_fn = (vec_folder+"mode1.mat", vec_folder+"mode2.mat") 
        vocab_fn = "dump/vocab.txt"
        selected_ind_fn = vec_folder+"mode1.map"
        isAvg = True
        z_vec_fn = vec_folder+"mode3.mat"
        z_vocab_fn = "data/prepositions_49.txt" 
        vocab = Vocab(word_vec_fn, vocab_fn, selected_ind_fn, isAvg,\
                      z_vec_fn, z_vocab_fn)
        prep_vecs = vocab.getPrepVecs(prep_list)
    else:
        prep_num = 49
        vocab = WeightedVocab(embedding_path, prep_num)
        prep_vecs = vocab.getPrepVecs()       

    cxt_prep_scores_win = [] # cxt_prep_scores with different window sizes
    side_prep_scores_win = [] # side_prep_scores with different window sizes
    for window in win_size_list:
        cxt_vecs = vocab.getContextVecs(contexts, window)
        cxt_prep_scores = [] # cxt
        side_prep_scores = [] # max(left, right)
        for cxt_vec in cxt_vecs:
            left_vecs, right_vecs = cxt_vec
            left_vec = np.sum(left_vecs, axis=0)
            right_vec = np.sum(right_vecs, axis=0)
            cxt_vec = left_vec + right_vec
            # score each context with all prepositions
            score_list = []
            side_score_list = []
            for prep_vec in prep_vecs:
                s = getSim(prep_vec, cxt_vec)
                score_list.append(s)
                s = max(getSim(prep_vec, left_vec), getSim(prep_vec, right_vec))
                side_score_list.append(s)
            cxt_prep_scores.append(score_list[:])
            side_prep_scores.append(side_score_list[:])
        cxt_prep_scores_win.append(cxt_prep_scores[:])
        side_prep_scores_win.append(side_prep_scores[:])
    cxt_prep_scores_win = np.array(cxt_prep_scores_win)
    side_prep_scores_win = np.array(side_prep_scores_win)
    print "cxt_prep_scores_win shape", cxt_prep_scores_win.shape,
    print "side_prep_scores_win shape", side_prep_scores_win.shape
    return cxt_prep_scores_win, side_prep_scores_win
    

def multiSelectTrainFeatures(prep_list, raw_features, orig_prep_list, correct_prep_list):
    """
    features: for each replaced sentence, cand_score, cand_rank, orig->cand prob
    """
    prep_num = len(prep_list)
    cxt_prep_scores_win, side_prep_scores_win, cxt_prep_ranks, conf_mat = raw_features
    features = []
    labels = []

    # ind: cxt_ind
    for ind in range(len(cxt_prep_scores_win[0])):
        prep_scores = cxt_prep_scores_win[:, ind, :] # prep_scores shape: (win_sizes, prep_num)
        s_prep_scores = side_prep_scores_win[:, ind, :] # s_prep_scores shape:  (win_sizes, prep_num)
        prep_ranks = cxt_prep_ranks[ind]
        orig_prep_ind = prep_list.index(orig_prep_list[ind])
        gold_prep_ind = prep_list.index(correct_prep_list[ind])
        if (orig_prep_ind == gold_prep_ind):
            continue

        for test_prep_ind in range(prep_num):
            if (test_prep_ind == orig_prep_ind):
                continue
            orig_to_test_prob = conf_mat[gold_prep_ind][test_prep_ind]
            test_prep_score = prep_scores[:, test_prep_ind] # (win_sizes, )
            test_side_score = s_prep_scores[:, test_prep_ind] # (win_sizes, )
            test_prep_rank = prep_ranks[test_prep_ind]
            # preposition one-hot vector
            one_hot_vec = [0] * prep_num
            one_hot_vec[test_prep_ind] = 1

            # feature o. rank
            feature = list(test_prep_score[:]) + list(test_side_score[:] )+ [orig_to_test_prob] + \
                      one_hot_vec[:]
            label = int(test_prep_ind == gold_prep_ind)
            features.append(feature[:])
            labels.append(label)
    features = np.array(features)
    labels = np.array(labels)
    print "train label # of 1's:", np.sum(labels)
    print "features:", features.shape, "labels:", labels.shape
    return features, labels


def multiSelectTestFeatures(prep_list, raw_features, orig_prep_list):
    prep_num = len(prep_list)
    cxt_prep_scores_win, side_prep_scores_win, cxt_prep_ranks, conf_mat = raw_features
    features = []
    candidate_inds = []
    # corpus bigram matching: (test_sent_num, prep_num)

    for ind in range(len(cxt_prep_scores_win[0])):
        cand = []
        prep_scores = cxt_prep_scores_win[:, ind, :] # (win_sizes, prep_num)
        s_prep_scores = side_prep_scores_win[:, ind, :] # (win_sizes, prep_num)
        prep_ranks = cxt_prep_ranks[ind]
        orig_prep_ind = prep_list.index(orig_prep_list[ind])

        for test_prep_ind in range(prep_num):
            if (test_prep_ind == orig_prep_ind):
                continue
            cand.append(test_prep_ind)
            test_prep_score = prep_scores[:, test_prep_ind] # (win_sizes, )
            test_side_score = s_prep_scores[:, test_prep_ind] # # (win_sizes, )
            test_prep_rank = prep_ranks[test_prep_ind]
            orig_to_test_prob = conf_mat[orig_prep_ind][test_prep_ind]
            # preposition one-hot vector
            one_hot_vec = [0] * prep_num
            one_hot_vec[test_prep_ind] = 1

            # feature o. rank
            feature = list(test_prep_score[:]) + list(test_side_score[:]) + [orig_to_test_prob] + \
                      one_hot_vec[:]
            features.append(feature[:])
        candidate_inds.append(cand[:])
    features = np.array(features)
    candidate_inds = np.array(candidate_inds)
    print "test feature shape:", features.shape, "candidate inds shape:", candidate_inds.shape
    return features, candidate_inds


def getRawMultiFeatures(contexts, win_size_list, isALS, embedding_path):
    """
    raw features: scores, ranks, confusion probability
    """
    prep_list = getPrepList()
    prep_num = len(prep_list)
    # get scores
    cxt_prep_scores_win, side_prep_scores_win = scoreCxtPrepVarWin(contexts, win_size_list, \
                                                                   prep_list, isALS, embedding_path)
     
    #cxt_prep_scores = np.abs(cxt_prep_scores)
    #print "processing raw data, absolute value..."

    # rank according to cxt_prep_score_win=1
    cxt_prep_ranks = []
    for prep_scores in cxt_prep_scores_win[0]:
        prep_ranks = [1+ np.sum(np.array(prep_scores > prep_scores[prep_ind], "int")) for prep_ind in range(len(prep_scores))]
        cxt_prep_ranks.append(prep_ranks[:])
    print "prep_ranks:", prep_ranks
    # load confusion matrix
    conf_mat = loadConfMat()
    return (cxt_prep_scores_win, side_prep_scores_win, cxt_prep_ranks, conf_mat)

    

def dumpMultiFeatures(train_fn, test_fn, window, win_size_list, isALS, embedding_path):

    train_orig_prep_list, train_sent_list, train_correct_prep_list = readData(train_fn)
    test_orig_prep_list, test_sent_list, test_correct_prep_list = readData(test_fn)

    # binary decision
    prep_list = getPrepList()
    prep_num = len(prep_list)

    with open("model/binary_decision", "rb") as handle:
        binary_test_label = pickle.load(handle)

    total_valid_corrections = np.sum([int(test_orig_prep_list[ind] != test_correct_prep_list[ind]) for ind in range(len(test_correct_prep_list))])

    raw_train_features =  getRawMultiFeatures(train_sent_list, win_size_list, isALS, embedding_path)
    raw_test_features = getRawMultiFeatures(test_sent_list, win_size_list, isALS, embedding_path)
    # select test examples to be corrected
    selected_test_inds = [ind for ind in range(len(binary_test_label)) if binary_test_label[ind] == 1]
    print "selected test instances:", len(selected_test_inds)
    test_prep_scores_win, test_side_prep_scores_win, test_prep_ranks, test_conf_mat = raw_test_features
    test_prep_scores_win = test_prep_scores_win[:, selected_test_inds, :]
    test_side_prep_scores_win = test_side_prep_scores_win[:, selected_test_inds, :]
    test_prep_ranks = [test_prep_ranks[ind] for ind in selected_test_inds]
    raw_test_features = (test_prep_scores_win, test_side_prep_scores_win, test_prep_ranks, test_conf_mat)
    test_orig_prep_list = [test_orig_prep_list[ind] for ind in selected_test_inds]
    test_correct_prep_list = [test_correct_prep_list[ind] for ind in selected_test_inds]

    # preposition selection from multiple candidates
    print "using corpus bigram count as feature..."
    train_features, train_labels = multiSelectTrainFeatures(prep_list, raw_train_features, train_orig_prep_list, train_correct_prep_list)
    test_features, test_candidate_prep_inds = multiSelectTestFeatures(prep_list, raw_test_features, test_orig_prep_list)


    # dump features: train_features, train_labels, test_features, test_candidate_prep_inds
    model_folder = "model/"
    with open(model_folder+"train.feature_label_win="+str(len(win_size_list)), "wb") as handle:
        pickle.dump((train_features, train_labels), handle)
    # dump test data
    test_sent_list = [test_sent_list[ind] for ind in selected_test_inds]
    with open(model_folder+"test.data_win="+str(len(win_size_list)), "wb") as handle:
        pickle.dump((test_sent_list, test_orig_prep_list, test_correct_prep_list), handle)
    with open(model_folder+"test.feature_label_win="+str(len(win_size_list)), "wb") as handle:
        pickle.dump((test_features, test_candidate_prep_inds), handle)
    print "done dumping train and test data..."
    return total_valid_corrections


def mlpClf(train_features, train_labels, test_features, seed):
    print "train_features shape", train_features.shape, "train_labels shape", train_labels.shape
    print "test_features shape", test_features.shape
    print "MLP classifier..."
    clf = MLPClassifier(hidden_layer_sizes = (50,10, ), solver="sgd", alpha=1e-5, random_state=seed)
    clf.fit(train_features, train_labels)
    test_labels = clf.predict_proba(test_features)
    return test_labels


def clfEval(total_valid_corrections):
    # load data
    model_folder = "model/"
    with open(model_folder+"train.feature_label_win="+str(len(win_size_list)), "rb") as handle:
        train_features, train_labels = pickle.load(handle)
    with open(model_folder+"test.data_win="+str(len(win_size_list)), "rb") as handle:
        test_sent_list, test_orig_prep_list, test_correct_prep_list = pickle.load(handle)
    with open(model_folder+"test.feature_label_win="+str(len(win_size_list)), "rb") as handle:
        test_features, test_candidate_prep_inds = pickle.load(handle)
    
    prep_list = getPrepList()
    prep_num = len(prep_list)
    test_num = len(test_orig_prep_list)

    test_probs = mlpClf(train_features, train_labels, test_features, 23)
    test_probs = test_probs[:, -1]
    test_probs = test_probs.reshape((test_num, (prep_num-1)))
    max_score_prep_inds = np.argmax(test_probs, axis=1)
    selected_prep_inds = [test_candidate_prep_inds[test_ind][max_score_prep_inds[test_ind]] for test_ind in range(test_num)]
    selected_preps = [prep_list[ind] for ind in selected_prep_inds]

    # evaluation
    valid_suggested_corrections = np.sum([int(selected_preps[ind] == test_correct_prep_list[ind]) for ind in range(test_num) \
                                          if test_correct_prep_list[ind] != test_orig_prep_list[ind]])
    total_suggested_corrections = test_num
    print "valid_suggested_corrections", valid_suggested_corrections
    print "total_suggested_corrections", total_suggested_corrections
    print "total_valiad_corrections", total_valid_corrections
    prec = 1.0 * valid_suggested_corrections / total_suggested_corrections
    recall = 1.0 * valid_suggested_corrections / total_valid_corrections
    fscore = 2 * prec * recall / (prec + recall)
    print "prec: %f, recall: %f, fscore: %f" % (prec, recall, fscore)



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default=1, type=int)
    parser.add_argument('--test', default='CoNLL.txt', type=str)
    parser.add_argument('--isALS', default=False, action="store_true")
    parser.add_argument('--embedding', default='../model/ALS_tensor_vectors/', type=str)
    
    args = parser.parse_args()
    window = args.window
    test = args.test
    isALS = args.isALS
    embedding_path = args.embedding
    file_dir = '../data/cikm2014/'

    win_size_list =  range(1, window+1)
    print "win_sizes:", win_size_list
    train_fn = file_dir + "CLC_FCE.txt"
    test_fn = file_dir + test

    total_valid_corrections = dumpMultiFeatures(train_fn, test_fn, window, win_size_list, isALS, embedding_path)
    clfEval(total_valid_corrections)
    
    

