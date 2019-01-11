"""
binary detection
"""
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from conf_mat import getPrepList, loadConfMat
from util import readData
from als_vocab import ALSVocab
from weighted_vocab import WeightedVocab
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import pickle


def getSim(a, b):
    prod = np.dot(a,b)
    normA = np.sqrt(np.dot(a,a))
    normB = np.sqrt(np.dot(b,b))
    if (normA*normB > 0):
        sim = float(prod) / (normA * normB)
    else:
        sim = 0
    return sim


def scoreCxtPrep(contexts, window, prep_list, isALS, embedding_path):
    """
    detect whether a preposition should be replaced
    """
    if (isALS):
        vec_folder = embedding_path
        word_vec_fn = (vec_folder+"mode1.mat", vec_folder+"mode2.mat") 
        vocab_fn = "../model/vocab.txt"
        selected_ind_fn = vec_folder+"mode1.map"
        isAvg = True
        z_vec_fn = vec_folder+"mode3.mat"
        z_vocab_fn = "../data/prepositions_49.txt" 
        vocab = ALSVocab(word_vec_fn, vocab_fn, selected_ind_fn, isAvg,\
                      z_vec_fn, z_vocab_fn)
        cxt_vecs = vocab.getContextVecs(contexts, window)
        prep_vecs = vocab.getPrepVecs(prep_list)
    else:
        prep_num = 49
        vocab = WeightedVocab(embedding_path, prep_num)
        cxt_vecs = vocab.getContextVecs(contexts, window)
        prep_vecs = vocab.getPrepVecs()
        
    # modify from here
    side_prep_scores = [] # max(left, right)
    cxt_prep_scores = [] # cxt
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
    return cxt_prep_scores, side_prep_scores


def binarySelectFeatures(contexts, window, orig_prep_list, isALS, embedding_path):
    prep_list = getPrepList()
    prep_total_num = len(prep_list)
    cxt_prep_scores, side_prep_scores = scoreCxtPrep(contexts, window, prep_list, isALS, embedding_path)
    # extract features: scores, ranking, keep probability
    orig_prep_pos = [prep_list.index(orig_prep) for orig_prep in orig_prep_list]
    conf_mat = loadConfMat()
    score_list = [] # scores
    rank_list = [] # rank
    prob_list = [] # prob
    prep_indicator_list = [] # one-hot prep indicator
    rankOne_counter = 0
    for ind in range(len(contexts)):
        #orig_prep = orig_prep_list[ind]
        prep_pos = orig_prep_pos[ind]
        score = cxt_prep_scores[ind][prep_pos]
        if (score > 1.0):
	   score = 1.0
	   print "error: score > 1.0 !!"
        rank = 1
        for s in cxt_prep_scores[ind]:
            if (s > score):
                rank += 1
        if (rank == 1):
            rankOne_counter += 1
        prob = conf_mat[prep_pos][prep_pos]
        score_list.append(score)
        rank_list.append(rank)
        prob_list.append(prob)
    feature = [score_list, rank_list, prob_list]
    feature = np.transpose(feature)
    return feature


def TreeClf(train_feature, train_label, test_feature):
    print "DecisionTree..."
    clf = DecisionTreeClassifier(class_weight={1:5, 0:1})
    clf.fit(train_feature, train_label)
    test_label = clf.predict(test_feature)
    return test_label


def binaryDecision(train_fn, test_fn, window, isALS, embedding_path, isSave=False):
    # train data, sent (<b> </b> marked preposition)
    # train label: 1 - should be corrected, 0 - not corrected
    orig_prep_list, sent_list, correct_prep_list = readData(train_fn)
    train_label = [int(orig_prep_list[ind]!=correct_prep_list[ind]) for ind in range(len(sent_list))]
    train_feature = binarySelectFeatures(sent_list, window, orig_prep_list, isALS, embedding_path)

    # test data
    orig_prep_list, sent_list, correct_prep_list = readData(test_fn)
    gold_label = [int(orig_prep_list[ind]!=correct_prep_list[ind]) for ind in range(len(sent_list))]
    test_feature = binarySelectFeatures(sent_list, window, orig_prep_list, isALS, embedding_path)

    # DecisionTree classifier
    test_label = TreeClf(train_feature, train_label, test_feature)

    if (isSave):
        with open("model/binary_decision", "wb") as handle:
	    pickle.dump(test_label, handle)
        print "done dumping binary labels..."

    # binary evaluation
    precision = precision_score(gold_label, test_label)
    recall = recall_score(gold_label, test_label)
    fscore = f1_score(gold_label, test_label)
    print "selected instances:", np.sum(test_label)
    print "prec: %f, recall: %f, fscore: %f" % (precision, recall, fscore)


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
    isSave = True
    
    file_dir =  '../data/cikm2014/'
    train_fn = file_dir + "CLC_FCE.txt"
    test_fn = file_dir + test
    binaryDecision(train_fn, test_fn, window, isALS, embedding_path, isSave)
    
