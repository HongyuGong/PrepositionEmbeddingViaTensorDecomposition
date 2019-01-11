"""
classification
"""
from sklearn.externals import joblib
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
from sklearn import tree
from sklearn.svm import SVC


def flattenData(features, labels):
    """
    return flattened data as model inputs
    """
    features_flatten = []
    labels_flatten = []
    heads_num = []
    # flatten test features
    for ind in range(len(features)):
        feat = features[ind]
        lab = labels[ind]
        features_flatten = features_flatten + feat
        labels_flatten = labels_flatten + lab
        heads_num.append(len(feat))
    #print "flatten feature:", features_flatten[:2]
    #print "label shape:", np.shape(np.array(labels_flatten)), "should be: (?, )"
    return features_flatten, labels_flatten, heads_num

    
def stackData(prob_flatten, heads_num):
    """
    transform flatten data to stacked data per instance
    return pred_prob
    """
    pred_prob = []
    start_ind = 0
    inst_ind = 0
    while (start_ind < len(prob_flatten)):
        head_count = heads_num[inst_ind]
        pred_prob.append(prob_flatten[start_ind: start_ind+head_count])
        start_ind = start_ind+head_count
        inst_ind += 1
    return pred_prob


def trainTest(train_dump_path, test_dump_path):
    # load stacked feature+label
    train_features, train_labels = joblib.load(train_dump_path)
    test_features, test_labels = joblib.load(test_dump_path)
    print "example stacked labels:", train_labels[:6]
    # gold_test_labels: chosen head for each sentence
    gold_test_labels = [np.argmax(row) for row in test_labels]
    # flatten train data
    train_features_flatten, train_labels_flatten, train_heads_num = flattenData(train_features, train_labels)
    # flatten test data
    test_features_flatten, test_labels_flatten, test_heads_num = flattenData(test_features, test_labels)
    
    # train-test
    for seed in range(0,1):
        print "round:", seed
        # train with MLP classifier
        clf = MLPClassifier(hidden_layer_sizes = (50, 5, ), solver="adam", activation="relu", alpha=1e-5, random_state=seed)
        clf.fit(train_features_flatten, train_labels_flatten)

        # train prob
        pred_train_prob_flatten = clf.predict_proba(train_features_flatten)
        # class-1 probability
        pred_train_prob_flatten = pred_train_prob_flatten[:, -1]
        # stack train prob
        pred_train_prob = stackData(pred_train_prob_flatten, train_heads_num)
        # save stacked train_prob
        joblib.dump(np.array(pred_train_prob), "../data/pred_train_prob_"+str(seed)+".pkl")
        print "saving predicted train scores..."
        
        # test prob
        pred_test_prob_flatten = clf.predict_proba(test_features_flatten)
        # class-1 probability
        pred_test_prob_flatten = pred_test_prob_flatten[:, -1]
        # stack test prob
        pred_test_prob = stackData(pred_test_prob_flatten, test_heads_num)
        # save stacked test_prob
        joblib.dump(np.array(pred_test_prob), "../data/pred_test_prob_"+str(seed)+".pkl")
        print "saving predicted test scores..."
        
        print "example pred_test_prob:", pred_test_prob[:2] 
        # test label
        pred_test_labels = [] # label for each sentence
        for test_ind in range(len(pred_test_prob)):
            probs = pred_test_prob[test_ind]
            selected_head_pos = np.argmax(probs)
            pred_test_labels.append(selected_head_pos)

        # eval result
        evalRes(gold_test_labels, pred_test_labels)

        # log result: heads prep children | scores | correct
        # log result
        data_folder = "../pp-attachment/data/pp-data-english/"
        test_prefix = "wsj.23.txt.dep.pp."
        cand_fn  = data_folder+test_prefix+"heads.words"
        prep_fn = data_folder+test_prefix+"preps.words"
        child_fn = data_folder+test_prefix+"children.words"
        logRes(gold_test_labels, pred_test_labels, pred_test_prob, cand_fn, prep_fn, child_fn)
        


def evalRes(gold_labels, pred_labels):

    right_count = 0
    total_count = len(gold_labels)
    for test_ind in range(total_count):
        right_count += int(gold_labels[test_ind] == pred_labels[test_ind])
    acc = 1.0 * right_count / total_count
    print "total count:", total_count, "acc:", acc


def logRes(gold_labels, pred_labels, pred_test_prob, cand_fn, prep_fn, child_fn):
    """
    log wrong instances: sent, correct head (score), wrong head (score)
    gold_labels: [2, 3, 1, ...]
    pred_labels: [2, 1, 1, ...]
    pred_test_prob: [[0.1, 0.4, 0.5], [0.0, 0.6, 0.4], [0.2, 0.7, 0.1], ...]
    cand_fn: [["eat", "food",], ["play", "the", "piano"], ...]
    child_fn: 
    log format:     head1(score1) head2(score2) prep children correctHead
    """
    f = open(cand_fn, "r")
    cand_list = f.readlines()
    cand_list = [cand.strip().split() for cand in cand_list]
    f.close()

    f = open(prep_fn, "r")
    prep_list = f.readlines()
    prep_list = [prep.strip() for prep in prep_list]
    f.close()

    f = open(child_fn, "r")
    child_list = f.readlines()
    child_list = [child.strip() for child in child_list]
    f.close()

    g = open("log.txt", "w")
    for inst_ind in range(len(gold_labels)):
        gold_label_ind = gold_labels[inst_ind]
        pred_label_ind = pred_labels[inst_ind]
        if (gold_label_ind != pred_label_ind):
            cand_num = len(cand_list[inst_ind])
            word_list = []
            # head (score), head (score)
            for cand_ind in range(cand_num):
                word_list.append(cand_list[inst_ind][cand_ind]+\
                                 " ("+str(pred_test_prob[inst_ind][cand_ind])+")")
            # preposition
            word_list.append(prep_list[inst_ind])
            # child
            word_list.append(child_list[inst_ind])
            # correct head
            word_list.append(cand_list[inst_ind][gold_label_ind])
            print >> g, "\t".join(word_list)
    g.close()
    print "done logging results..."

    

if __name__=="__main__":

    train_dump_path = "../data/train_dump"
    test_dump_path = "../data/test_dump"
    trainTest(train_dump_path, test_dump_path)
    
