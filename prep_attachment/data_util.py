"""
read preposition attachment data
"""
from vocab import Vocab
import numpy as np
from sklearn.externals import joblib


def readChildren(fn):
    """
    fn: children.words
    children word of a preposition
    """
    f = open(fn, "r")
    children_words = f.readlines()
    f.close()
    children_words = [word.strip() for word in children_words]
    return children_words


def readHeadCandidates(fn):
    """
    fn: heads.words
    head candidates for a preposition
    """
    f = open(fn, "r")
    str_seq = f.readlines()
    f.close()
    # candidate words
    head_candidates = [s.strip().split() for s in str_seq]
    # closeness betweent cand and prep
    cand_distance = []
    for cands in head_candidates:
        #dist = 1.0 * np.arange(len(cands)) / len(cands)
	#dist = len(cands) - 1 - np.arange(len(cands))
	dist = 1.0 * (len(cands) - np.arange(len(cands))) / len(cands)
	##dist = np.arange(len(cands))
        cand_distance.append(dist[:])
    return head_candidates, cand_distance

    
def readGoldHead(head_candidates, fn):
    """
    fn: pp.labels
    gold head for the preposition
    """
    f = open(fn, "r")
    head_inds = f.readlines()
    f.close()
    # head_inds: [4, 4, 3]
    head_inds = [int(ind.strip()) for ind in head_inds]
    # head words
    head_words = [head_candidates[inst_ind][head_inds[inst_ind]-1] for inst_ind in range(len(head_inds))]
    #return head_inds, head_words
    return head_words

def readWordPOS(fn):
    """
    part-of-speech of head candidates for each preposition
    1: verb, -1: noun
    """
    f = open(fn, "r")
    str_list = f.readlines()
    f.close()
    pos_list = []
    for s in str_list:
        seq = s.strip().split()
        num_seq = [int(num) for num in seq]
        pos_list.append(num_seq[:])
    return pos_list

def readTags(fn):
    # count all POS taggs
    f = open(fn, "r")
    lines = f.readlines()
    f.close()
    seq = [line.strip().split() for line in lines]
    tag_set = []
    for row in seq:
        tag_set.extend(row[:])
    tag_set = list(set(tag_set))
    print "# of part-of-speech tags:", len(tag_set)
    return tag_set


def readNextPOS(fn, tag_set):
    """
    read the pos tagging of the next word
    """
    f = open(fn, "r")
    lines = f.readlines()
    f.close()
    seq = [line.strip().split() for line in lines]
    # index the tag: one-hot representation
    test_num = len(seq)
    tag_num = len(tag_set) + 1 # including special token for the last word
    next_pos_list = []
    for ind in range(test_num):
        pos_vecs = []
        tag_row = seq[ind]
        for cand_ind in range(len(tag_row)):
            tag_ind = tag_set.index(tag_row[cand_ind])
            vec = [0] * tag_num
            vec[tag_ind] = 1
            pos_vecs.append(vec[:])
        vec = [0] * tag_num
        vec[-1] = 1
        pos_vecs.append(vec[:])
        next_pos_list.append(pos_vecs[:])
    return next_pos_list
    


def readPrep(fn):
    """
    preposition word
    """
    f = open(fn, "r")
    prep_words = f.readlines()
    f.close()
    prep_words = [word.strip() for word in prep_words]
    prep_set = list(set(prep_words))
    return prep_words, prep_set

def getL3Norm(vec):
    return np.power(np.sum(np.power(np.abs(vec), 3)), 1.0/3)


def getSim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if (norm_a == 0 or norm_b == 0):
        sim = 0
    else:
        sim = 1.0 * np.dot(a, b) / (norm_a * norm_b)
    return sim
        

def getTripleSim(head_vec, prep_vec, child_vec):
    head_norm = getL3Norm(head_vec)
    prep_norm = getL3Norm(prep_vec)
    child_norm = getL3Norm(child_vec)
    if (head_norm == 0 or prep_norm == 0 or child_norm == 0):
        sim = 0
    else:
        prod = np.multiply(head_vec, prep_vec)
        prod = np.multiply(prod, child_vec)
        sim = 1.0 * np.sum(prod) / (head_norm*prep_norm*child_norm)
    return sim
    

def scoreHead(vocab, head_candidates, prep_words, children, prep_set, prep_vecs):
    """
    input: [h1, h2, ..., ], prep, child
    return [s1, s2, ..., ]
    triple score_list
    """
    test_num = len(head_candidates)
    child_vecs = vocab.getMultiVecs(children)
    triple_score_list = [] # (h,p,c) similarity
    hp_score_list = [] # (h, p) similarity
    hc_score_list = [] # (h, c) similarity
    for test_ind in range(test_num):
        cands = head_candidates[test_ind]
        cand_vecs = vocab.getMultiVecs(cands)
        prep = prep_words[test_ind]
        try:
            prep_vec = prep_vecs[prep_set.index(prep)]
        except:
            prep_vec = [0] * 300
        triple_scores = []
        hp_scores = []
        hc_scores = []
        word_num = len(cands)
        for word_ind in range(word_num):
            head_vec = cand_vecs[word_ind]
            child_vec = child_vecs[test_ind]
            triple_score = getTripleSim(head_vec, prep_vec, child_vec)
            hp_score = getSim(head_vec, prep_vec)
            hc_score = getSim(head_vec, child_vec)
            triple_scores.append(triple_score)
            hp_scores.append(hp_score)
            hc_scores.append(hc_score)
        triple_score_list.append(triple_scores[:])
        hp_score_list.append(hp_scores[:])
        hc_score_list.append(hc_scores[:])
    return triple_score_list, hp_score_list, hc_score_list
            
    
    
def labelCand(cands_list, head_list):
    """
    label tuple (head, prep, word) with 1 (isHead) or 0 (notHead) 
    """
    test_num = len(cands_list)
    label_list = []
    for ind in range(test_num):
        cands = cands_list[ind]
        head = head_list[ind]
        labels = [int(cand == head) for cand in cands]
        label_list.append(labels[:])
    return label_list


def transformFeatureLabel(vocab, head_candidates, cand_dist, gold_heads, prep_words, children, \
                          head_pos_tags, next_pos_tags, cand_distance, all_prep_set, prep_vecs):
    """
    feature: score, pos, distance, prep one hot
    label
    """
    inst_num = len(head_candidates)
    feature_list = []
    label_list = []
    score_list = []

    # get scores for each tuple (head, prep, word) -- each sentence
    triple_score_list, hp_score_list, hc_score_list = scoreHead(vocab, head_candidates, prep_words, children, all_prep_set, prep_vecs)
    # label for each sentence
    label_list = labelCand(head_candidates, gold_heads)
    # train features: each sentence
    feature_list = []
    for inst_ind in range(inst_num):
        # scores
        triple_scores = triple_score_list[inst_ind]
        hp_scores = hp_score_list[inst_ind]
        hc_scores = hc_score_list[inst_ind]
        # other features
        tags = head_pos_tags[inst_ind]
        next_tags = next_pos_tags[inst_ind]
        distances = cand_distance[inst_ind]
        prep = prep_words[inst_ind]
        prep_onehot = [int(p==prep) for p in all_prep_set]
        word_num = len(triple_scores)
	features = []
        for word_ind in range(word_num):
            triple_score = triple_scores[word_ind]
            hp_score = hp_scores[word_ind]
            hc_score = hc_scores[word_ind]
            tag = tags[word_ind]
            next_tag = next_tags[word_ind]
            dist = distances[word_ind]
            feature = [triple_score, hp_score, hc_score, tag, dist] + prep_onehot + next_tag
            features.append(feature[:])
        feature_list.append(features[:])
    print "feature dimension:", len(feature_list[0][0])
    return feature_list, label_list
        

def processData(data_folder, prefix, vocab, all_prep_set, all_tag_set):
    print "prep set size", len(all_prep_set)
    # read prep
    prep_words,  _ = readPrep(data_folder+prefix+"preps.words")
    # prep vecs
    prep_vecs = vocab.getPrepVecs(all_prep_set)
    # read children
    children = readChildren(data_folder+prefix+"children.words")
    # read head candidates
    head_candidates, cand_distance = readHeadCandidates(data_folder+prefix+"heads.words")
    # read pos tag of candidates
    head_pos_tags = readWordPOS(data_folder+prefix+"heads.pos")
    # read pos tag of next word
    next_pos_tags = readNextPOS(data_folder+prefix+"heads.next.pos", all_tag_set)
    # read gold heads
    gold_heads = readGoldHead(head_candidates, data_folder+prefix+"labels")
    # feature, label
    features, labels = transformFeatureLabel(vocab, head_candidates, cand_distance, gold_heads, prep_words, children, \
                          head_pos_tags, next_pos_tags, cand_distance, all_prep_set, prep_vecs)

    return features, labels
    

def cleanData(pre_prep_fn, train_prep_fn, test_prep_fn):
    """
    pre_prep_fn: prep_49
    train_prep_fn: prep.words
    return selected inds and prep_set
    """
    # prep_49
    f = open(pre_prep_fn, "r")
    prep_set = [line.strip() for line in f.readlines()]
    f.close()

    # train_prep
    f = open(train_prep_fn, "r")
    train_prep_list = [line.strip() for line in f.readlines()]
    train_prep_set = set(train_prep_list)
    f.close()

    # test_prep
    f = open(test_prep_fn, "r")
    test_prep_list = [line.strip() for line in f.readlines()]
    test_prep_set = set(test_prep_list)
    f.close()

    selected_prep = [prep for prep in prep_set if (prep in train_prep_set or prep in test_prep_set)]
    train_selected_inds = [ind for ind in range(len(train_prep_list)) if train_prep_list[ind] in selected_prep]
    test_selected_inds = [ind for ind in range(len(test_prep_list)) if test_prep_list[ind] in selected_prep]

    print "# of selected_prep:", len(selected_prep)
    print "train total size:", len(train_prep_list), " selected size:", len(train_selected_inds)
    print "test total size:", len(test_prep_list), " selected size:", len(test_selected_inds)

    return selected_prep, train_selected_inds, test_selected_inds



if __name__=="__main__":
    data_folder = "../pp-attachment/data/pp-data-english/"
    train_prefix = "wsj.2-21.txt.dep.pp."
    test_prefix = "wsj.23.txt.dep.pp."

    
    # read prep
    _, train_prep_set = readPrep(data_folder+train_prefix+"preps.words")
    _, test_prep_set = readPrep(data_folder+test_prefix+"preps.words")
    all_prep_set = list(set(train_prep_set+test_prep_set))

    # read pos tags
    train_tag_set = readTags(data_folder+train_prefix+"heads.next.pos")
    test_tag_set = readTags(data_folder+test_prefix+"heads.next.pos")
    all_tag_set = list(set(train_tag_set+test_tag_set))
    print "# of tags:", len(all_tag_set)
    

    # load vocab parameters
    vec_folder = "tensor_prepNum=76/"
    word_vec_fn = (vec_folder+"mode1.mat", vec_folder+"mode2.mat")
    vocab_fn = "dump/vocab.txt"
    selected_ind_fn = vec_folder+"mode1.map"
    isAvg = False
    z_vec_fn = vec_folder+"mode3.mat"
    z_vocab_fn = "dump/prepositions_76.txt"
    vocab = Vocab(word_vec_fn, vocab_fn, selected_ind_fn, isAvg,\
                  z_vec_fn, z_vocab_fn)

    # train_features, train_labels
    print "training data..."
    train_dump_path = "../data/train_dump"
    train_features, train_labels = processData(data_folder, train_prefix, vocab, all_prep_set, all_tag_set)
    # select train data
    #train_features = [train_features[ind] for ind in range(len(train_features)) if ind in train_selected_inds]
    #train_labels = [train_labels[ind] for ind in range(len(train_labels)) if ind in train_selected_inds]
    joblib.dump((train_features, train_labels), train_dump_path)


    # test_features, test_labels
    print "test data..."
    test_dump_path = "../data/test_dump"
    test_features, test_labels = processData(data_folder, test_prefix, vocab, all_prep_set, all_tag_set)
    joblib.dump((test_features, test_labels), test_dump_path)
    
    
    








    
