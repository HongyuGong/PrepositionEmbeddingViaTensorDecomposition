"""
process test data from data/cikm2014/CoNLL.txt and
process test data from data/cikm2014/stackexchange.txt
"""
import nltk
from sklearn.metrics import classification_report

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


def readCorrectedData(fn):
    """
    fn: 
    return prep_sent_list, correct_prep_list
    """
    prep_list = getPrepList()
    orig_prep_list = []
    sent_list = []
    correct_prep_list = []
    #f = open(cikm_folder+fn, "r")
    f = open(fn, "r")
    for line in f:
        #seq = line.strip().lower().split()
        line = line.strip().lower()
        seq = nltk.word_tokenize(line)
        for ind in range(len(seq)):
            word = seq[ind]
            # prep with correction
            if ("*/" in word):
                prep_pair = word.split("*/")
                prep, prep_cor = prep_pair
		if ((prep not in prep_list) or (prep_cor not in prep_list)):
		    continue
	        prep = prep.lower()
	        prep_cor = prep_cor.lower()
                sent = seq[:]
                sent[ind] = "<b> "+prep+" </b>"
                orig_prep_list.append(prep)
                sent_list.append(" ".join(sent))
                correct_prep_list.append(prep_cor)
    f.close()
    print "%d prepositions in %s test dataset" % (len(sent_list), fn)
    return orig_prep_list, sent_list, correct_prep_list


def readData(fn):
    """
    fn: 
    return prep_sent_list, correct_prep_list
    """
    prep_list = getPrepList()
    orig_prep_list = []
    sent_list = []
    correct_prep_list = []
    f = open(fn, "r")
    for line in f:
        #seq = line.strip().lower().split()
        line = line.decode('utf8').strip().lower()
        seq = nltk.word_tokenize(line)
        for ind in range(len(seq)):
            word = seq[ind]
            # prep without correction
            if (word in prep_list):
                sent = seq[:]
                prep = word
                sent[ind] = "<b> "+word+" </b>"
                orig_prep_list.append(prep)
                sent_list.append(" ".join(sent))
                correct_prep_list.append(prep)
            # prep with correction
            elif ("*/" in word):
                prep_pair = word.split("*/")
                prep, prep_cor = prep_pair
		if ((prep not in prep_list) or (prep_cor not in prep_list)):
		    continue
	        prep = prep.lower()
	        prep_cor = prep_cor.lower()
                sent = seq[:]
                sent[ind] = "<b> "+prep+" </b>"
                orig_prep_list.append(prep)
                sent_list.append(" ".join(sent))
                correct_prep_list.append(prep_cor)
    f.close()
    print "%d instances in %s test dataset" % (len(sent_list), fn)
    return orig_prep_list, sent_list, correct_prep_list


def evalDetection(gold_res, algo_res):
     print(classification_report(gold_res, algo_res))


def evalSelection(orig_prep_list, algo_prep_list, correct_prep_list):
    """
    prec = valid suggested corrections / total suggested corrections
    recall = valid suggested corrections / total valid corrections
    f1-score = 2*precision*recall/(precision*recall)
    """
    print "orig == correct == prep", len(orig_prep_list)==len(correct_prep_list), \
          len(correct_prep_list)==len(algo_prep_list)
    prep_list = getPrepList()
    correct_prep_count = dict()
    valid_change_count = dict()
    
    suggested_change_count = dict()
    valid_suggested_change_count = dict()

    for prep in prep_list:
	correct_prep_count[prep] = 0
	valid_change_count[prep] = 0
	suggested_change_count[prep] = 0
	valid_suggested_change_count[prep] = 0
	
    valid_corrections = 0
    suggest_corrections = 0
    valid_suggest_corrections = 0
    
    for ind in range(len(orig_prep_list)):
        correct_prep_count[correct_prep_list[ind]] += 1
        if (orig_prep_list[ind] != correct_prep_list[ind]):
	    valid_change_count[correct_prep_list[ind]] += 1
            valid_corrections += 1

    for ind in range(len(algo_prep_list)):
        if (orig_prep_list[ind] != algo_prep_list[ind]):
            suggest_corrections += 1
            suggested_change_count[correct_prep_list[ind]] += 1
            if (algo_prep_list[ind] == correct_prep_list[ind]):
                valid_suggest_corrections += 1
                valid_suggested_change_count[correct_prep_list[ind]] += 1

    prec = float(valid_suggest_corrections) / suggest_corrections
    recall = float(valid_suggest_corrections) / valid_corrections
    fscore = 2 * prec * recall / (prec + recall)

    print "precision: %f, recall: %f, fscore: %f" % (prec, recall, fscore)

    # write result analysis into file
    f = open("logs.txt", "w")
    print >> f, "\t".join(["preposition", "total count", "valid change", "non-change", \
                          "suggested count", "valid suggested count", "wrong suggested count"])
    for prep in prep_list:
	try:
            c1 = correct_prep_count[prep]
            c2 = valid_change_count[prep]
            c3 = c1 -c2
            c4 = suggested_change_count[prep]
            c5 = valid_suggested_change_count[prep]
            c6 = c4 - c5
	    print >> f, "\t".join([prep, str(c1), str(c2), str(c3), str(c4), str(c5), str(c6)])
	except:
	    print >> f, "\t".join([prep, str(correct_prep_count[prep]), str(correct_prep_count[prep])])
    f.close()







